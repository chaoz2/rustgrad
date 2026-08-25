//! RGBS: portable initial host-state bootstrap around canonical RGMB bytes.
//!
//! This envelope packages only the earliest exact state needed to begin a
//! captured mixed batch.  It deliberately does not change RGMB/RGSM, encode a
//! destination namespace, or retain any runtime resource/current-state data.
use super::{CapturedMixedBatch, MixedBatchArtifactError};
use crate::engine::mixed_capture::CapturedMixedSchedule;
use crate::tensor::artifact::{decode_from as decode_tensor, encode_into as encode_tensor};
use crate::uop::artifact::{
    ArtifactError, Reader, Writer, checksum, read_buffer_state, write_buffer_state,
};
use crate::{
    BufferState, CapturedReplayExecutor, EffectBatchStep, EffectRuntime, MixedStateRebinding,
    ReplayError, RuntimeError, TensorData,
};
use std::collections::BTreeMap;

const MAGIC: &[u8; 4] = b"RGBS";
const VERSION: u8 = 1;
const MAX_BYTES: usize = 64 << 20;
const MAX_STATES: usize = 1 << 16;

/// A raw, immutable persistent snapshot keyed by its captured logical state.
#[derive(Clone, Debug)]
pub struct PortableMixedState {
    state: BufferState,
    value: TensorData,
}

impl PortableMixedState {
    /// Creates one exact initial-state snapshot. Its relationship to a batch
    /// frontier is checked by [`CapturedMixedStateBundle::new`].
    pub fn new(state: BufferState, value: TensorData) -> Self {
        Self { state, value }
    }

    pub fn state(&self) -> &BufferState {
        &self.state
    }

    pub fn tensor(&self) -> &TensorData {
        &self.value
    }
}

/// Structured portable bootstrap failures. These errors always arise before
/// a runtime slot, device resource, or replay execution is observed.
#[derive(Debug)]
pub enum MixedStateBundleError {
    Corrupt(&'static str),
    Batch(MixedBatchArtifactError),
    Tensor(ArtifactError),
    Replay(ReplayError),
    Runtime(RuntimeError),
}

impl std::fmt::Display for MixedStateBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RGBS bundle: {self:?}")
    }
}
impl std::error::Error for MixedStateBundleError {}
impl From<MixedBatchArtifactError> for MixedStateBundleError {
    fn from(value: MixedBatchArtifactError) -> Self {
        Self::Batch(value)
    }
}
impl From<ArtifactError> for MixedStateBundleError {
    fn from(value: ArtifactError) -> Self {
        Self::Tensor(value)
    }
}
impl From<ReplayError> for MixedStateBundleError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}
impl From<RuntimeError> for MixedStateBundleError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

/// A portable logical mixed batch and the exact immutable host-state frontier
/// required to instantiate it in a fresh [`EffectRuntime`].
#[derive(Clone, Debug)]
pub struct CapturedMixedStateBundle {
    batch: CapturedMixedBatch,
    snapshots: Vec<PortableMixedState>,
}

/// A batch paired with one caller-selected runtime namespace after its
/// snapshots were atomically registered. It owns neither the runtime nor any
/// runtime resource identity.
#[derive(Clone, Debug)]
pub struct InstantiatedMixedBatch {
    batch: CapturedMixedBatch,
    rebindings: Vec<MixedStateRebinding>,
}

impl InstantiatedMixedBatch {
    pub fn batch(&self) -> &CapturedMixedBatch {
        &self.batch
    }

    pub fn rebindings(&self) -> &[MixedStateRebinding] {
        &self.rebindings
    }

    pub fn replay(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<Vec<BufferState>, MixedStateBundleError> {
        Ok(self.batch.replay_with_rebindings(
            runtime,
            inputs,
            &self.rebindings,
            injected_failure,
        )?)
    }

    pub fn replay_native(
        &self,
        runtime: &mut EffectRuntime,
        inputs: &[BTreeMap<String, TensorData>],
        executor: &CapturedReplayExecutor,
        vectorized: bool,
        injected_failure: Option<EffectBatchStep>,
    ) -> Result<Vec<BufferState>, MixedStateBundleError> {
        Ok(self.batch.replay_native_with_rebindings(
            runtime,
            inputs,
            &self.rebindings,
            executor,
            vectorized,
            injected_failure,
        )?)
    }
}

impl CapturedMixedStateBundle {
    /// Validates the exact minimal initial-state frontier for this immutable
    /// RGMB batch. State values are raw owned storage, never runtime snapshots.
    pub fn new(
        batch: CapturedMixedBatch,
        snapshots: Vec<PortableMixedState>,
    ) -> Result<Self, MixedStateBundleError> {
        validate_frontier(&batch, &snapshots)?;
        Ok(Self { batch, snapshots })
    }

    pub fn batch(&self) -> &CapturedMixedBatch {
        &self.batch
    }

    pub fn snapshots(&self) -> &[PortableMixedState] {
        &self.snapshots
    }

    /// Encodes canonical RGMB bytes followed by the exact initial raw storage
    /// frontier. The embedded RGMB stream is copied byte-for-byte.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MixedStateBundleError> {
        validate_frontier(&self.batch, &self.snapshots)?;
        let rgmb = self.batch.to_bytes()?;
        if rgmb.len() > MAX_BYTES || self.snapshots.len() > MAX_STATES {
            return Err(MixedStateBundleError::Corrupt("byte/count limit"));
        }
        let mut writer = Writer::new();
        writer.bytes(MAGIC)?;
        writer.u8(VERSION)?;
        writer.u32(
            u32::try_from(rgmb.len()).map_err(|_| MixedStateBundleError::Corrupt("RGMB length"))?,
        )?;
        writer.bytes(&rgmb)?;
        writer.u32(
            u32::try_from(self.snapshots.len())
                .map_err(|_| MixedStateBundleError::Corrupt("state count"))?,
        )?;
        for snapshot in &self.snapshots {
            write_buffer_state(&mut writer, &snapshot.state)?;
            encode_tensor(&mut writer, &snapshot.value)?;
        }
        if writer
            .out
            .len()
            .checked_add(4)
            .is_none_or(|n| n > MAX_BYTES)
        {
            return Err(MixedStateBundleError::Corrupt("byte limit"));
        }
        let sum = checksum(&writer.out);
        writer.u32(sum)?;
        Ok(writer.out)
    }

    /// Decodes and validates a complete bundle before any runtime registration.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MixedStateBundleError> {
        if bytes.len() < 17 || bytes.len() > MAX_BYTES {
            return Err(MixedStateBundleError::Corrupt("length"));
        }
        let body = bytes.len() - 4;
        let stored = u32::from_le_bytes(
            bytes[body..]
                .try_into()
                .map_err(|_| MixedStateBundleError::Corrupt("checksum"))?,
        );
        if checksum(&bytes[..body]) != stored {
            return Err(MixedStateBundleError::Corrupt("checksum"));
        }
        let mut reader = Reader::new(&bytes[..body]);
        if reader.take(4)? != MAGIC {
            return Err(MixedStateBundleError::Corrupt("magic"));
        }
        if reader.u8()? != VERSION {
            return Err(MixedStateBundleError::Corrupt("version"));
        }
        let rgmb_len = reader.u32()? as usize;
        if rgmb_len == 0 || rgmb_len > MAX_BYTES {
            return Err(MixedStateBundleError::Corrupt("RGMB length"));
        }
        let batch = CapturedMixedBatch::from_bytes(reader.take(rgmb_len)?)?;
        let count = reader.count(MAX_STATES)?;
        if count == 0 {
            return Err(MixedStateBundleError::Corrupt("state count"));
        }
        let mut snapshots = Vec::with_capacity(count);
        for _ in 0..count {
            snapshots.push(PortableMixedState {
                state: read_buffer_state(&mut reader)?,
                value: decode_tensor(&mut reader)?,
            });
        }
        if !reader.done() {
            return Err(MixedStateBundleError::Corrupt("trailing bytes"));
        }
        Self::new(batch, snapshots)
    }

    /// Validates all caller-selected destination names and atomically registers
    /// every exact raw snapshot into `runtime`. A failed validation or lease
    /// setup leaves the runtime's logical namespace unchanged.
    pub fn instantiate(
        &self,
        runtime: &mut EffectRuntime,
        rebindings: &[MixedStateRebinding],
    ) -> Result<InstantiatedMixedBatch, MixedStateBundleError> {
        // Rebuild once as a complete validation pass, but retain canonical
        // captured bytes in the ready handle so every replay uses the existing
        // `*_with_rebindings` entry point exactly once.
        self.batch.rebound(rebindings)?;
        let frontier = frontier(&self.batch)?;
        let values = self
            .snapshots
            .iter()
            .map(|snapshot| (snapshot.state.buffer, snapshot.value.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut destinations = BTreeMap::<u64, (u64, TensorData)>::new();
        for (capture, rebinding) in self.batch.captures().iter().zip(rebindings) {
            for state in capture.initial_states() {
                let expected = frontier
                    .get(&state.buffer)
                    .ok_or(MixedStateBundleError::Corrupt("frontier state"))?;
                if state != expected {
                    return Err(MixedStateBundleError::Corrupt("non-frontier start"));
                }
                let destination = rebinding.destination(state.buffer)?;
                let value = values
                    .get(&state.buffer)
                    .ok_or(MixedStateBundleError::Corrupt("snapshot state"))?
                    .clone();
                match destinations.entry(destination) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((state.buffer, value));
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get().0 == state.buffer
                            && raw_tensor_equal(&entry.get().1, &value)? => {}
                    std::collections::btree_map::Entry::Occupied(_) => {
                        return Err(MixedStateBundleError::Corrupt("destination collision"));
                    }
                }
            }
        }
        runtime.register_initial_states(
            destinations
                .into_iter()
                .map(|(destination, (_, value))| (destination, value))
                .collect(),
        )?;
        Ok(InstantiatedMixedBatch {
            batch: self.batch.clone(),
            rebindings: rebindings.to_vec(),
        })
    }
}

fn raw_tensor_equal(left: &TensorData, right: &TensorData) -> Result<bool, MixedStateBundleError> {
    let mut left_bytes = Writer::new();
    let mut right_bytes = Writer::new();
    encode_tensor(&mut left_bytes, left)?;
    encode_tensor(&mut right_bytes, right)?;
    Ok(left_bytes.out == right_bytes.out)
}

fn frontier(
    batch: &CapturedMixedBatch,
) -> Result<BTreeMap<u64, BufferState>, MixedStateBundleError> {
    let mut out = BTreeMap::new();
    for capture in batch.captures() {
        collect_frontier(capture, &mut out)?;
    }
    if out.is_empty() {
        return Err(MixedStateBundleError::Corrupt("empty state frontier"));
    }
    if out.len() > MAX_STATES {
        return Err(MixedStateBundleError::Corrupt("state count"));
    }
    Ok(out)
}

fn collect_frontier(
    capture: &CapturedMixedSchedule,
    out: &mut BTreeMap<u64, BufferState>,
) -> Result<(), MixedStateBundleError> {
    for state in &capture.states {
        match out.get(&state.buffer) {
            None => {
                out.insert(state.buffer, state.clone());
            }
            Some(existing) if state.version < existing.version => {
                out.insert(state.buffer, state.clone());
            }
            Some(existing)
                if existing.shape != state.shape
                    || existing.dtype != state.dtype
                    || existing.bytes != state.bytes =>
            {
                return Err(MixedStateBundleError::Corrupt("state descriptor"));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

fn validate_frontier(
    batch: &CapturedMixedBatch,
    snapshots: &[PortableMixedState],
) -> Result<(), MixedStateBundleError> {
    let expected = frontier(batch)?;
    if snapshots.len() != expected.len() {
        return Err(MixedStateBundleError::Corrupt("frontier count"));
    }
    let supplied = snapshots
        .iter()
        .map(|snapshot| (snapshot.state.buffer, snapshot))
        .collect::<BTreeMap<_, _>>();
    if supplied.len() != snapshots.len() || supplied.len() != expected.len() {
        return Err(MixedStateBundleError::Corrupt("duplicate state"));
    }
    for (buffer, state) in expected {
        if state.version != 0 {
            return Err(MixedStateBundleError::Corrupt(
                "noninitial frontier version",
            ));
        }
        let snapshot = supplied
            .get(&buffer)
            .ok_or(MixedStateBundleError::Corrupt("missing state"))?;
        if snapshot.state != state
            || snapshot.value.shape() != &state.shape
            || snapshot.value.dtype() != state.dtype
            || snapshot
                .value
                .len()
                .checked_mul(snapshot.value.dtype().itemsize())
                != Some(state.bytes)
        {
            return Err(MixedStateBundleError::Corrupt("state descriptor"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mixed_batch::test_support;
    use crate::{
        CapturedSchedule, DType, EffectBatchStep, EffectGraph, Shape, Storage, schedule_effects,
    };
    use std::sync::Arc;

    fn zero(dtype: DType, len: usize) -> Storage {
        match dtype {
            DType::Bool => Storage::Bool(vec![false; len]),
            DType::I8 => Storage::I8(vec![0; len]),
            DType::U8 => Storage::U8(vec![0; len]),
            dtype if dtype.is_float8() => Storage::Float8(crate::Float8Storage::from_raw(
                dtype.float8_format().expect("float8 dtype"),
                vec![0; len],
            )),
            DType::I16 => Storage::I16(vec![0; len]),
            DType::U16 => Storage::U16(vec![0; len]),
            DType::I32 => Storage::I32(vec![0; len]),
            DType::U32 => Storage::U32(vec![0; len]),
            DType::I64 => Storage::I64(vec![0; len]),
            DType::U64 => Storage::U64(vec![0; len]),
            DType::F16 => Storage::F16(vec![0; len]),
            DType::BF16 => Storage::BF16(vec![0; len]),
            DType::F32 => Storage::F32(vec![0.; len]),
            DType::F64 => Storage::F64(vec![0.; len]),
            DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ => {
                unreachable!("float8 handled by the transport guard")
            }
        }
    }

    fn bootstrap(batch: CapturedMixedBatch) -> CapturedMixedStateBundle {
        let snapshots = frontier(&batch)
            .unwrap()
            .into_values()
            .map(|state| {
                let tensor = TensorData::from_storage(
                    state.shape.clone(),
                    zero(state.dtype, state.shape.numel().unwrap()),
                )
                .unwrap();
                PortableMixedState::new(state, tensor)
            })
            .collect();
        CapturedMixedStateBundle::new(batch, snapshots).unwrap()
    }

    fn mapping(capture: &CapturedMixedSchedule, offset: u64) -> MixedStateRebinding {
        MixedStateRebinding::new(
            capture
                .states
                .iter()
                .map(|state| (state.buffer, state.buffer + offset))
                .collect(),
        )
        .unwrap()
    }

    fn effect_only(storage: Storage) -> CapturedMixedBatch {
        let shape = Shape::from([storage.len()]);
        let value = TensorData::from_storage(shape, storage).unwrap();
        let mut graph = EffectGraph::default();
        let target = graph.insert(31, value.clone()).unwrap();
        let source = graph.insert(32, value).unwrap();
        let next = graph.assign(&target, &source).unwrap();
        let schedule = schedule_effects(&graph).unwrap();
        let capture = CapturedMixedSchedule::from_parts(
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
        CapturedMixedBatch::new(vec![capture]).unwrap()
    }

    fn indexed_f16_batch() -> (CapturedMixedBatch, BufferState) {
        use crate::ir::indexing::{StaticIndex, StaticIndexPlan};
        let mut graph = EffectGraph::default();
        let target = graph
            .insert(
                71,
                TensorData::from_storage([3], Storage::F16(vec![1, 0x8000, 3])).unwrap(),
            )
            .unwrap();
        let source = graph
            .insert(
                72,
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
        let next = graph.static_index_assign(&target, &source, plan).unwrap();
        let schedule = schedule_effects(&graph).unwrap();
        let capture = CapturedMixedSchedule::from_parts(
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
        (
            CapturedMixedBatch::new(vec![capture]).unwrap(),
            next.state().clone(),
        )
    }

    #[test]
    fn rgbs_round_trips_raw_bootstrap_storage_without_changing_rgmb() {
        let cases = [
            Storage::Bool(vec![false, true]),
            Storage::I8(vec![i8::MIN, i8::MAX]),
            Storage::U8(vec![0, u8::MAX]),
            Storage::I16(vec![i16::MIN, i16::MAX]),
            Storage::U16(vec![0, u16::MAX]),
            Storage::I32(vec![i32::MIN, i32::MAX]),
            Storage::U32(vec![0, u32::MAX]),
            Storage::I64(vec![i64::MIN, i64::MAX]),
            Storage::U64(vec![u64::MAX]),
            Storage::F16(vec![0x8000, 0x7e01]),
            Storage::BF16(vec![0x8000, 0x7fc1]),
            Storage::F32(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_0123),
            ]),
            Storage::F64(vec![
                f64::from_bits(0x8000_0000_0000_0000),
                f64::from_bits(0x7ff8_0000_0000_0123),
            ]),
        ];
        for storage in cases {
            let batch = effect_only(storage.clone());
            let expected_rgmb = batch.to_bytes().unwrap();
            let bundle = CapturedMixedStateBundle::new(
                batch,
                frontier(&effect_only(storage.clone()))
                    .unwrap()
                    .into_values()
                    .map(|state| {
                        PortableMixedState::new(
                            state.clone(),
                            TensorData::from_storage(state.shape.clone(), storage.clone()).unwrap(),
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let bytes = bundle.to_bytes().unwrap();
            assert_eq!(bytes, bundle.to_bytes().unwrap());
            let decoded = CapturedMixedStateBundle::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.batch().to_bytes().unwrap(), expected_rgmb);
            // Re-encoding rather than `PartialEq` is intentional: NaN payloads
            // are part of this raw-storage contract.
            assert_eq!(decoded.to_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn rgbs_instantiates_fresh_runtime_for_interpreter_native_and_ptx_mock() {
        let (capture, end) = test_support::pure_add_capture(800);
        let batch = CapturedMixedBatch::new(vec![capture.clone()]).unwrap();
        let bundle = bootstrap(batch);
        let bytes = bundle.to_bytes().unwrap();
        let decoded = CapturedMixedStateBundle::from_bytes(&bytes).unwrap();
        let inputs = [test_support::add_inputs()];

        let mut interpreter = EffectRuntime::new();
        let ready = decoded
            .instantiate(&mut interpreter, &[mapping(&capture, 1_000)])
            .unwrap();
        ready.replay(&mut interpreter, &inputs, None).unwrap();
        let expected = Storage::F32(vec![4., 6.]);
        let rebound_end = BufferState {
            buffer: end.buffer + 1_000,
            ..end.clone()
        };
        assert_eq!(
            interpreter
                .snapshot(&rebound_end)
                .unwrap()
                .tensor()
                .storage(),
            &expected
        );

        let mut native_runtime = EffectRuntime::new();
        let native_ready = decoded
            .instantiate(&mut native_runtime, &[mapping(&capture, 2_000)])
            .unwrap();
        native_ready
            .replay_native(
                &mut native_runtime,
                &inputs,
                &CapturedReplayExecutor::default(),
                false,
                None,
            )
            .unwrap();
        assert_eq!(
            native_runtime
                .snapshot(&BufferState {
                    buffer: end.buffer + 2_000,
                    ..end.clone()
                })
                .unwrap()
                .tensor()
                .storage(),
            &expected
        );

        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = crate::Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut ptx_runtime = EffectRuntime::new();
        let ptx_ready = decoded
            .instantiate(&mut ptx_runtime, &[mapping(&capture, 3_000)])
            .unwrap();
        ptx_ready
            .batch()
            .replay_ptx_with_rebindings(
                &mut ptx_runtime,
                &inputs,
                ptx_ready.rebindings(),
                primary,
                crate::PtxRenderer::new(80).unwrap(),
                None,
            )
            .unwrap();
        assert_eq!(
            ptx_runtime
                .snapshot(&BufferState {
                    buffer: end.buffer + 3_000,
                    ..end
                })
                .unwrap()
                .tensor()
                .storage(),
            &expected
        );
        assert!(mock.calls().contains(&"launch"));
    }

    #[test]
    fn rgbs_bootstraps_independent_two_buffer_version_chains() {
        let (left, left_end) = test_support::pure_add_capture(811);
        let (right, right_end) = test_support::pure_add_capture(822);
        let bundle = bootstrap(
            CapturedMixedBatch::new(vec![
                left.clone(),
                right.clone(),
                left.clone(),
                right.clone(),
            ])
            .unwrap(),
        );
        let mut runtime = EffectRuntime::new();
        let ready = bundle
            .instantiate(
                &mut runtime,
                &[
                    mapping(&left, 1_000),
                    mapping(&right, 2_000),
                    mapping(&left, 1_000),
                    mapping(&right, 2_000),
                ],
            )
            .unwrap();
        ready
            .replay(
                &mut runtime,
                &[
                    test_support::add_inputs(),
                    test_support::add_inputs(),
                    test_support::add_inputs(),
                    test_support::add_inputs(),
                ],
                None,
            )
            .unwrap();
        for (end, offset) in [(left_end, 1_000), (right_end, 2_000)] {
            assert_eq!(
                runtime
                    .snapshot(&BufferState {
                        buffer: end.buffer + offset,
                        version: 2,
                        ..end
                    })
                    .unwrap()
                    .tensor()
                    .storage(),
                &Storage::F32(vec![4., 6.])
            );
        }
    }

    #[test]
    fn rgbs_replays_signed_affine_state_input_after_fresh_bootstrap() {
        let (capture, end) = test_support::signed_state_add_capture();
        let mut bundle = bootstrap(CapturedMixedBatch::new(vec![capture.clone()]).unwrap());
        let target = capture
            .initial_states()
            .find(|state| state.buffer == 90)
            .unwrap()
            .clone();
        bundle
            .snapshots
            .iter_mut()
            .find(|snapshot| snapshot.state == target)
            .unwrap()
            .value = TensorData::from_storage([4], Storage::F32(vec![1., 2., 3., 4.])).unwrap();
        let mut runtime = EffectRuntime::new();
        let ready = bundle
            .instantiate(&mut runtime, &[mapping(&capture, 5_000)])
            .unwrap();
        ready
            .replay(
                &mut runtime,
                &[BTreeMap::from([(
                    "bias".into(),
                    TensorData::from_storage([4], Storage::F32(vec![10., 20., 30., 40.])).unwrap(),
                )])],
                None,
            )
            .unwrap();
        assert_eq!(
            runtime
                .snapshot(&BufferState {
                    buffer: end.buffer + 5_000,
                    ..end
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![14., 23., 32., 41.])
        );
    }

    #[test]
    fn rgbs_bootstraps_zero_byte_frontier_without_reusable_state_slots() {
        let (capture, end) = test_support::zero_extent_add_capture();
        let bundle = bootstrap(CapturedMixedBatch::new(vec![capture.clone()]).unwrap());
        let mut runtime = EffectRuntime::new();
        let ready = bundle
            .instantiate(&mut runtime, &[mapping(&capture, 5_500)])
            .unwrap();
        assert_eq!(runtime.stats().unwrap().zero_byte_sentinels, 2);
        ready
            .replay(
                &mut runtime,
                &[BTreeMap::from([
                    (
                        "x".into(),
                        TensorData::from_storage([0], Storage::F32(vec![])).unwrap(),
                    ),
                    (
                        "y".into(),
                        TensorData::from_storage([0], Storage::F32(vec![])).unwrap(),
                    ),
                ])],
                None,
            )
            .unwrap();
        assert_eq!(
            runtime
                .snapshot(&BufferState {
                    buffer: end.buffer + 5_500,
                    ..end
                })
                .unwrap()
                .tensor()
                .len(),
            0
        );
    }

    #[test]
    fn rgbs_replays_indexed_raw_f16_with_duplicate_last_writer_and_retry() {
        let (batch, end) = indexed_f16_batch();
        let capture = batch.captures()[0].clone();
        let mut bundle = bootstrap(batch);
        for snapshot in &mut bundle.snapshots {
            snapshot.value = match snapshot.state.buffer {
                71 => TensorData::from_storage([3], Storage::F16(vec![1, 0x8000, 3])).unwrap(),
                72 => TensorData::from_storage([3], Storage::F16(vec![0x7e01, 7, 0x8000])).unwrap(),
                _ => unreachable!("indexed frontier"),
            };
        }
        let mut runtime = EffectRuntime::new();
        let ready = bundle
            .instantiate(&mut runtime, &[mapping(&capture, 6_000)])
            .unwrap();
        assert!(
            ready
                .replay(
                    &mut runtime,
                    &[BTreeMap::new()],
                    Some(EffectBatchStep { entry: 0, step: 0 }),
                )
                .is_err()
        );
        let target = BufferState {
            buffer: 6_071,
            ..capture
                .initial_states()
                .find(|state| state.buffer == 71)
                .unwrap()
                .clone()
        };
        assert_eq!(
            runtime.snapshot(&target).unwrap().tensor().storage(),
            &Storage::F16(vec![1, 0x8000, 3])
        );
        ready
            .replay(&mut runtime, &[BTreeMap::new()], None)
            .unwrap();
        assert_eq!(
            runtime
                .snapshot(&BufferState {
                    buffer: end.buffer + 6_000,
                    ..end
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F16(vec![1, 7, 0x8000])
        );
    }

    #[test]
    fn rgbs_rejects_corruption_and_transactionally_refuses_namespace_collisions() {
        let (capture, _) = test_support::pure_add_capture(900);
        let bundle = bootstrap(CapturedMixedBatch::new(vec![capture.clone()]).unwrap());
        let bytes = bundle.to_bytes().unwrap();
        for mut malformed in [
            bytes[..bytes.len() - 1].to_vec(),
            {
                let mut bad = bytes.clone();
                bad[0] ^= 1;
                bad
            },
            {
                let mut bad = bytes.clone();
                bad.push(0);
                bad
            },
        ] {
            if malformed.len() >= 4 {
                let body = malformed.len() - 4;
                if body > 0 {
                    let sum = checksum(&malformed[..body]);
                    malformed[body..].copy_from_slice(&sum.to_le_bytes());
                }
            }
            assert!(CapturedMixedStateBundle::from_bytes(&malformed).is_err());
        }
        let mut runtime = EffectRuntime::new();
        let rebound = mapping(&capture, 4_000);
        let collision = rebound.mappings().values().next().copied().unwrap();
        runtime
            .register(
                collision,
                TensorData::from_storage([2], Storage::F32(vec![0., 0.])).unwrap(),
            )
            .unwrap();
        let before = runtime.stats().unwrap();
        assert!(bundle.instantiate(&mut runtime, &[rebound]).is_err());
        assert_eq!(runtime.stats().unwrap(), before);
    }

    #[test]
    fn rgbs_rejects_nonfrontier_snapshot_tables_before_registration() {
        let (capture, _) = test_support::pure_add_capture(901);
        let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
        let bundle = bootstrap(batch.clone());
        let mut missing = bundle.snapshots.clone();
        missing.pop();
        assert!(CapturedMixedStateBundle::new(batch.clone(), missing).is_err());
        let mut duplicate = bundle.snapshots.clone();
        duplicate.push(duplicate[0].clone());
        assert!(CapturedMixedStateBundle::new(batch.clone(), duplicate).is_err());
        let mut mismatch = bundle.snapshots.clone();
        mismatch[0].state.bytes = mismatch[0].state.bytes.checked_add(4).unwrap();
        assert!(CapturedMixedStateBundle::new(batch, mismatch).is_err());
    }
}
