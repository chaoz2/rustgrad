//! Immutable capture boundary for mixed pure/effect schedules.
//!
//! This type intentionally owns only logical schedule/state metadata. Runtime
//! leases, slot generations, pointers, and current bytes remain caller-owned.
use super::NativeReplayTraffic;
use super::persistent_inputs::bind_persistent_inputs;
use crate::uop::artifact::{ArtifactError, Reader, Writer, checksum};
use crate::{
    BufferDesc, BufferState, CapturedSchedule, EffectPayload, MixedStateRebinding, NodeId,
    Operation, ReplayError, ReplayInput, Schedule, ScheduleStateBinding, ScheduleValueBinding, UOp,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

const MAGIC: &[u8; 4] = b"RGSM";
/// v3 adopts canonical schedule item/state-binding keys. v1-v2 retain opaque
/// historical keys and are upgraded only after their stored envelope passes.
const VERSION: u8 = 3;
const HEADER_LEN: usize = MAGIC.len() + 1 + std::mem::size_of::<u64>();
const MAX_BYTES: usize = 64 << 20;
const MAX_ITEMS: usize = 1 << 16;
const MAX_BINDINGS: usize = 1 << 16;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreparedReplayValidationCounts {
    pub(crate) mixed_capture_validations: usize,
    pub(crate) schedule_rekeys: usize,
    pub(crate) identity_serializations: usize,
    pub(crate) recurrent_frontier_plans: usize,
    pub(crate) cursor_projection_preparations: usize,
    pub(crate) recurrent_bank_layouts: usize,
}

#[cfg(test)]
std::thread_local! {
    static PREPARED_REPLAY_VALIDATION_COUNTS:
        std::cell::Cell<PreparedReplayValidationCounts> = const {
            std::cell::Cell::new(PreparedReplayValidationCounts {
                mixed_capture_validations: 0,
                schedule_rekeys: 0,
                identity_serializations: 0,
                recurrent_frontier_plans: 0,
                cursor_projection_preparations: 0,
                recurrent_bank_layouts: 0,
            })
        };
}

#[cfg(test)]
std::thread_local! {
    static INDEXED_RECURRENT_BANK_BINDINGS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
fn record_prepared_replay_validation(update: impl FnOnce(&mut PreparedReplayValidationCounts)) {
    PREPARED_REPLAY_VALIDATION_COUNTS.with(|counts| {
        let mut next = counts.get();
        update(&mut next);
        counts.set(next);
    });
}

#[cfg(test)]
pub(crate) fn reset_prepared_replay_validation_counts() {
    PREPARED_REPLAY_VALIDATION_COUNTS.with(|counts| counts.set(Default::default()));
    INDEXED_RECURRENT_BANK_BINDINGS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn prepared_replay_validation_counts() -> PreparedReplayValidationCounts {
    PREPARED_REPLAY_VALIDATION_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn indexed_recurrent_bank_binding_count() -> usize {
    INDEXED_RECURRENT_BANK_BINDINGS.with(std::cell::Cell::get)
}

fn requested_materializations(capture: &CapturedSchedule) -> Vec<u64> {
    crate::schedule::physical_requested_materializations(
        &capture.items,
        &capture.requested_passthroughs,
        capture.requested.iter().copied(),
    )
}

fn validate_requested_selection(all: &[u64], selected: &[u64]) -> Result<(), ReplayError> {
    let mut next = 0;
    for selected in selected {
        let Some(offset) = all[next..]
            .iter()
            .position(|candidate| candidate == selected)
        else {
            return Err(ReplayError::Corrupt(
                "selected recurrent output is not an ordered capture output".into(),
            ));
        };
        next = next
            .checked_add(offset + 1)
            .ok_or_else(|| ReplayError::Corrupt("selected output index overflow".into()))?;
    }
    Ok(())
}

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

#[derive(Debug)]
pub(crate) struct NativeMixedReplayResult {
    pub(crate) replay: MixedReplayResult,
    pub(crate) traffic: NativeReplayTraffic,
    pub(crate) executor_wall_time: Duration,
}

/// Logical persistent-state frontier for recurrent replay of one exact mixed
/// capture. The cursor owns only buffer/version tensor descriptors; runtime
/// slots, host pointers, storage generations, and tensor bytes remain owned by
/// [`crate::EffectRuntime`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedReplayCursor {
    capture_identity: u64,
    frontier: Vec<BufferState>,
}

impl MixedReplayCursor {
    /// Creates the initial cursor for `capture`. `frontier` must be the exact
    /// canonical version-zero state set required by its persistent reads and
    /// writes. Use [`CapturedMixedSchedule::initial_recurrent_cursor`] when the
    /// capture-declared frontier is desired directly.
    pub fn new(
        capture: &CapturedMixedSchedule,
        frontier: impl IntoIterator<Item = BufferState>,
    ) -> Result<Self, ReplayError> {
        validate(capture, true)?;
        let expected = recurrent_initial_frontier(capture)?;
        let frontier = canonical_frontier(frontier)?;
        if frontier != expected {
            return Err(ReplayError::Descriptor(
                "recurrent cursor initial state frontier mismatch".into(),
            ));
        }
        Ok(Self {
            capture_identity: identity(capture)?,
            frontier,
        })
    }

    /// Restores a canonical recurrent frontier at its saved logical versions.
    /// Every buffer and descriptor must match the capture's version-zero
    /// schema; versions may advance but cannot precede that schema.
    pub fn resume(
        capture: &CapturedMixedSchedule,
        frontier: impl IntoIterator<Item = BufferState>,
    ) -> Result<Self, ReplayError> {
        validate(capture, true)?;
        let cursor = Self {
            capture_identity: identity(capture)?,
            frontier: canonical_frontier(frontier)?,
        };
        validate_recurrent_cursor(capture, &cursor)?;
        Ok(cursor)
    }

    /// Stable logical RGSM identity accepted by this cursor.
    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    /// Canonical buffer-ordered persistent descriptors at the next replay.
    pub fn frontier(&self) -> &[BufferState] {
        &self.frontier
    }
}

/// Preparation-time proof that one recurrent capture is an exact descriptor-
/// preserving projection of another capture's canonical frontier. Runtime
/// tensor ownership, leases, generations, and pointers deliberately remain
/// outside this witness.
#[derive(Clone, Debug)]
pub(crate) struct PreparedRecurrentCursorProjection {
    source_capture_identity: u64,
    target_capture_identity: u64,
    source_schema: Box<[BufferState]>,
    target_schema: Box<[BufferState]>,
    target_to_source: Box<[usize]>,
}

/// One call-local projected cursor plus the prevalidated source-version
/// updates to publish after the target replay commits successfully.
pub(crate) struct ProjectedRecurrentCursor {
    cursor: MixedReplayCursor,
    source_updates: Box<[(usize, u64)]>,
}

/// Runtime projection failures whose session-facing compatibility differs
/// from general capture corruption. The witness remains engine-private while
/// callers can preserve their established error contract without inspecting
/// error strings.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum RecurrentCursorProjectionError {
    IncompleteSourceFrontier,
    VersionOverflow,
    InvalidSource(ReplayError),
}

impl PreparedRecurrentCursorProjection {
    pub(crate) fn prepare(
        source: &CapturedMixedSchedule,
        target: &CapturedMixedSchedule,
        target_buffers: impl IntoIterator<Item = u64>,
    ) -> Result<Self, ReplayError> {
        #[cfg(test)]
        record_prepared_replay_validation(|counts| counts.cursor_projection_preparations += 1);
        validate(source, true)?;
        validate(target, true)?;
        let source_capture_identity = identity(source)?;
        let target_capture_identity = identity(target)?;
        let source_schema = recurrent_initial_frontier(source)?;
        let target_schema = recurrent_initial_frontier(target)?;
        let mut target_buffer_set = BTreeSet::new();
        for buffer in target_buffers {
            if !target_buffer_set.insert(buffer) {
                return Err(ReplayError::Descriptor(
                    "prepared recurrent cursor projection target buffer repeats".into(),
                ));
            }
        }
        if target_buffer_set.len() != target_schema.len()
            || target_schema
                .iter()
                .any(|state| !target_buffer_set.contains(&state.buffer))
        {
            return Err(ReplayError::Descriptor(
                "prepared recurrent cursor projection target frontier mismatch".into(),
            ));
        }
        let mut target_to_source = Vec::with_capacity(target_schema.len());
        for target_state in &target_schema {
            let source_ordinal = source_schema
                .binary_search_by_key(&target_state.buffer, |state| state.buffer)
                .map_err(|_| {
                    ReplayError::Descriptor(
                        "prepared recurrent cursor projection source state is absent".into(),
                    )
                })?;
            let source_state = &source_schema[source_ordinal];
            if source_state.shape != target_state.shape
                || source_state.dtype != target_state.dtype
                || source_state.bytes != target_state.bytes
            {
                return Err(ReplayError::Descriptor(
                    "prepared recurrent cursor projection descriptor mismatch".into(),
                ));
            }
            target_to_source.push(source_ordinal);
        }
        Ok(Self {
            source_capture_identity,
            target_capture_identity,
            source_schema: source_schema.into_boxed_slice(),
            target_schema: target_schema.into_boxed_slice(),
            target_to_source: target_to_source.into_boxed_slice(),
        })
    }

    pub(crate) const fn target_capture_identity(&self) -> u64 {
        self.target_capture_identity
    }

    #[cfg(test)]
    pub(crate) fn source_ordinals(&self) -> &[usize] {
        &self.target_to_source
    }

    pub(crate) fn project(
        &self,
        source: &MixedReplayCursor,
    ) -> Result<ProjectedRecurrentCursor, RecurrentCursorProjectionError> {
        if source.frontier.len() != self.source_schema.len() {
            return Err(RecurrentCursorProjectionError::IncompleteSourceFrontier);
        }
        if source.capture_identity != self.source_capture_identity {
            return Err(RecurrentCursorProjectionError::InvalidSource(
                ReplayError::Descriptor(
                    "prepared recurrent cursor projection source frontier mismatch".into(),
                ),
            ));
        }
        for (actual, expected) in source.frontier.iter().zip(self.source_schema.iter()) {
            if actual.buffer != expected.buffer
                || actual.version < expected.version
                || actual.shape != expected.shape
                || actual.dtype != expected.dtype
                || actual.bytes != expected.bytes
            {
                return Err(RecurrentCursorProjectionError::InvalidSource(
                    ReplayError::Descriptor(
                        "prepared recurrent cursor projection source descriptor mismatch".into(),
                    ),
                ));
            }
        }

        let mut frontier = Vec::with_capacity(self.target_schema.len());
        let mut source_updates = Vec::with_capacity(self.target_schema.len());
        for (target_state, source_ordinal) in
            self.target_schema.iter().zip(self.target_to_source.iter())
        {
            let projected = source.frontier[*source_ordinal].clone();
            debug_assert_eq!(projected.buffer, target_state.buffer);
            let next_version = projected
                .version
                .checked_add(1)
                .ok_or(RecurrentCursorProjectionError::VersionOverflow)?;
            source_updates.push((*source_ordinal, next_version));
            frontier.push(projected);
        }
        Ok(ProjectedRecurrentCursor {
            cursor: MixedReplayCursor {
                capture_identity: self.target_capture_identity,
                frontier,
            },
            source_updates: source_updates.into_boxed_slice(),
        })
    }
}

impl ProjectedRecurrentCursor {
    pub(crate) fn cursor(&self) -> &MixedReplayCursor {
        &self.cursor
    }

    pub(crate) fn cursor_mut(&mut self) -> &mut MixedReplayCursor {
        &mut self.cursor
    }

    /// Publishes only prevalidated version words after the target runtime bank
    /// transaction has committed. This is deliberately infallible.
    pub(crate) fn publish(self, source: &mut MixedReplayCursor) {
        for ((source_ordinal, next_version), target) in self
            .source_updates
            .iter()
            .copied()
            .zip(&self.cursor.frontier)
        {
            debug_assert_eq!(target.version, next_version);
            source.frontier[source_ordinal].version = next_version;
        }
    }
}

struct StagedMixedReplay {
    entry: crate::EffectBatchEntry,
    outputs: Vec<crate::TensorData>,
}

struct RecurrentReplayRequest<'a> {
    native: Option<NativeReplayContext<'a>>,
    selected_requested: Option<&'a [u64]>,
    injected_failure: Option<u64>,
}

/// Execution context for a strict-native replay of a captured pure prefix.
/// The prepared program owns kernels, immutable schema, and invalidatable
/// scratch; authoritative tensor bytes remain caller/runtime-owned.
pub(crate) struct NativeReplayContext<'a> {
    executor: &'a super::captured_replay::CapturedReplayExecutor,
    prepared: &'a mut PreparedRecurrentNativeReplay,
}

impl<'a> NativeReplayContext<'a> {
    pub(crate) const fn new(
        executor: &'a super::captured_replay::CapturedReplayExecutor,
        prepared: &'a mut PreparedRecurrentNativeReplay,
    ) -> Self {
        Self { executor, prepared }
    }

    pub(crate) fn replay_recurrent_checked<F>(
        self,
        runtime: &mut crate::EffectRuntime,
        cursor: &mut MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        injected_failure: Option<u64>,
        validate_transition: F,
    ) -> Result<NativeMixedReplayResult, ReplayError>
    where
        F: FnOnce(&[crate::TensorData], &[&crate::TensorData]) -> Result<(), String>,
    {
        self.replay_recurrent_selected_checked(
            runtime,
            cursor,
            provided,
            None,
            injected_failure,
            validate_transition,
        )
    }

    pub(crate) fn replay_recurrent_selected_checked<F>(
        self,
        runtime: &mut crate::EffectRuntime,
        cursor: &mut MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        selected_requested: Option<&[u64]>,
        injected_failure: Option<u64>,
        validate_transition: F,
    ) -> Result<NativeMixedReplayResult, ReplayError>
    where
        F: FnOnce(&[crate::TensorData], &[&crate::TensorData]) -> Result<(), String>,
    {
        let Self { executor, prepared } = self;
        prepared.validate_cursor(cursor)?;
        let current = &cursor.frontier;
        let next = current
            .iter()
            .cloned()
            .map(|mut state| {
                state.version = state
                    .version
                    .checked_add(1)
                    .ok_or_else(|| ReplayError::Corrupt("recurrent version overflow".into()))?;
                Ok(state)
            })
            .collect::<Result<Vec<_>, ReplayError>>()?;
        let PreparedRecurrentNativeReplay {
            trace,
            plan,
            pure,
            requested,
            banks: bank_layout,
        } = prepared;
        let all_requested = requested.as_slice();
        let requested = selected_requested.unwrap_or(all_requested);
        validate_requested_selection(all_requested, requested)?;
        bank_layout.validate_external_inputs(provided)?;
        let public = requested.iter().copied().collect::<BTreeSet<_>>();
        let staged = runtime.transact_recurrent_native_full_frontier(
            current,
            &next,
            &bank_layout.modes,
            |banks| {
                let (mut values, traffic, executor_wall_time) = {
                    if banks.len() != bank_layout.banks.len() {
                        return Err(ReplayError::Corrupt(
                            "prepared recurrent bank cardinality mismatch".into(),
                        ));
                    }
                    let bank_count = banks.len();
                    let mut active = Vec::with_capacity(bank_count);
                    let mut inactive = Vec::with_capacity(bank_count);
                    for (ordinal, (bank, binding)) in
                        banks.iter_mut().zip(bank_layout.banks.iter()).enumerate()
                    {
                        let mode = bank_layout.validate_bank(ordinal, bank, binding)?;
                        if matches!(mode, crate::effects::runtime::RecurrentBankMode::Retain) {
                            active.push(bank.successor());
                            inactive.push(None);
                        } else {
                            let (current, successor) = bank.tensors();
                            active.push(current);
                            inactive.push(Some(successor));
                        }
                    }
                    #[cfg(test)]
                    INDEXED_RECURRENT_BANK_BINDINGS.with(|count| {
                        count.set(count.get().saturating_add(bank_count));
                    });
                    let mut borrowed = plan.new_bindings();
                    let executor_started = Instant::now();
                    let executed = executor.execute_sealed_planned_native_items_resolved(
                        pure,
                        plan,
                        &mut borrowed,
                        Some(&public),
                        |workspace, borrowed| {
                            for (ordinal, (binding, mode)) in bank_layout
                                .banks
                                .iter()
                                .zip(bank_layout.modes.iter())
                                .enumerate()
                            {
                                if matches!(
                                    mode,
                                    crate::effects::runtime::RecurrentBankMode::Retain
                                ) {
                                    continue;
                                }
                                let successor = inactive[ordinal].take().ok_or_else(|| {
                                    ReplayError::Missing(format!(
                                        "recurrent successor state {}",
                                        binding.initial.buffer
                                    ))
                                })?;
                                workspace.borrow_recurrent_output(
                                    binding.replacement.producer,
                                    successor,
                                    borrowed,
                                )?;
                            }
                            if inactive.iter().any(Option::is_some) {
                                return Err(ReplayError::Corrupt(
                                    "recurrent successor binding set mismatch".into(),
                                ));
                            }
                            Ok(())
                        },
                        |input_ordinal, input, _workspace| {
                            let binding =
                                bank_layout.inputs.get(input_ordinal).ok_or_else(|| {
                                    ReplayError::Corrupt(
                                        "prepared recurrent input ordinal is absent".into(),
                                    )
                                })?;
                            let value = match binding.source {
                                PreparedRecurrentInputSource::State { ordinal } => {
                                    active.get(ordinal).copied().ok_or_else(|| {
                                        ReplayError::Missing(format!(
                                            "recurrent input state ordinal {ordinal}"
                                        ))
                                    })?
                                }
                                PreparedRecurrentInputSource::External => provided
                                    .get(&input.name)
                                    .ok_or_else(|| ReplayError::Missing(input.name.clone()))?,
                            };
                            match binding.source {
                                PreparedRecurrentInputSource::State { .. } => Ok(
                                    super::native_replay_workspace::ResolvedNativeReplayInput::Recurrent(
                                        value,
                                    ),
                                ),
                                PreparedRecurrentInputSource::External => Ok(
                                    super::native_replay_workspace::ResolvedNativeReplayInput::External(
                                        value,
                                    ),
                                ),
                            }
                        },
                    );
                    let executor_wall_time = executor_started.elapsed();
                    let (values, traffic) = executed?;
                    (values, traffic, executor_wall_time)
                };
                let outputs = requested
                    .iter()
                    .map(|id| {
                        if bank_layout.successor_outputs.contains(id) {
                            // A public state successor was materialized as an
                            // independent snapshot; preserve it while the inactive
                            // recurrent bank becomes authoritative.
                            values.tensor(*id, "requested mixed output").cloned()
                        } else {
                            values.take_tensor(*id, "requested mixed output")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let successor_values = banks
                    .iter()
                    .zip(bank_layout.banks.iter())
                    .enumerate()
                    .map(|(ordinal, (bank, binding))| {
                        bank_layout.validate_bank(ordinal, bank, binding)?;
                        Ok(bank.successor())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                validate_transition(&outputs, &successor_values).map_err(ReplayError::Execute)?;
                if let Some(step) = injected_failure
                    && bank_layout.replacement_steps.contains(&step)
                {
                    return Err(ReplayError::Execute(format!(
                        "recurrent commit: {:?}",
                        crate::RuntimeError::InjectedFailure(step)
                    )));
                }
                let mut traffic = traffic;
                if bank_layout.retained_count != 0 {
                    traffic.retained_recurrent_state_count = bank_layout.retained_count;
                    traffic.retained_recurrent_state_bytes = bank_layout.retained_bytes;
                    traffic.replaced_recurrent_state_count = bank_layout.replaced_count;
                    traffic.replaced_recurrent_state_bytes =
                        traffic.borrowed_recurrent_output_bytes;
                }
                Ok((outputs, traffic, executor_wall_time))
            },
        );
        let (outputs, traffic, executor_wall_time) = match staged {
            Ok(staged) => staged,
            Err(crate::effects::runtime::RecurrentTransactionError::Stage(error)) => {
                return Err(error);
            }
            Err(crate::effects::runtime::RecurrentTransactionError::Runtime(error)) => {
                return Err(ReplayError::Execute(format!(
                    "recurrent transaction: {error:?}"
                )));
            }
            Err(crate::effects::runtime::RecurrentTransactionError::Contract(reason)) => {
                return Err(ReplayError::Corrupt(reason.into()));
            }
        };
        cursor.frontier = next.clone();
        Ok(NativeMixedReplayResult {
            replay: MixedReplayResult {
                outputs,
                committed: next,
                native_trace: Some(trace.replay.clone()),
            },
            traffic,
            executor_wall_time,
        })
    }
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

/// Preparation-only evidence for one strict-native mixed pure prefix.
/// No persistent state is executed or mutated while producing this value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeMixedPreparationTrace {
    pub(crate) replay: NativeMixedReplayTrace,
    pub(crate) item_count: usize,
    pub(crate) cache_hit_count: usize,
    pub(crate) cache_miss_count: usize,
    pub(crate) module: crate::backend::NativeScheduleModulePreparation,
    pub(crate) dispatch_segmentation: super::native_replay_workspace::NativeDispatchSegmentation,
}

/// Reusable strict-native ownership for one recurrent mixed capture. It keeps
/// prepared kernels, authenticated immutable schema, capture constants, and
/// zero-initialized scratch, never witness bindings, runtime leases, or cursor
/// state from preparation.
pub(crate) struct PreparedRecurrentNativeReplay {
    trace: NativeMixedPreparationTrace,
    plan: super::captured_replay::SealedPlannedNativeItems,
    pure: CapturedSchedule,
    requested: Vec<u64>,
    banks: PreparedRecurrentBankLayout,
}

pub(crate) struct RecurrentNativePreparation {
    pure: CapturedSchedule,
    inputs: Option<BTreeMap<String, crate::TensorData>>,
    requested: Vec<u64>,
    replacements: PreparedRecurrentReplacementPlan,
    replay: NativeMixedReplayTrace,
    retained_recurrent_states: Vec<super::captured_replay::NativeRecurrentStateRetention>,
}

impl RecurrentNativePreparation {
    pub(crate) fn pure(&self) -> &CapturedSchedule {
        &self.pure
    }

    pub(crate) fn pure_and_inputs(
        &self,
    ) -> (&CapturedSchedule, &BTreeMap<String, crate::TensorData>) {
        (
            &self.pure,
            self.inputs
                .as_ref()
                .expect("native recurrent input witnesses were already released"),
        )
    }

    pub(crate) fn release_input_witnesses(&mut self) {
        self.inputs = None;
    }

    pub(crate) fn retained_recurrent_states(
        &self,
    ) -> &[super::captured_replay::NativeRecurrentStateRetention] {
        &self.retained_recurrent_states
    }

    pub(crate) fn finish(
        self,
        plan: super::captured_replay::PlannedNativeItems,
    ) -> Result<PreparedRecurrentNativeReplay, ReplayError> {
        let trace = NativeMixedPreparationTrace {
            replay: self.replay,
            item_count: plan.item_count(),
            cache_hit_count: plan.cache_hit_count(),
            cache_miss_count: plan.cache_miss_count(),
            module: plan.module_preparation(),
            dispatch_segmentation: plan.dispatch_segmentation(),
        };
        PreparedRecurrentNativeReplay::new(
            trace,
            plan,
            self.pure,
            self.requested,
            self.replacements,
        )
    }
}

#[derive(Clone, Debug)]
struct PreparedRecurrentReplacement {
    step: u64,
    producer: u64,
    buffer: u64,
}

#[derive(Clone, Debug)]
struct PreparedRecurrentReplacementPlan {
    external_inputs: BTreeSet<String>,
    state_inputs: BTreeMap<String, u64>,
    initial_frontier: Vec<BufferState>,
    replacements: Vec<PreparedRecurrentReplacement>,
}

#[derive(Clone, Copy, Debug)]
enum PreparedRecurrentInputSource {
    External,
    State { ordinal: usize },
}

#[derive(Clone, Debug)]
struct PreparedRecurrentInputBinding {
    source: PreparedRecurrentInputSource,
}

#[derive(Clone, Debug)]
struct PreparedRecurrentBankBinding {
    initial: BufferState,
    replacement: PreparedRecurrentReplacement,
}

/// Immutable projection from canonical frontier ordinals to replay bindings.
/// It owns descriptors and indices only; leases, generations, tensors, and
/// pointers remain call-scoped and are revalidated by `EffectRuntime`.
#[derive(Clone, Debug)]
struct PreparedRecurrentBankLayout {
    inputs: Box<[PreparedRecurrentInputBinding]>,
    banks: Box<[PreparedRecurrentBankBinding]>,
    modes: Box<[crate::effects::runtime::RecurrentBankMode]>,
    retained_count: u64,
    retained_bytes: u64,
    replaced_count: u64,
    successor_outputs: BTreeSet<u64>,
    replacement_steps: BTreeSet<u64>,
    external_inputs: BTreeSet<String>,
}

#[cfg(test)]
pub(crate) struct RecurrentBankLayoutEvidence {
    pub(crate) buffers: Vec<u64>,
    pub(crate) input_ordinals: Vec<usize>,
    pub(crate) retained: Vec<bool>,
    pub(crate) retained_count: u64,
    pub(crate) retained_bytes: u64,
}

impl PreparedRecurrentNativeReplay {
    fn new(
        trace: NativeMixedPreparationTrace,
        plan: super::captured_replay::PlannedNativeItems,
        pure: CapturedSchedule,
        requested: Vec<u64>,
        replacements: PreparedRecurrentReplacementPlan,
    ) -> Result<Self, ReplayError> {
        replacements.authenticate_recurrent_store_groups(plan.recurrent_store_groups())?;
        replacements
            .authenticate_retained_recurrent_states(&pure, plan.retained_recurrent_states())?;
        let banks = PreparedRecurrentBankLayout::new(
            &pure,
            replacements,
            plan.retained_recurrent_states(),
        )?;
        let plan = plan.seal(&pure)?;
        if trace.item_count != plan.item_count()
            || trace.cache_hit_count != plan.cache_hit_count()
            || trace.cache_miss_count != plan.cache_miss_count()
            || trace.module != plan.module_preparation()
            || trace.replay.vectorized != plan.vectorized()
            || trace.replay.pure_item_cache_keys.as_slice() != plan.schedule_cache_keys()
            || requested.iter().any(|id| !pure.requested.contains(id))
        {
            return Err(ReplayError::Corrupt(
                "prepared recurrent native plan inventory mismatch".into(),
            ));
        }
        Ok(Self {
            trace,
            plan,
            pure,
            requested,
            banks,
        })
    }

    pub(crate) fn preparation_trace(&self) -> &NativeMixedPreparationTrace {
        &self.trace
    }

    #[cfg(test)]
    pub(crate) fn workspace_stats(
        &self,
    ) -> super::native_replay_workspace::NativeReplayWorkspaceStats {
        self.plan.workspace_stats()
    }

    #[cfg(test)]
    pub(crate) fn zero_domain_item_count(&self) -> usize {
        self.plan.zero_domain_item_count()
    }

    #[cfg(test)]
    pub(crate) fn last_executed_native_item_count(&self) -> usize {
        self.plan.last_executed_native_item_count()
    }

    #[cfg(test)]
    pub(crate) fn retained_recurrent_state_count(&self) -> usize {
        usize::try_from(self.banks.retained_count)
            .expect("prepared retained-state count was checked from usize")
    }

    #[cfg(test)]
    pub(crate) fn retained_recurrent_state_buffers(&self) -> BTreeSet<u64> {
        self.banks
            .banks
            .iter()
            .zip(self.banks.modes.iter())
            .filter_map(|(binding, mode)| {
                matches!(mode, crate::effects::runtime::RecurrentBankMode::Retain)
                    .then_some(binding.initial.buffer)
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn recurrent_bank_layout_evidence(&self) -> RecurrentBankLayoutEvidence {
        let buffers = self
            .banks
            .banks
            .iter()
            .map(|binding| binding.initial.buffer)
            .collect();
        let input_ordinals = self
            .banks
            .inputs
            .iter()
            .filter_map(|binding| match binding.source {
                PreparedRecurrentInputSource::External => None,
                PreparedRecurrentInputSource::State { ordinal } => Some(ordinal),
            })
            .collect();
        let retained = self
            .banks
            .modes
            .iter()
            .map(|mode| matches!(mode, crate::effects::runtime::RecurrentBankMode::Retain))
            .collect();
        RecurrentBankLayoutEvidence {
            buffers,
            input_ordinals,
            retained,
            retained_count: self.banks.retained_count,
            retained_bytes: self.banks.retained_bytes,
        }
    }

    #[cfg(test)]
    pub(crate) fn last_module_dispatch_counts(&self) -> (usize, usize) {
        self.plan.last_module_dispatch_counts()
    }

    #[cfg(test)]
    pub(crate) fn inject_dispatch_failure(&mut self, index: usize) {
        self.plan.inject_dispatch_failure(index);
    }

    #[cfg(test)]
    pub(crate) fn recurrent_store_group_indices(&self) -> Vec<Vec<usize>> {
        self.plan.recurrent_store_group_indices()
    }

    #[cfg(test)]
    pub(crate) fn recurrent_store_group_admission_diagnostics(&self) -> Vec<String> {
        self.plan
            .recurrent_store_group_admissions()
            .iter()
            .enumerate()
            .map(|(manifest, diagnostic)| format!("manifest {manifest}: {}", diagnostic.describe()))
            .collect()
    }

    fn validate_cursor(&self, cursor: &MixedReplayCursor) -> Result<(), ReplayError> {
        if cursor.capture_identity != self.trace.replay.artifact_identity {
            return Err(ReplayError::Descriptor(
                "recurrent cursor belongs to a different mixed capture".into(),
            ));
        }
        if cursor.frontier.len() != self.banks.banks.len() {
            return Err(ReplayError::Descriptor(
                "recurrent cursor state frontier is incomplete".into(),
            ));
        }
        for (actual, initial) in cursor.frontier.iter().zip(self.banks.banks.iter()) {
            let initial = &initial.initial;
            if actual.buffer != initial.buffer
                || actual.version < initial.version
                || actual.shape != initial.shape
                || actual.dtype != initial.dtype
                || actual.bytes != initial.bytes
            {
                return Err(ReplayError::Descriptor(
                    "recurrent cursor state descriptor mismatch".into(),
                ));
            }
        }
        Ok(())
    }

    /// The generic detached staging path still accepts a capture separately,
    /// so it retains the historical full artifact check. Compiled recurrent
    /// replay instead enters through the sealed context above.
    fn validate_capture(
        &self,
        capture: &CapturedMixedSchedule,
    ) -> Result<NativeMixedReplayTrace, ReplayError> {
        let replay = capture.native_replay_trace(self.plan.vectorized())?;
        if replay != self.trace.replay {
            return Err(ReplayError::Corrupt(
                "prepared recurrent native capture identity mismatch".into(),
            ));
        }
        Ok(replay)
    }

    #[cfg(test)]
    pub(crate) fn structure_validation_count(&self) -> usize {
        self.plan.structure_validation_count()
    }
}

impl PreparedRecurrentBankLayout {
    fn validate_bank(
        &self,
        ordinal: usize,
        bank: &crate::host_buffer::HostBufferBank<'_>,
        binding: &PreparedRecurrentBankBinding,
    ) -> Result<crate::effects::runtime::RecurrentBankMode, ReplayError> {
        let mode =
            self.modes.get(ordinal).copied().ok_or_else(|| {
                ReplayError::Corrupt("prepared recurrent bank mode is absent".into())
            })?;
        if bank.ordinal() != ordinal
            || bank.buffer_id() != binding.initial.buffer
            || bank.is_retained()
                != matches!(mode, crate::effects::runtime::RecurrentBankMode::Retain)
        {
            return Err(ReplayError::Corrupt(
                "prepared recurrent bank order mismatch".into(),
            ));
        }
        Ok(mode)
    }

    fn new(
        pure: &CapturedSchedule,
        replacements: PreparedRecurrentReplacementPlan,
        retained: &[super::captured_replay::NativeRecurrentStateRetention],
    ) -> Result<Self, ReplayError> {
        let PreparedRecurrentReplacementPlan {
            external_inputs,
            state_inputs,
            initial_frontier,
            replacements,
        } = replacements;
        if initial_frontier.len() != replacements.len() {
            return Err(ReplayError::Corrupt(
                "prepared recurrent bank cardinality mismatch".into(),
            ));
        }
        let retained = retained
            .iter()
            .map(|state| state.state_buffer)
            .collect::<BTreeSet<_>>();
        let retained_count = u64::try_from(retained.len()).map_err(|_| {
            ReplayError::Corrupt("retained recurrent state count exceeds u64".into())
        })?;
        let replaced_count = u64::try_from(
            initial_frontier
                .len()
                .checked_sub(retained.len())
                .ok_or_else(|| {
                    ReplayError::Corrupt("retained recurrent state count exceeds frontier".into())
                })?,
        )
        .map_err(|_| ReplayError::Corrupt("replaced recurrent state count exceeds u64".into()))?;
        let mut retained_bytes = 0u64;
        let mut modes = Vec::with_capacity(initial_frontier.len());
        let mut banks = Vec::with_capacity(initial_frontier.len());
        let mut successor_outputs = BTreeSet::new();
        let mut replacement_steps = BTreeSet::new();
        let mut previous_buffer = None;
        for (initial, replacement) in initial_frontier.into_iter().zip(replacements) {
            if initial.buffer != replacement.buffer
                || previous_buffer.is_some_and(|buffer| buffer >= initial.buffer)
                || !successor_outputs.insert(replacement.producer)
                || !replacement_steps.insert(replacement.step)
            {
                return Err(ReplayError::Corrupt(
                    "prepared recurrent bank binding mismatch".into(),
                ));
            }
            previous_buffer = Some(initial.buffer);
            let is_retained = retained.contains(&initial.buffer);
            if is_retained {
                retained_bytes = retained_bytes
                    .checked_add(u64::try_from(initial.bytes).map_err(|_| {
                        ReplayError::Corrupt("retained recurrent state bytes exceed u64".into())
                    })?)
                    .ok_or_else(|| {
                        ReplayError::Corrupt("retained recurrent state bytes overflow".into())
                    })?;
            }
            modes.push(if is_retained {
                crate::effects::runtime::RecurrentBankMode::Retain
            } else {
                crate::effects::runtime::RecurrentBankMode::Replace
            });
            banks.push(PreparedRecurrentBankBinding {
                initial,
                replacement,
            });
        }
        if retained.iter().any(|buffer| {
            banks
                .binary_search_by_key(buffer, |binding| binding.initial.buffer)
                .is_err()
        }) {
            return Err(ReplayError::Corrupt(
                "retained recurrent state is absent from the frontier".into(),
            ));
        }
        let mut inputs = Vec::with_capacity(pure.inputs.len());
        for input in &pure.inputs {
            let source = match state_inputs.get(&input.name) {
                Some(buffer) => {
                    let ordinal = banks
                        .binary_search_by_key(buffer, |binding| binding.initial.buffer)
                        .map_err(|_| {
                            ReplayError::Corrupt(
                                "prepared recurrent input is absent from the frontier".into(),
                            )
                        })?;
                    PreparedRecurrentInputSource::State { ordinal }
                }
                None if external_inputs.contains(&input.name) => {
                    PreparedRecurrentInputSource::External
                }
                None => {
                    return Err(ReplayError::Corrupt(
                        "prepared recurrent input binding is absent".into(),
                    ));
                }
            };
            inputs.push(PreparedRecurrentInputBinding { source });
        }
        #[cfg(test)]
        record_prepared_replay_validation(|counts| {
            counts.recurrent_bank_layouts = counts.recurrent_bank_layouts.saturating_add(1);
        });
        Ok(Self {
            inputs: inputs.into_boxed_slice(),
            banks: banks.into_boxed_slice(),
            modes: modes.into_boxed_slice(),
            retained_count,
            retained_bytes,
            replaced_count,
            successor_outputs,
            replacement_steps,
            external_inputs,
        })
    }

    fn validate_external_inputs(
        &self,
        provided: &BTreeMap<String, crate::TensorData>,
    ) -> Result<(), ReplayError> {
        if let Some(name) = provided
            .keys()
            .find(|name| !self.external_inputs.contains(*name))
        {
            return Err(ReplayError::Extra(name.clone()));
        }
        if let Some(name) = self
            .external_inputs
            .iter()
            .find(|name| !provided.contains_key(*name))
        {
            return Err(ReplayError::Missing(name.clone()));
        }
        Ok(())
    }
}

impl PreparedRecurrentReplacementPlan {
    fn authenticate_retained_recurrent_states(
        &self,
        pure: &CapturedSchedule,
        retained: &[super::captured_replay::NativeRecurrentStateRetention],
    ) -> Result<(), ReplayError> {
        let replacements = self
            .replacements
            .iter()
            .map(|replacement| ((replacement.producer, replacement.buffer), replacement))
            .collect::<BTreeMap<_, _>>();
        let mut buffers = BTreeSet::new();
        let state_inputs = pure
            .inputs
            .iter()
            .filter_map(|input| {
                self.state_inputs
                    .get(&input.name)
                    .map(|buffer| (input.desc.id, *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        for state in retained {
            if !buffers.insert(state.state_buffer)
                || !replacements.contains_key(&(state.output, state.state_buffer))
                || state_inputs.get(&state.input) != Some(&state.state_buffer)
            {
                return Err(ReplayError::Corrupt(
                    "prepared retained recurrent replacement mismatch".into(),
                ));
            }
        }
        Ok(())
    }

    fn exact_dense_passthroughs(
        &self,
        pure: &CapturedSchedule,
        requested: &[u64],
    ) -> Result<Vec<super::captured_replay::NativeRecurrentStateRetention>, ReplayError> {
        let state_inputs = pure
            .inputs
            .iter()
            .filter_map(|input| {
                self.state_inputs
                    .get(&input.name)
                    .map(|buffer| (input.desc.id, *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        let mut retained: Vec<super::captured_replay::NativeRecurrentStateRetention> = Vec::new();
        for replacement in &self.replacements {
            if requested.contains(&replacement.producer) {
                continue;
            }
            let Some((logical_index, item)) = pure
                .items
                .iter()
                .enumerate()
                .find(|(_, item)| item.primary_output().id == replacement.producer)
            else {
                continue;
            };
            let Some((input, output)) = crate::backend::canonical_dense_copy(item) else {
                continue;
            };
            if state_inputs.get(&input) != Some(&replacement.buffer)
                || item
                    .consumers
                    .iter()
                    .any(|consumer| pure.items.iter().any(|candidate| candidate.id == *consumer))
                || retained
                    .iter()
                    .any(|state| state.input == input || state.output == output)
            {
                continue;
            }
            retained.push(super::captured_replay::NativeRecurrentStateRetention {
                logical_index,
                input,
                output,
                state_buffer: replacement.buffer,
            });
        }
        Ok(retained)
    }

    fn authenticate_recurrent_store_groups(
        &self,
        groups: &[super::captured_replay::RecurrentStoreGroupManifest],
    ) -> Result<(), ReplayError> {
        let replacements = self
            .replacements
            .iter()
            .map(|replacement| (replacement.producer, replacement.buffer))
            .collect::<BTreeMap<_, _>>();
        for member in groups.iter().flat_map(|group| &group.members) {
            if replacements.get(&member.output) != Some(&member.state_buffer) {
                return Err(ReplayError::Corrupt(
                    "prepared recurrent store-group replacement mismatch".into(),
                ));
            }
        }
        Ok(())
    }

    fn from_capture(capture: &CapturedMixedSchedule) -> Result<Self, ReplayError> {
        // Prepared replacement metadata authenticates the capture's immutable
        // version-zero descriptor floor, never the cursor used at preparation.
        // The same sealed replay can therefore accept any matching resumed
        // frontier whose versions remain at or above this canonical floor.
        let frontier = recurrent_initial_frontier(capture)?;
        let frontier_by_buffer = frontier
            .iter()
            .map(|state| (state.buffer, state))
            .collect::<BTreeMap<_, _>>();
        let mut state_inputs = BTreeMap::new();
        for binding in &capture.state_bindings {
            if binding.view.is_some() {
                return Err(ReplayError::Unsupported(
                    "prepared recurrent native replacement does not support state input views"
                        .into(),
                ));
            }
            let input = capture
                .schedule
                .inputs
                .iter()
                .find(|input| input.node == binding.input_node)
                .ok_or_else(|| ReplayError::Corrupt("state input ABI is absent".into()))?;
            if !frontier_by_buffer.contains_key(&binding.state.buffer) {
                return Err(ReplayError::Corrupt(
                    "state input is absent from recurrent frontier".into(),
                ));
            }
            match state_inputs.insert(input.name.clone(), binding.state.buffer) {
                Some(buffer) if buffer != binding.state.buffer => {
                    return Err(ReplayError::Corrupt(
                        "persistent state input has conflicting bindings".into(),
                    ));
                }
                _ => {}
            }
        }
        let external_inputs = capture
            .schedule
            .inputs
            .iter()
            .filter(|input| !state_inputs.contains_key(&input.name))
            .map(|input| input.name.clone())
            .collect::<BTreeSet<_>>();

        let schedule = Schedule {
            items: capture.schedule.items.clone(),
            requested_materializations: requested_materializations(&capture.schedule),
            requested_passthroughs: capture.schedule.requested_passthroughs.clone(),
            value_bindings: capture.value_bindings.clone(),
            state_bindings: capture.state_bindings.clone(),
        };
        let plan = effect_plan(&schedule)?;
        let mut producers = BTreeMap::new();
        for binding in &capture.value_bindings {
            let payload = effect_payload(&capture.schedule.items[binding.effect_item as usize])?;
            if producers
                .insert(payload.step, binding.producer_output.id)
                .is_some()
            {
                return Err(ReplayError::Corrupt(
                    "recurrent effect has multiple pure successors".into(),
                ));
            }
        }
        let mut replacements = Vec::with_capacity(plan.steps.len());
        let mut replaced = BTreeSet::new();
        for step in &plan.steps {
            let producer = producers.get(&step.id).copied().ok_or_else(|| {
                ReplayError::Unsupported(
                    "prepared recurrent native effect is not pure-sourced".into(),
                )
            })?;
            let initial = frontier_by_buffer.get(&step.write.buffer).ok_or_else(|| {
                ReplayError::Unsupported(
                    "prepared recurrent native effect targets no frontier state".into(),
                )
            })?;
            let Some(read) = step.reads.first() else {
                return Err(ReplayError::Corrupt(
                    "recurrent effect has no predecessor state".into(),
                ));
            };
            if step.target_view.is_some()
                || step.index_plan.is_some()
                || read != *initial
                || step.write.buffer != initial.buffer
                || initial.version.checked_add(1) != Some(step.write.version)
                || step.write.shape != initial.shape
                || step.write.dtype != initial.dtype
                || step.write.bytes != initial.bytes
                || !replaced.insert(step.write.buffer)
            {
                return Err(ReplayError::Unsupported(
                    "prepared recurrent native effect is not one full frontier replacement".into(),
                ));
            }
            replacements.push(PreparedRecurrentReplacement {
                step: step.id,
                producer,
                buffer: step.write.buffer,
            });
        }
        replacements.sort_by_key(|replacement| replacement.buffer);
        if replacements.len() != frontier.len()
            || replacements
                .iter()
                .map(|replacement| replacement.buffer)
                .ne(frontier.iter().map(|state| state.buffer))
        {
            return Err(ReplayError::Unsupported(
                "prepared recurrent native replacement frontier is incomplete".into(),
            ));
        }
        Ok(Self {
            external_inputs,
            state_inputs,
            initial_frontier: frontier,
            replacements,
        })
    }
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
        let inputs = bind_persistent_inputs(
            &capture.schedule.inputs,
            &capture.state_bindings,
            provided,
            |binding| {
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
                Ok(value)
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )?;
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
        for requested in &self.capture.schedule.requested {
            if !pure.requested.contains(requested) {
                pure.requested.push(*requested);
            }
        }
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

    pub(crate) fn item_count(&self) -> usize {
        self.plan.item_count()
    }

    pub(crate) fn cache_hit_count(&self) -> usize {
        self.plan.cache_hit_count()
    }

    pub(crate) fn cache_miss_count(&self) -> usize {
        self.plan.cache_miss_count()
    }

    pub(crate) fn execute(
        &mut self,
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
        executor.execute_planned_native_items(&pure, &self.bound.inputs, &mut self.plan)
    }

    pub(crate) fn execute_stage(
        mut self,
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

    /// Creates the canonical version-zero cursor for repeated interpreter
    /// replay. Only persistent states actually required by this capture enter
    /// the frontier; pure value bindings never manufacture runtime buffers.
    pub fn initial_recurrent_cursor(&self) -> Result<MixedReplayCursor, ReplayError> {
        MixedReplayCursor::new(self, recurrent_initial_frontier(self)?)
    }

    /// Replays one recurrent interpreter step against `cursor`'s exact logical
    /// state frontier. Pure outputs and all effect candidates are detached
    /// before the one [`crate::EffectRuntime`] batch commit. The cursor advances
    /// only after that commit succeeds, so every validation, execution, or
    /// injected failure leaves both runtime state and cursor unchanged.
    pub fn replay_recurrent(
        &self,
        runtime: &mut crate::EffectRuntime,
        cursor: &mut MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<MixedReplayResult, ReplayError> {
        self.replay_recurrent_checked(runtime, cursor, provided, injected_failure, |_, _| Ok(()))
    }

    /// Replays one interpreter transition while admitting its detached outputs
    /// and fully applied final successor candidates immediately before commit.
    /// Staging and commit remain bound to this exact runtime borrow, so a
    /// validator cannot move an authenticated transition between runtimes.
    pub(crate) fn replay_recurrent_checked<F>(
        &self,
        runtime: &mut crate::EffectRuntime,
        cursor: &mut MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        injected_failure: Option<u64>,
        validate_transition: F,
    ) -> Result<MixedReplayResult, ReplayError>
    where
        F: FnOnce(&[crate::TensorData], &[crate::TensorData]) -> Result<(), String>,
    {
        self.replay_recurrent_selected_checked(
            runtime,
            cursor,
            provided,
            None,
            injected_failure,
            validate_transition,
        )
    }

    pub(crate) fn replay_recurrent_selected_checked<F>(
        &self,
        runtime: &mut crate::EffectRuntime,
        cursor: &mut MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        selected_requested: Option<&[u64]>,
        injected_failure: Option<u64>,
        validate_transition: F,
    ) -> Result<MixedReplayResult, ReplayError>
    where
        F: FnOnce(&[crate::TensorData], &[crate::TensorData]) -> Result<(), String>,
    {
        self.replay_recurrent_checked_impl(
            runtime,
            cursor,
            provided,
            RecurrentReplayRequest {
                native: None,
                selected_requested,
                injected_failure,
            },
            validate_transition,
        )
    }

    /// Compiles the exact recurrent pure prefix without executing it or
    /// publishing any persistent state. The supplied values are descriptor
    /// witnesses only; native cache identity never depends on their bytes.
    #[cfg(test)]
    pub(crate) fn prepare_recurrent_native(
        &self,
        runtime: &crate::EffectRuntime,
        cursor: &MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        executor: &super::captured_replay::CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<PreparedRecurrentNativeReplay, ReplayError> {
        let preparation = self.preflight_recurrent_native(runtime, cursor, provided, vectorized)?;
        let (pure, inputs) = preparation.pure_and_inputs();
        let plan = executor.plan_native_items(pure, inputs, vectorized)?;
        preparation.finish(plan)
    }

    pub(crate) fn preflight_recurrent_native(
        &self,
        runtime: &crate::EffectRuntime,
        cursor: &MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        vectorized: bool,
    ) -> Result<RecurrentNativePreparation, ReplayError> {
        self.preflight_recurrent_native_impl(runtime, cursor, provided, vectorized, false)
    }

    pub(crate) fn preflight_recurrent_native_retaining_unchanged(
        &self,
        runtime: &crate::EffectRuntime,
        cursor: &MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        vectorized: bool,
    ) -> Result<RecurrentNativePreparation, ReplayError> {
        self.preflight_recurrent_native_impl(runtime, cursor, provided, vectorized, true)
    }

    fn preflight_recurrent_native_impl(
        &self,
        runtime: &crate::EffectRuntime,
        cursor: &MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        vectorized: bool,
        retain_unchanged: bool,
    ) -> Result<RecurrentNativePreparation, ReplayError> {
        validate(self, true)?;
        validate_recurrent_cursor(self, cursor)?;
        let starts = recurrent_rebase_starts(self, cursor)?;
        let replacements = PreparedRecurrentReplacementPlan::from_capture(self)?;
        let mut candidates = BTreeMap::new();
        for state in &cursor.frontier {
            let value = runtime
                .snapshot(state)
                .map_err(|error| ReplayError::Execute(format!("recurrent preflight: {error:?}")))?
                .tensor()
                .clone();
            candidates.insert(state.clone(), value);
        }
        let bound = BoundMixedCapture::bind(self, &candidates, starts, provided)?;
        let inputs = bound.inputs;
        let mut pure = self.schedule.clone();
        let split = pure
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| ReplayError::Unsupported("mixed capture has no effects".into()))?;
        pure.items.truncate(split);
        pure.requested = self
            .value_bindings
            .iter()
            .map(|binding| binding.producer_output.id)
            .collect();
        for requested in &self.schedule.requested {
            if !pure.requested.contains(requested) {
                pure.requested.push(*requested);
            }
        }
        pure.identity = 0;
        let retained_recurrent_states = if retain_unchanged {
            replacements.exact_dense_passthroughs(&pure, &self.schedule.requested)?
        } else {
            Vec::new()
        };
        Ok(RecurrentNativePreparation {
            pure,
            inputs: Some(inputs),
            requested: self.schedule.requested.clone(),
            replacements,
            replay: self.native_replay_trace(vectorized)?,
            retained_recurrent_states,
        })
    }

    fn replay_recurrent_checked_impl<F>(
        &self,
        runtime: &mut crate::EffectRuntime,
        cursor: &mut MixedReplayCursor,
        provided: &BTreeMap<String, crate::TensorData>,
        request: RecurrentReplayRequest<'_>,
        validate_transition: F,
    ) -> Result<MixedReplayResult, ReplayError>
    where
        F: FnOnce(&[crate::TensorData], &[crate::TensorData]) -> Result<(), String>,
    {
        validate(self, true)?;
        validate_recurrent_cursor(self, cursor)?;
        let RecurrentReplayRequest {
            native,
            selected_requested,
            injected_failure,
        } = request;
        let native_trace = native
            .as_ref()
            .map(|native| native.prepared.validate_capture(self))
            .transpose()?;

        let starts = recurrent_rebase_starts(self, cursor)?;
        let mut candidates = BTreeMap::new();
        for state in &cursor.frontier {
            let value = runtime
                .snapshot(state)
                .map_err(|error| ReplayError::Execute(format!("recurrent preflight: {error:?}")))?
                .tensor()
                .clone();
            candidates.insert(state.clone(), value);
        }

        let requested = selected_requested.unwrap_or(&self.schedule.requested);
        validate_requested_selection(&self.schedule.requested, requested)?;
        let staged = self.stage(&mut candidates, starts, provided, native, Some(requested))?;
        let batch = crate::EffectBatch::new(vec![staged.entry])
            .map_err(|error| ReplayError::Execute(format!("recurrent stage: {error:?}")))?;
        let next_frontier = recurrent_advanced_frontier(&cursor.frontier, &batch)?;
        let successors = next_frontier
            .iter()
            .map(|state| {
                candidates
                    .get(state)
                    .cloned()
                    .ok_or_else(|| ReplayError::Missing("staged recurrent successor".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_transition(&staged.outputs, &successors).map_err(ReplayError::Execute)?;
        let injected_failure =
            injected_failure.map(|step| crate::EffectBatchStep { entry: 0, step });
        let committed = runtime
            .execute_batch(&batch, injected_failure)
            .map_err(|error| ReplayError::Execute(format!("recurrent commit: {error:?}")))?;
        cursor.frontier = next_frontier;
        Ok(MixedReplayResult {
            outputs: staged.outputs,
            committed,
            native_trace,
        })
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
        Ok(self.stage(candidates, starts, provided, None, None)?.entry)
    }

    fn stage(
        &self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        provided: &BTreeMap<String, crate::TensorData>,
        native: Option<NativeReplayContext<'_>>,
        requested: Option<&[u64]>,
    ) -> Result<StagedMixedReplay, ReplayError> {
        validate(self, true)?;
        let schedule = Schedule {
            items: self.schedule.items.clone(),
            requested_materializations: requested_materializations(&self.schedule),
            requested_passthroughs: self.schedule.requested_passthroughs.clone(),
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
        if let Some(requested) = requested {
            for requested in requested {
                if !pure.requested.contains(requested) {
                    pure.requested.push(*requested);
                }
            }
        }
        pure.identity = 0;
        let inputs = bind_persistent_inputs(
            &self.schedule.inputs,
            &self.state_bindings,
            provided,
            |binding| {
                let state = rebase(&binding.state)?;
                let value = candidates
                    .get(&state)
                    .ok_or_else(|| ReplayError::Missing("batch state candidate".into()))?;
                let value = match &binding.view {
                    Some(view) => value.affine_read(view),
                    None => Ok(value.clone()),
                }
                .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
                Ok(value)
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )?;
        let values = match native {
            Some(native) => native.executor.execute_sealed_planned_native_items(
                &pure,
                &inputs,
                &mut native.prepared.plan,
            )?,
            None => super::captured_replay::replay_interpreter_items(&pure, &inputs)?,
        };
        let outputs = if let Some(requested) = requested {
            requested
                .iter()
                .map(|id| values.tensor(*id, "requested mixed output").cloned())
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let entry = self.stage_values(candidates, starts, values)?;
        Ok(StagedMixedReplay { entry, outputs })
    }

    pub(crate) fn stage_values(
        &self,
        candidates: &mut BTreeMap<BufferState, crate::TensorData>,
        starts: BTreeMap<u64, BufferState>,
        values: super::captured_replay::ReplayValues,
    ) -> Result<crate::EffectBatchEntry, ReplayError> {
        let schedule = Schedule {
            items: self.schedule.items.clone(),
            requested_materializations: requested_materializations(&self.schedule),
            requested_passthroughs: self.schedule.requested_passthroughs.clone(),
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
            requested_passthroughs: Vec::new(),
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
            requested_materializations: requested_materializations(&self.schedule),
            requested_passthroughs: self.schedule.requested_passthroughs.clone(),
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

        let inputs = bind_persistent_inputs(
            &self.schedule.inputs,
            &self.state_bindings,
            provided,
            |binding| {
                let snapshot = runtime.snapshot(&binding.state).map_err(|error| {
                    ReplayError::Execute(format!("persistent state preflight: {error:?}"))
                })?;
                let value = match &binding.view {
                    Some(view) => snapshot.tensor().affine_read(view),
                    None => Ok(snapshot.tensor().clone()),
                }
                .map_err(|error| {
                    ReplayError::Descriptor(format!("persistent affine read: {error}"))
                })?;
                Ok(value)
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )?;
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
            requested_materializations: requested_materializations(&self.schedule),
            requested_passthroughs: self.schedule.requested_passthroughs.clone(),
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
        let inputs = bind_persistent_inputs(
            &self.schedule.inputs,
            &self.state_bindings,
            provided,
            |binding| {
                let snapshot = runtime.snapshot(&binding.state).map_err(|error| {
                    ReplayError::Execute(format!("persistent state preflight: {error:?}"))
                })?;
                let value = match &binding.view {
                    Some(view) => snapshot.tensor().affine_read(view),
                    None => Ok(snapshot.tensor().clone()),
                }
                .map_err(|error| {
                    ReplayError::Descriptor(format!("persistent affine read: {error}"))
                })?;
                Ok(value)
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )?;
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

fn canonical_frontier(
    states: impl IntoIterator<Item = BufferState>,
) -> Result<Vec<BufferState>, ReplayError> {
    let mut frontier = BTreeMap::new();
    for state in states {
        crate::effects::validate_buffer_state(&state)
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let buffer = state.buffer;
        if frontier.insert(buffer, state).is_some() {
            return Err(ReplayError::Descriptor(format!(
                "duplicate recurrent cursor buffer {buffer}"
            )));
        }
    }
    Ok(frontier.into_values().collect())
}

fn recurrent_initial_frontier(
    capture: &CapturedMixedSchedule,
) -> Result<Vec<BufferState>, ReplayError> {
    #[cfg(test)]
    record_prepared_replay_validation(|counts| counts.recurrent_frontier_plans += 1);
    let schedule = Schedule {
        items: capture.schedule.items.clone(),
        requested_materializations: requested_materializations(&capture.schedule),
        requested_passthroughs: capture.schedule.requested_passthroughs.clone(),
        value_bindings: capture.value_bindings.clone(),
        state_bindings: capture.state_bindings.clone(),
    };
    let plan = effect_plan(&schedule)?;
    let pure_source_steps = capture
        .value_bindings
        .iter()
        .map(|binding| {
            effect_payload(&schedule.items[binding.effect_item as usize])
                .map(|payload| payload.step)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut required = capture
        .state_bindings
        .iter()
        .map(|binding| binding.state.buffer)
        .collect::<BTreeSet<_>>();
    for step in &plan.steps {
        required.insert(step.write.buffer);
        required.insert(step.reads[0].buffer);
        if !pure_source_steps.contains(&step.id) {
            required.insert(step.reads[1].buffer);
        }
    }

    let mut frontier = Vec::with_capacity(required.len());
    for buffer in required {
        let initial = capture
            .states
            .iter()
            .find(|state| state.buffer == buffer && state.version == 0)
            .ok_or_else(|| ReplayError::Corrupt(format!("missing initial state {buffer}")))?;
        if capture.states.iter().any(|state| {
            state.buffer == buffer
                && (state.shape != initial.shape
                    || state.dtype != initial.dtype
                    || state.bytes != initial.bytes)
        }) {
            return Err(ReplayError::Corrupt(format!(
                "recurrent state descriptor drift for buffer {buffer}"
            )));
        }
        frontier.push(initial.clone());
    }
    canonical_frontier(frontier)
}

fn validate_recurrent_cursor(
    capture: &CapturedMixedSchedule,
    cursor: &MixedReplayCursor,
) -> Result<(), ReplayError> {
    if cursor.capture_identity != identity(capture)? {
        return Err(ReplayError::Descriptor(
            "recurrent cursor belongs to a different mixed capture".into(),
        ));
    }
    let expected = recurrent_initial_frontier(capture)?;
    if cursor.frontier.len() != expected.len() {
        return Err(ReplayError::Descriptor(
            "recurrent cursor state frontier is incomplete".into(),
        ));
    }
    for (actual, initial) in cursor.frontier.iter().zip(expected) {
        if actual.buffer != initial.buffer
            || actual.version < initial.version
            || actual.shape != initial.shape
            || actual.dtype != initial.dtype
            || actual.bytes != initial.bytes
        {
            return Err(ReplayError::Descriptor(
                "recurrent cursor state descriptor mismatch".into(),
            ));
        }
    }
    Ok(())
}

fn recurrent_rebase_starts(
    capture: &CapturedMixedSchedule,
    cursor: &MixedReplayCursor,
) -> Result<BTreeMap<u64, BufferState>, ReplayError> {
    let mut starts = cursor
        .frontier
        .iter()
        .cloned()
        .map(|state| (state.buffer, state))
        .collect::<BTreeMap<_, _>>();
    let mut steps = BTreeSet::new();
    for binding in &capture.value_bindings {
        let payload = effect_payload(&capture.schedule.items[binding.effect_item as usize])?;
        if !steps.insert(payload.step) {
            return Err(ReplayError::Corrupt(format!(
                "duplicate recurrent pure source for effect step {}",
                payload.step
            )));
        }
        let source = capture
            .states
            .iter()
            .find(|state| state.buffer == payload.source.buffer && state.version == 0)
            .ok_or_else(|| {
                ReplayError::Corrupt(format!(
                    "missing recurrent pure-source descriptor {}",
                    payload.source.buffer
                ))
            })?
            .clone();
        if payload.source.version != 0
            || source.shape != payload.source.shape
            || source.dtype != payload.source.dtype
            || source.bytes != payload.source.bytes
        {
            return Err(ReplayError::Corrupt(format!(
                "recurrent pure-source descriptor mismatch for buffer {}",
                payload.source.buffer
            )));
        }
        match starts.get(&source.buffer) {
            Some(previous) if previous != &source => {
                return Err(ReplayError::Corrupt(format!(
                    "recurrent rebase descriptor conflict for buffer {}",
                    source.buffer
                )));
            }
            Some(_) => {}
            None => {
                starts.insert(source.buffer, source);
            }
        }
    }
    Ok(starts)
}

fn recurrent_advanced_frontier(
    current: &[BufferState],
    batch: &crate::EffectBatch,
) -> Result<Vec<BufferState>, ReplayError> {
    let mut next = current
        .iter()
        .cloned()
        .map(|state| (state.buffer, state))
        .collect::<BTreeMap<_, _>>();
    for rebased in batch
        .rebased_steps()
        .map_err(|error| ReplayError::Execute(format!("recurrent stage: {error:?}")))?
    {
        let initial = next
            .get(&rebased.step.write.buffer)
            .ok_or_else(|| ReplayError::Corrupt("recurrent write has no frontier state".into()))?;
        if rebased.step.write.shape != initial.shape
            || rebased.step.write.dtype != initial.dtype
            || rebased.step.write.bytes != initial.bytes
        {
            return Err(ReplayError::Descriptor(
                "recurrent write descriptor mismatch".into(),
            ));
        }
        next.insert(rebased.step.write.buffer, rebased.step.write);
    }
    canonical_frontier(next.into_values())
}

#[cfg(test)]
mod recurrent_tests {
    use super::*;
    use crate::{
        BinaryOp, CapturedReplayExecutor, DType, EffectGraph, EffectRuntime, Graph,
        ScheduleStateBinding, ScheduleValueBinding, Shape, Storage, TensorData,
        bind_schedule_states, combine_mixed_schedules, schedule, schedule_effects,
    };

    #[test]
    fn one_persistent_snapshot_serves_normal_and_transposed_consumers() {
        let input_node = NodeId::from_index(0);
        let base_desc = BufferDesc {
            id: input_node.index() as u64,
            shape: Shape::from([2, 3]),
            dtype: DType::F32,
            bytes: 24,
            alignment: 4,
            read_only: true,
            view: None,
        };
        let input = ReplayInput {
            name: "state".into(),
            node: input_node,
            desc: base_desc.clone(),
        };
        let state = BufferState {
            buffer: 17,
            version: 0,
            shape: Shape::from([2, 3]),
            dtype: DType::F32,
            bytes: 24,
        };
        let binding = |consumer_item, consumer_node, desc| ScheduleStateBinding {
            state: state.clone(),
            view: None,
            consumer_item,
            consumer_node,
            input_node,
            desc,
            abi_index: 0,
        };
        let mut transposed_desc = base_desc.clone();
        transposed_desc.view = Some(
            crate::AffineView::identity(Shape::from([2, 3]))
                .permute(&[1, 0])
                .unwrap(),
        );
        let bindings = vec![
            binding(0, NodeId::from_index(1), base_desc),
            binding(1, NodeId::from_index(2), transposed_desc),
        ];
        let value = TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let calls = std::cell::Cell::new(0);
        let inputs = bind_persistent_inputs(
            std::slice::from_ref(&input),
            &bindings,
            &BTreeMap::new(),
            |_| {
                calls.set(calls.get() + 1);
                Ok(value.clone())
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(inputs.get("state"), Some(&value));

        let mut conflicting = bindings.clone();
        conflicting[1].state.buffer = 18;
        calls.set(0);
        let error = bind_persistent_inputs(
            std::slice::from_ref(&input),
            &conflicting,
            &BTreeMap::new(),
            |_| {
                calls.set(calls.get() + 1);
                Ok(value.clone())
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::Corrupt(message)
                if message == "persistent state input has conflicting consumer bindings"
        ));
        assert_eq!(calls.get(), 1);

        let mut conflicting = bindings.clone();
        conflicting[1].view = Some(crate::AffineView::identity(Shape::from([2, 3])));
        let error = bind_persistent_inputs(
            std::slice::from_ref(&input),
            &conflicting,
            &BTreeMap::new(),
            |_| Ok(value.clone()),
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::Corrupt(message)
                if message == "persistent state input has conflicting consumer bindings"
        ));

        let mut incompatible = bindings.clone();
        incompatible[1].desc.alignment = 8;
        calls.set(0);
        let error = bind_persistent_inputs(
            std::slice::from_ref(&input),
            &incompatible,
            &BTreeMap::new(),
            |_| {
                calls.set(calls.get() + 1);
                Ok(value.clone())
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::Corrupt(message)
                if message == "persistent state consumer has incompatible base descriptor"
        ));
        assert_eq!(calls.get(), 1);

        calls.set(0);
        let error = bind_persistent_inputs(
            &[input],
            &bindings,
            &BTreeMap::from([("state".into(), value.clone())]),
            |_| {
                calls.set(calls.get() + 1);
                Ok(value.clone())
            },
            |reason| ReplayError::Corrupt(reason.into()),
            |reason| ReplayError::Descriptor(reason.into()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::Descriptor(message)
                if message == "external input shadows persistent state binding"
        ));
        assert_eq!(calls.get(), 0);
    }

    fn fixture(buffer: u64) -> (CapturedMixedSchedule, EffectRuntime) {
        let mut graph = Graph::new();
        let state = graph.input_dtype("state", [2], DType::F32);
        let delta = graph.input_dtype("delta", [2], DType::F32);
        let sum = graph.binary(BinaryOp::Add, state, delta).unwrap();
        let pure = schedule(&graph, sum).unwrap();
        let mut captured = CapturedSchedule::capture(&graph, &pure, &[sum]).unwrap();
        let state_input = pure.items[0]
            .input_bindings
            .iter()
            .find(|binding| binding.input_node == state)
            .unwrap()
            .clone();

        let mut effects = EffectGraph::default();
        let target = effects
            .insert(
                buffer,
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
        let pure = bind_schedule_states(
            pure,
            vec![ScheduleStateBinding {
                state: target.state().clone(),
                view: None,
                consumer_item: 0,
                consumer_node: sum,
                input_node: state,
                desc: state_input.desc,
                abi_index: state_input.abi_index,
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
        captured.items = mixed.items.clone();
        let captured = CapturedMixedSchedule::from_parts(
            captured,
            &mixed,
            vec![
                target.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap();
        let mut runtime = EffectRuntime::new();
        runtime
            .register(
                buffer,
                TensorData::from_storage([2], Storage::F32(vec![0.0, 0.0])).unwrap(),
            )
            .unwrap();
        (captured, runtime)
    }

    fn delta(value: f32) -> BTreeMap<String, TensorData> {
        BTreeMap::from([(
            "delta".into(),
            TensorData::from_storage([2], Storage::F32(vec![value, value])).unwrap(),
        )])
    }

    fn frontier_values(runtime: &EffectRuntime, cursor: &MixedReplayCursor) -> Vec<TensorData> {
        cursor
            .frontier()
            .iter()
            .map(|state| runtime.snapshot(state).unwrap().tensor().clone())
            .collect()
    }

    #[test]
    fn recurrent_replay_preserves_outputs_and_advances_one_logical_frontier() {
        let (capture, mut runtime) = fixture(300);
        let artifact = capture.to_bytes().unwrap();
        let mut cursor = capture.initial_recurrent_cursor().unwrap();
        let identity = cursor.capture_identity();
        assert_eq!(cursor.frontier().len(), 1);
        assert_eq!(cursor.frontier()[0].buffer, 300);
        let pure_source =
            effect_payload(&capture.schedule.items[capture.value_bindings[0].effect_item as usize])
                .unwrap()
                .source
                .buffer;
        assert!(
            !cursor
                .frontier()
                .iter()
                .any(|state| state.buffer == pure_source)
        );

        for (version, expected) in [(1, 1.0), (2, 2.0), (3, 3.0)] {
            let replay = capture
                .replay_recurrent(&mut runtime, &mut cursor, &delta(1.0), None)
                .unwrap();
            assert_eq!(cursor.capture_identity(), identity);
            assert_eq!(cursor.frontier()[0].version, version);
            assert_eq!(replay.committed, cursor.frontier());
            assert_eq!(replay.outputs.len(), 1);
            assert_eq!(
                replay.outputs[0].storage(),
                &Storage::F32(vec![expected, expected])
            );
            assert_eq!(
                frontier_values(&runtime, &cursor)[0].storage(),
                &Storage::F32(vec![expected, expected])
            );
        }
        assert_eq!(capture.to_bytes().unwrap(), artifact);
    }

    #[test]
    fn recurrent_cursor_rejects_wrong_incomplete_and_mismatched_frontiers() {
        let (capture, _) = fixture(310);
        let initial = capture.initial_recurrent_cursor().unwrap();
        let mut resumed_state = initial.frontier()[0].clone();
        resumed_state.version = 7;
        let resumed = MixedReplayCursor::resume(&capture, [resumed_state]).unwrap();
        assert_eq!(resumed.frontier()[0].version, 7);
        assert_eq!(resumed.capture_identity(), initial.capture_identity());
        assert!(MixedReplayCursor::new(&capture, resumed.frontier().to_vec()).is_err());
        assert!(MixedReplayCursor::new(&capture, Vec::new()).is_err());
        let mut wrong = initial.frontier()[0].clone();
        wrong.shape = crate::Shape::from([1]);
        wrong.bytes = DType::F32.itemsize();
        assert!(MixedReplayCursor::new(&capture, [wrong]).is_err());

        let (other, mut other_runtime) = fixture(311);
        let mut foreign = initial.clone();
        assert!(
            other
                .replay_recurrent(&mut other_runtime, &mut foreign, &delta(1.0), None)
                .is_err()
        );
        assert_eq!(foreign, initial);
        assert_eq!(
            frontier_values(&other_runtime, &other.initial_recurrent_cursor().unwrap())[0]
                .storage(),
            &Storage::F32(vec![0.0, 0.0])
        );

        let (descriptor_capture, _) = fixture(312);
        let descriptor_initial = descriptor_capture.initial_recurrent_cursor().unwrap();
        let mut descriptor_cursor = descriptor_initial.clone();
        let mut descriptor_runtime = EffectRuntime::new();
        descriptor_runtime
            .register(
                312,
                TensorData::from_storage([1], Storage::F32(vec![7.0])).unwrap(),
            )
            .unwrap();
        assert!(
            descriptor_capture
                .replay_recurrent(
                    &mut descriptor_runtime,
                    &mut descriptor_cursor,
                    &delta(1.0),
                    None,
                )
                .is_err()
        );
        assert_eq!(descriptor_cursor, descriptor_initial);
        assert_eq!(
            descriptor_runtime
                .snapshot(&BufferState {
                    buffer: 312,
                    version: 0,
                    shape: crate::Shape::from([1]),
                    dtype: DType::F32,
                    bytes: DType::F32.itemsize(),
                })
                .unwrap()
                .tensor()
                .storage(),
            &Storage::F32(vec![7.0])
        );
    }

    #[test]
    fn recurrent_failures_leave_runtime_and_cursor_unchanged() {
        let (capture, mut runtime) = fixture(320);
        let initial = capture.initial_recurrent_cursor().unwrap();
        let initial_values = frontier_values(&runtime, &initial);

        let mut rejected = initial.clone();
        let error = capture
            .replay_recurrent_checked(
                &mut runtime,
                &mut rejected,
                &delta(1.0),
                None,
                |outputs, successors| {
                    assert_eq!(outputs[0].storage(), &Storage::F32(vec![1.0, 1.0]));
                    assert_eq!(successors[0].storage(), &Storage::F32(vec![1.0, 1.0]));
                    Err("fixture admission rejection".into())
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::Execute(message) if message == "fixture admission rejection"
        ));
        assert_eq!(rejected, initial);
        assert_eq!(frontier_values(&runtime, &initial), initial_values);

        let mut shadowed = initial.clone();
        let mut inputs = delta(1.0);
        inputs.insert(
            "state".into(),
            TensorData::from_storage([2], Storage::F32(vec![9.0, 9.0])).unwrap(),
        );
        assert!(
            capture
                .replay_recurrent(&mut runtime, &mut shadowed, &inputs, None)
                .is_err()
        );
        assert_eq!(shadowed, initial);
        assert_eq!(frontier_values(&runtime, &initial), initial_values);

        let mut injected = initial.clone();
        assert!(
            capture
                .replay_recurrent(&mut runtime, &mut injected, &delta(1.0), Some(0))
                .is_err()
        );
        assert_eq!(injected, initial);
        assert_eq!(frontier_values(&runtime, &initial), initial_values);

        let mut current = initial.clone();
        capture
            .replay_recurrent(&mut runtime, &mut current, &delta(1.0), None)
            .unwrap();
        let version_one = frontier_values(&runtime, &current);
        let mut stale = initial.clone();
        assert!(
            capture
                .replay_recurrent(&mut runtime, &mut stale, &delta(1.0), None)
                .is_err()
        );
        assert_eq!(stale, initial);
        assert_eq!(frontier_values(&runtime, &current), version_one);

        let mut absent = capture.initial_recurrent_cursor().unwrap();
        assert!(
            capture
                .replay_recurrent(&mut EffectRuntime::new(), &mut absent, &delta(1.0), None)
                .is_err()
        );
        assert_eq!(absent, initial);
    }

    #[test]
    fn prepared_native_recurrent_replacement_borrows_and_commits_once() {
        let (capture, mut runtime) = fixture(321);
        let executor = CapturedReplayExecutor::default();
        let mut cursor = capture.initial_recurrent_cursor().unwrap();
        let inputs = delta(1.0);
        reset_prepared_replay_validation_counts();
        crate::host_buffer::reset_host_bank_transaction_test_counts();
        let mut prepared = capture
            .prepare_recurrent_native(&runtime, &cursor, &inputs, &executor, false)
            .unwrap();
        let sealed_validation_counts = prepared_replay_validation_counts();
        assert!(sealed_validation_counts.mixed_capture_validations > 0);
        assert!(sealed_validation_counts.schedule_rekeys > 0);
        assert!(sealed_validation_counts.identity_serializations > 0);
        assert_eq!(sealed_validation_counts.recurrent_bank_layouts, 1);
        assert_eq!(indexed_recurrent_bank_binding_count(), 0);
        let layout = prepared.recurrent_bank_layout_evidence();
        assert_eq!(layout.buffers, vec![321]);
        assert_eq!(
            layout.buffers,
            cursor
                .frontier()
                .iter()
                .map(|state| state.buffer)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            layout.buffers,
            cursor
                .frontier()
                .iter()
                .map(|state| runtime.slot_identity(state).unwrap().slot)
                .collect::<Vec<_>>()
        );
        assert_eq!(layout.input_ordinals, vec![0]);
        assert_eq!(layout.retained, vec![false]);
        assert_eq!((layout.retained_count, layout.retained_bytes), (0, 0));
        assert_eq!(prepared.structure_validation_count(), 1);
        let prepared_workspace = prepared.workspace_stats();
        assert_eq!(prepared_workspace.binding_layout_build_count, 1);
        assert_eq!(prepared_workspace.input_validation_layout_build_count, 1);
        assert_eq!(prepared_workspace.sealed_input_validator_count, 0);
        assert!(prepared_workspace.sealed_pointer_count > 0);
        assert_eq!(prepared_workspace.last_borrowed_binding_count, 0);
        assert!(prepared_workspace.borrowed_binding_capacity > 0);
        crate::engine::captured_replay::reset_whole_capture_input_validation_scan_count();
        let mut viewed = capture.clone();
        viewed.state_bindings[0].view = Some(crate::AffineView::identity(Shape::from([2])));
        assert!(matches!(
            PreparedRecurrentReplacementPlan::from_capture(&viewed),
            Err(ReplayError::Unsupported(message))
                if message == "prepared recurrent native replacement does not support state input views"
        ));
        let mut incomplete = capture.clone();
        incomplete.value_bindings.clear();
        assert!(matches!(
            PreparedRecurrentReplacementPlan::from_capture(&incomplete),
            Err(ReplayError::Unsupported(message))
                if message == "prepared recurrent native effect is not pure-sourced"
        ));
        let hot_replay_validation_counts = prepared_replay_validation_counts();
        let initial_cursor = cursor.clone();
        let initial_values = frontier_values(&runtime, &cursor);
        let initial_runtime_counts = runtime.recurrent_test_counts();
        let mut wrong_descriptor = cursor.clone();
        wrong_descriptor.frontier[0].shape = Shape::from([1]);
        wrong_descriptor.frontier[0].bytes = DType::F32.itemsize();
        assert!(matches!(
            NativeReplayContext::new(&executor, &mut prepared).replay_recurrent_checked(
                &mut runtime,
                &mut wrong_descriptor,
                &inputs,
                None,
                |_, _| Ok(()),
            ),
            Err(ReplayError::Descriptor(message))
                if message == "recurrent cursor state descriptor mismatch"
        ));
        assert_eq!(indexed_recurrent_bank_binding_count(), 0);
        assert_eq!(runtime.recurrent_test_counts(), initial_runtime_counts);
        let malformed = BTreeMap::from([(
            "delta".into(),
            TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap(),
        )]);
        assert!(matches!(
            NativeReplayContext::new(&executor, &mut prepared).replay_recurrent_checked(
                &mut runtime,
                &mut cursor,
                &malformed,
                None,
                |_, _| Ok(()),
            ),
            Err(ReplayError::Descriptor(_))
        ));
        assert_eq!(cursor, initial_cursor);
        assert_eq!(runtime.recurrent_test_counts(), initial_runtime_counts);
        assert_eq!(frontier_values(&runtime, &cursor), initial_values);
        assert_eq!(prepared.structure_validation_count(), 1);
        assert_eq!(
            crate::engine::captured_replay::whole_capture_input_validation_scan_count(),
            0
        );
        assert_eq!(
            crate::host_buffer::host_bank_transaction_test_counts(),
            crate::host_buffer::HostBankTransactionTestCounts {
                ordered_full_frontier_transactions: 1,
                request_map_builds: 0,
                ordinal_sorts: 0,
            },
            "the malformed external input is rejected inside one atomic ordered transaction"
        );
        assert_eq!(indexed_recurrent_bank_binding_count(), 1);
        let before = runtime.recurrent_test_counts();
        let replay_started = Instant::now();
        let replay = NativeReplayContext::new(&executor, &mut prepared)
            .replay_recurrent_checked(
                &mut runtime,
                &mut cursor,
                &inputs,
                None,
                |outputs, successors| {
                    assert_eq!(outputs[0].storage(), &Storage::F32(vec![1.0, 1.0]));
                    assert_eq!(successors[0], &outputs[0]);
                    Ok(())
                },
            )
            .unwrap();
        assert!(replay.executor_wall_time <= replay_started.elapsed());
        let after = runtime.recurrent_test_counts();
        assert_eq!(after.0, before.0, "native replay must not snapshot state");
        assert_eq!(after.1, before.1 + 1);
        assert_eq!(replay.replay.committed, cursor.frontier());
        assert_eq!(
            replay.replay.native_trace.as_ref(),
            Some(&prepared.preparation_trace().replay)
        );
        assert_eq!(replay.traffic.external_input_import_count, 0);
        assert_eq!(replay.traffic.external_input_import_bytes, 0);
        assert_eq!(replay.traffic.borrowed_recurrent_input_bytes, 8);
        assert_eq!(replay.traffic.borrowed_recurrent_output_bytes, 8);
        assert!(replay.traffic.native_dispatcher_wall_time <= replay.executor_wall_time);
        assert_eq!(indexed_recurrent_bank_binding_count(), 2);
        let replay_workspace = prepared.workspace_stats();
        assert_eq!(replay_workspace.binding_layout_build_count, 1);
        assert!(replay_workspace.last_borrowed_binding_count > 0);
        assert_eq!(
            replay_workspace.borrowed_binding_capacity,
            prepared_workspace.borrowed_binding_capacity
        );
        let expected_traffic = replay.traffic;
        assert_eq!(
            prepared_replay_validation_counts(),
            hot_replay_validation_counts,
            "hot replay must not revalidate, rekey, or hash the sealed capture"
        );
        assert_eq!(prepared.structure_validation_count(), 1);
        assert_eq!(
            crate::engine::captured_replay::whole_capture_input_validation_scan_count(),
            0
        );

        let checkpoint = frontier_values(&runtime, &cursor);
        let failed_cursor = cursor.clone();
        let before_failure = runtime.recurrent_test_counts();
        assert!(
            NativeReplayContext::new(&executor, &mut prepared)
                .replay_recurrent_checked(
                    &mut runtime,
                    &mut cursor,
                    &inputs,
                    None,
                    |_outputs, _successors| Err("reject staged replacement".into()),
                )
                .is_err()
        );
        let after_failure = runtime.recurrent_test_counts();
        assert_eq!(cursor, failed_cursor);
        assert_eq!(after_failure.0, before_failure.0);
        assert_eq!(after_failure.1, before_failure.1);
        assert_eq!(frontier_values(&runtime, &cursor), checkpoint);
        assert_eq!(
            prepared.workspace_stats().borrowed_binding_capacity,
            prepared_workspace.borrowed_binding_capacity
        );

        let before_injected = runtime.recurrent_test_counts();
        NativeReplayContext::new(&executor, &mut prepared)
            .replay_recurrent_checked(&mut runtime, &mut cursor, &inputs, Some(0), |_, _| Ok(()))
            .unwrap_err();
        let after_injected = runtime.recurrent_test_counts();
        assert_eq!(cursor, failed_cursor);
        assert_eq!(after_injected.0, before_injected.0);
        assert_eq!(after_injected.1, before_injected.1);
        assert_eq!(frontier_values(&runtime, &cursor), checkpoint);
        let retried = NativeReplayContext::new(&executor, &mut prepared)
            .replay_recurrent_checked(&mut runtime, &mut cursor, &inputs, None, |_, _| Ok(()))
            .unwrap();
        assert!(retried.traffic.native_dispatcher_wall_time <= retried.executor_wall_time);
        let mut expected_deterministic_traffic = expected_traffic;
        expected_deterministic_traffic.native_dispatcher_wall_time = Duration::ZERO;
        let mut retried_deterministic_traffic = retried.traffic;
        retried_deterministic_traffic.native_dispatcher_wall_time = Duration::ZERO;
        assert_eq!(
            retried_deterministic_traffic,
            expected_deterministic_traffic
        );
        assert_eq!(
            prepared.workspace_stats().last_borrowed_binding_count,
            replay_workspace.last_borrowed_binding_count
        );
        assert_eq!(
            prepared.workspace_stats().borrowed_binding_capacity,
            prepared_workspace.borrowed_binding_capacity
        );
        assert_eq!(
            retried.replay.outputs[0].storage(),
            &Storage::F32(vec![2.0, 2.0])
        );
        assert_eq!(
            frontier_values(&runtime, &cursor)[0].storage(),
            &Storage::F32(vec![2.0, 2.0])
        );
        let current_values = frontier_values(&runtime, &cursor);
        let current_runtime_counts = runtime.recurrent_test_counts();
        let mut stale = failed_cursor;
        assert!(
            NativeReplayContext::new(&executor, &mut prepared)
                .replay_recurrent_checked(&mut runtime, &mut stale, &inputs, None, |_, _| Ok(()),)
                .is_err()
        );
        assert_eq!(runtime.recurrent_test_counts(), current_runtime_counts);
        assert_eq!(frontier_values(&runtime, &cursor), current_values);
        assert_eq!(
            prepared_replay_validation_counts(),
            hot_replay_validation_counts,
            "failure and retry must retain the prepared immutable seal"
        );
        assert_eq!(
            indexed_recurrent_bank_binding_count(),
            5,
            "only callbacks admitted by live runtime validation bind the sealed ordinal bank"
        );
        assert_eq!(prepared.structure_validation_count(), 1);
        assert_eq!(
            crate::engine::captured_replay::whole_capture_input_validation_scan_count(),
            0,
            "sealed success, failure, and retry must use only prepared input validators"
        );
        let host_transactions = crate::host_buffer::host_bank_transaction_test_counts();
        assert_eq!(host_transactions.ordered_full_frontier_transactions, 5);
        assert_eq!(host_transactions.request_map_builds, 0);
        assert_eq!(host_transactions.ordinal_sorts, 0);
    }

    #[test]
    fn prepared_recurrent_bank_layout_rejects_noncanonical_ordinals() {
        let (capture, _runtime) = fixture(330);
        let replacement = PreparedRecurrentReplacementPlan::from_capture(&capture).unwrap();

        let mut missing = replacement.clone();
        missing.replacements.clear();
        assert!(matches!(
            PreparedRecurrentBankLayout::new(&capture.schedule, missing, &[]),
            Err(ReplayError::Corrupt(message))
                if message == "prepared recurrent bank cardinality mismatch"
        ));

        let mut duplicate = replacement.clone();
        let mut duplicate_replacement = duplicate.replacements[0].clone();
        duplicate_replacement.producer += 1;
        duplicate_replacement.step += 1;
        duplicate
            .initial_frontier
            .push(duplicate.initial_frontier[0].clone());
        duplicate.replacements.push(duplicate_replacement);
        assert!(matches!(
            PreparedRecurrentBankLayout::new(&capture.schedule, duplicate, &[]),
            Err(ReplayError::Corrupt(message))
                if message == "prepared recurrent bank binding mismatch"
        ));

        let mut reordered = replacement;
        let mut second_state = reordered.initial_frontier[0].clone();
        second_state.buffer += 1;
        let mut second_replacement = reordered.replacements[0].clone();
        second_replacement.buffer = second_state.buffer;
        second_replacement.producer += 1;
        second_replacement.step += 1;
        reordered.initial_frontier.push(second_state);
        reordered.replacements.push(second_replacement);
        reordered.replacements.swap(0, 1);
        assert!(matches!(
            PreparedRecurrentBankLayout::new(&capture.schedule, reordered, &[]),
            Err(ReplayError::Corrupt(message))
                if message == "prepared recurrent bank binding mismatch"
        ));
    }

    #[test]
    fn prepared_cursor_projection_reuses_authenticated_schema_without_capture_work() {
        let (capture, _runtime) = fixture(331);
        let mut source = capture.initial_recurrent_cursor().unwrap();
        reset_prepared_replay_validation_counts();
        let projection =
            PreparedRecurrentCursorProjection::prepare(&capture, &capture, [331]).unwrap();
        let preparation = prepared_replay_validation_counts();
        assert_eq!(preparation.cursor_projection_preparations, 1);
        assert!(preparation.mixed_capture_validations > 0);
        assert!(preparation.schedule_rekeys > 0);
        assert!(preparation.identity_serializations > 0);
        assert!(preparation.recurrent_frontier_plans > 0);
        assert!(matches!(
            PreparedRecurrentCursorProjection::prepare(&capture, &capture, [331, 331]),
            Err(ReplayError::Descriptor(message))
                if message == "prepared recurrent cursor projection target buffer repeats"
        ));
        reset_prepared_replay_validation_counts();

        source.frontier[0].version = 53;
        let mut first = projection.project(&source).unwrap();
        assert_eq!(first.cursor().capture_identity(), source.capture_identity());
        assert_eq!(first.cursor().frontier()[0].version, 53);
        assert_eq!(prepared_replay_validation_counts(), Default::default());
        first.cursor.frontier[0].version = 54;
        first.publish(&mut source);
        assert_eq!(source.frontier()[0].version, 54);

        // The witness stores only descriptor/ordinal relationships, not a
        // current version floor, so restoring an older valid checkpoint does
        // not require rebuilding the prepared projection.
        source.frontier[0].version = 17;
        let mut second = projection.project(&source).unwrap();
        assert_eq!(second.cursor().frontier()[0].version, 17);
        assert_eq!(prepared_replay_validation_counts(), Default::default());
        second.cursor.frontier[0].version = 18;
        second.publish(&mut source);
        assert_eq!(source.frontier()[0].version, 18);

        let mut malformed = source.clone();
        malformed.frontier[0].bytes += 4;
        assert!(matches!(
            projection.project(&malformed),
            Err(RecurrentCursorProjectionError::InvalidSource(
                ReplayError::Descriptor(message)
            ))
                if message == "prepared recurrent cursor projection source descriptor mismatch"
        ));
        assert_eq!(source.frontier()[0].version, 18);

        let mut incomplete = source.clone();
        incomplete.frontier.clear();
        assert!(matches!(
            projection.project(&incomplete),
            Err(RecurrentCursorProjectionError::IncompleteSourceFrontier)
        ));
        assert_eq!(source.frontier()[0].version, 18);

        let mut overflow = source.clone();
        overflow.frontier[0].version = u64::MAX;
        assert!(matches!(
            projection.project(&overflow),
            Err(RecurrentCursorProjectionError::VersionOverflow)
        ));
        assert_eq!(source.frontier()[0].version, 18);
        assert_eq!(prepared_replay_validation_counts(), Default::default());
    }
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

    #[test]
    fn recurrent_output_selection_is_an_exact_ordered_subset() {
        let requested = [11, 17, 23, 29];
        assert!(validate_requested_selection(&requested, &[]).is_ok());
        assert!(validate_requested_selection(&requested, &[11, 23, 29]).is_ok());
        assert!(validate_requested_selection(&requested, &[23, 17]).is_err());
        assert!(validate_requested_selection(&requested, &[17, 17]).is_err());
        assert!(validate_requested_selection(&requested, &[31]).is_err());
    }

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
            requested_passthroughs: vec![],
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
            requested_materializations: requested_materializations(&base.schedule),
            requested_passthroughs: base.schedule.requested_passthroughs.clone(),
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
            requested_views: BTreeMap::new(),
            projected: BTreeMap::new(),
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
                requested_passthroughs: vec![],
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
    #[cfg(test)]
    record_prepared_replay_validation(|counts| {
        counts.mixed_capture_validations += 1;
        if validate_keys {
            counts.schedule_rekeys += 1;
        }
    });
    if value.schedule.symbolic.is_some() || value.schedule.specialized_from.is_some() {
        return Err(ReplayError::Unsupported(
            "RGSM does not encode symbolic schemas or specialization provenance".into(),
        ));
    }
    if !value.schedule.requested_passthroughs.is_empty() {
        return Err(ReplayError::Unsupported(
            "RGSM does not encode requested passthroughs".into(),
        ));
    }
    let schedule = Schedule {
        items: value.schedule.items.clone(),
        requested_materializations: requested_materializations(&value.schedule),
        requested_passthroughs: value.schedule.requested_passthroughs.clone(),
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
    #[cfg(test)]
    record_prepared_replay_validation(|counts| counts.identity_serializations += 1);
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
