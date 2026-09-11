//! Graph-free CPU replay for static training programs with recurrent state.

mod adamw_checkpoint;
mod module_adamw_checkpoint;
mod state_schema;

#[cfg(test)]
use self::adamw_checkpoint::{
    ADAMW_CHECKPOINT_FORMAT_V1, ADAMW_CHECKPOINT_FORMAT_V2, ADAMW_CHECKPOINT_FORMAT_V3,
    ADAMW_CHECKPOINT_FORMAT_V4, ADAMW_CHECKPOINT_FORMAT_V5, ADAMW_CHECKPOINT_FORMAT_V6,
    ADAMW_CHECKPOINT_FORMAT_V7, ADAMW_CHECKPOINT_FORMAT_V8,
};
use self::adamw_checkpoint::{
    AdamWCheckpointProgress, AdamWCheckpointTensors, decode_adamw_checkpoint,
    encode_adamw_checkpoint,
};
pub use self::adamw_checkpoint::{CompiledAdamWCheckpoint, CompiledAdamWCheckpointInfo};
pub use self::module_adamw_checkpoint::CompiledModuleAdamWCheckpoint;
use self::module_adamw_checkpoint::{
    DecodedModuleAdamWCheckpoint, ModuleCheckpointState, ModuleCheckpointStateKind,
    ModuleCheckpointVisit, decode_module_adamw_checkpoint, encode_module_adamw_checkpoint,
};
use self::state_schema::{
    AdamWGlobalState, AdamWParameterState, INTERNAL_PREFIX, RecurrentStateKey, StateSpec,
};
use super::native_training_scoreboard::CompiledAdamWInspection;
use super::target::{
    ConfiguredCpuSessionTarget, CpuNonFinitePolicy, CpuSessionTarget, MetalSessionTarget,
    NativeCpuSessionTarget, SessionTarget,
};
use crate::engine::mixed_capture::{NativeReplayContext, PreparedRecurrentNativeReplay};
use crate::engine::{NativeReplayTraffic, PlannedNativeItems};
use crate::nn::{
    Parameter, ParameterRestore, ParameterSnapshot, StateKind, TrainingDropoutProvider,
    next_version, restore_parameters,
};
use crate::runtime::metal::{
    MetalDevice, MetalDeviceRun, MetalDeviceRunReport, MetalDeviceSession,
    MetalDeviceSessionSummary, MetalError, MetalFixedStateReadPlan, MetalFixedStateReadSession,
    MetalFixedStateTransitionPlan, MetalFixedStateTransitionSession, MetalRenderer,
    MetalScoreboardContext, MetalScoreboardError, MetalScoreboardObserver, MetalSessionScoreboard,
    MetalSessionScoreboardReport, MetalStatefulInferencePlan, RenderedMetal,
};
use crate::{
    BufferState, CapturedMixedSchedule, CapturedReplayExecutor, CapturedSchedule,
    CapturedStatefulInference, CompareOp, DType, EffectGraph, EffectRuntime, Error,
    ExecutionPlanSummary, Graph, InferenceStateLink, LoadReport, MixedReplayCursor, Module,
    NativeMixedReplayTrace, NodeId, ParameterId, ReplayError, Result, Scalar, Schedule,
    ScheduleStateBinding, ScheduleValueBinding, Shape, TensorData, bind_schedule_states,
    combine_mixed_schedules, schedule_effects, schedule_many,
};
#[cfg(test)]
use crate::{load_safetensors, save_safetensors};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    time::{Duration, Instant},
};

const LEARNING_RATE_INPUT: &str = "__rustgrad_compiled_training_learning_rate";
const STATE_BUFFER_BASE: u64 = 1_u64 << 62;
const MAX_EXACT_F32_INTEGER_COUNT: u64 = 1_u64 << 24;

/// Immutable two-word key for compiled Transformer residual dropout.
///
/// The first word occupies the low 32 bits of Threefry's packed U64 key and
/// the second occupies the high 32 bits. This is deliberately an explicit key,
/// not a `u64` seed with an implicit or lossy mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledDropoutKey(pub [u32; 2]);

impl CompiledDropoutKey {
    pub const fn words(self) -> [u32; 2] {
        self.0
    }

    const fn packed(self) -> u64 {
        self.0[0] as u64 | ((self.0[1] as u64) << 32)
    }
}

/// Fixed compiled residual-dropout policy for one AdamW Transformer capture.
///
/// This policy defines a workload-specific U64-block stream with interleaved
/// low/high U32 words and low-mantissa F32 conversion. It is intentionally not
/// sequence-compatible with [`crate::RandomStream`] or its uniform conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledDropoutConfig {
    key: CompiledDropoutKey,
}

impl CompiledDropoutConfig {
    pub const fn new(key: CompiledDropoutKey) -> Self {
        Self { key }
    }

    pub const fn key(self) -> CompiledDropoutKey {
        self.key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompiledDropoutState {
    config: CompiledDropoutConfig,
    blocks_per_replay: u64,
}

struct CompiledDropoutStream {
    counter: NodeId,
    key: CompiledDropoutKey,
    reserved_blocks: u64,
}

impl CompiledDropoutStream {
    fn new(counter: NodeId, config: CompiledDropoutConfig) -> Self {
        Self {
            counter,
            key: config.key,
            reserved_blocks: 0,
        }
    }

    fn finish(self, graph: &mut Graph) -> Result<(NodeId, CompiledDropoutState)> {
        if self.reserved_blocks == 0 {
            return Err(training(
                "compiled dropout configuration produced no active F32 draw",
            ));
        }
        let increment =
            graph.full_with_dtype(Shape::from([]), Scalar::U(self.reserved_blocks), DType::U64)?;
        let successor = graph.add(self.counter, increment)?;
        Ok((
            successor,
            CompiledDropoutState {
                config: CompiledDropoutConfig::new(self.key),
                blocks_per_replay: self.reserved_blocks,
            },
        ))
    }
}

fn expected_dropout_counter(dropout: CompiledDropoutState, replay_step: u64) -> Result<u64> {
    replay_step
        .checked_mul(dropout.blocks_per_replay)
        .ok_or_else(|| training("compiled dropout block counter would overflow"))
}

impl TrainingDropoutProvider for CompiledDropoutStream {
    fn dropout(&mut self, graph: &mut Graph, input: NodeId, probability: f64) -> Result<NodeId> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(training("compiled dropout probability must be in [0, 1]"));
        }
        if graph.dtype(input)? != DType::F32 {
            return Err(training("compiled dropout requires F32 input"));
        }
        let shape = graph.shape(input)?.clone();
        let elements = shape.numel()?;
        if elements == 0 || probability == 0.0 {
            return Ok(input);
        }
        if probability == 1.0 {
            return graph.zeros_with_dtype(shape, DType::F32);
        }

        let blocks = elements
            .checked_add(1)
            .ok_or_else(|| training("compiled dropout block count overflow"))?
            / 2;
        let blocks =
            u64::try_from(blocks).map_err(|_| training("compiled dropout block count overflow"))?;
        let start = self.reserved_blocks;
        self.reserved_blocks = start
            .checked_add(blocks)
            .ok_or_else(|| training("compiled dropout reservation overflow"))?;
        let offsets = (start..self.reserved_blocks)
            .map(Scalar::U)
            .collect::<Vec<_>>();
        let offsets = graph.constant(TensorData::from_scalars(
            Shape::new([offsets.len()]),
            DType::U64,
            offsets,
        )?);
        let counters = graph.add(self.counter, offsets)?;
        let key =
            graph.full_with_dtype(Shape::from([]), Scalar::U(self.key.packed()), DType::U64)?;
        let blocks = graph.threefry(counters, key)?;
        let words = graph.bitcast(blocks, DType::U32)?;
        let words = graph.shrink(words, vec![(0, elements)])?;
        let words = graph.reshape(words, shape.clone())?;
        let mantissa = graph.bitwise_and_scalar(words, Scalar::U(0x007f_ffff))?;
        let bits = graph.bitwise_or_scalar(mantissa, Scalar::U(0x3f80_0000))?;
        let unit = graph.bitcast(bits, DType::F32)?;
        let one = graph.full_with_dtype(Shape::from([]), Scalar::F(1.0), DType::F32)?;
        let unit = graph.sub(unit, one)?;
        let threshold =
            graph.full_with_dtype(Shape::from([]), Scalar::F(probability), DType::F32)?;
        let keep = graph.ge(unit, threshold)?;
        let keep = graph.contiguous(keep)?;
        let zero = graph.full_with_dtype(Shape::from([]), Scalar::F(0.0), DType::F32)?;
        let masked = graph.select(keep, input, zero)?;
        let denominator =
            graph.full_with_dtype(Shape::from([]), Scalar::F(1.0 - probability), DType::F32)?;
        graph.div(masked, denominator)
    }
}

/// Detached initial value for one compiled training parameter.
///
/// Construction does not create an [`crate::nn::Parameter`] or retain a live
/// module handle. The value is consumed by compilation and subsequently owned
/// only by the compiled session's [`EffectRuntime`].
#[derive(Clone, Debug)]
pub struct TrainingParameterInit {
    name: String,
    value: TensorData,
}

impl TrainingParameterInit {
    pub fn new(name: impl Into<String>, value: TensorData) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "parameter")?;
        if value.dtype() != DType::F32 {
            return Err(training("compiled training parameters must be F32"));
        }
        checked_bytes(&value)?;
        Ok(Self { name, value })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> &TensorData {
        &self.value
    }
}

#[derive(Clone, Debug)]
struct ModuleParameterEntry {
    identity: ParameterId,
    name: String,
    value: TensorData,
    source_trainable: bool,
    trainable: bool,
    policy_frozen: bool,
}

/// Frozen snapshot of one module's parameter topology for compilation.
///
/// Trainable identities become recurrent optimizer-owned inputs. Frozen
/// parameters and buffers become immutable capture constants. Tied traversal
/// entries keep one identity and therefore resolve to exactly one graph node.
#[derive(Clone, Debug)]
struct ModuleParameterPlan {
    entries: Vec<ModuleParameterEntry>,
    noncanonical_names: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SealedModuleVisit {
    name: String,
    identity: ParameterId,
    kind: StateKind,
    trainable: bool,
}

#[derive(Clone, Debug)]
struct SealedModuleState {
    name: String,
    parameter: Parameter,
    snapshot: ParameterSnapshot,
    kind: StateKind,
    source_trainable: bool,
    trainable: bool,
    publication_value: Option<TensorData>,
}

/// Complete host-module state retained while an owned compiled session runs.
///
/// The seal is deliberately private: it is meaningful only together with the
/// exact module value consumed by [`CompiledModuleAdamWPlan`].
#[derive(Clone, Debug)]
struct CompiledModuleSeal {
    visits: Vec<SealedModuleVisit>,
    states: BTreeMap<ParameterId, SealedModuleState>,
    frozen_parameters: BTreeSet<String>,
}

impl CompiledModuleSeal {
    fn capture(
        module: &(impl Module + ?Sized),
        frozen_parameters: &BTreeSet<String>,
    ) -> Result<Self> {
        let mut names = BTreeSet::new();
        let mut visits = Vec::new();
        let mut states = BTreeMap::<ParameterId, SealedModuleState>::new();
        let mut error = None;
        module.visit("", &mut |name, parameter, kind| {
            if error.is_some() {
                return;
            }
            if !names.insert(name.clone()) {
                error = Some(training("compiled module state names repeat"));
                return;
            }
            let snapshot = match parameter.snapshot() {
                Ok(snapshot) => snapshot,
                Err(err) => {
                    error = Some(err);
                    return;
                }
            };
            let trainable = snapshot.trainable && matches!(kind, StateKind::Parameter);
            visits.push(SealedModuleVisit {
                name: name.clone(),
                identity: snapshot.identity,
                kind,
                trainable,
            });
            if let Some(first) = states.get(&snapshot.identity) {
                let same_bytes = first.snapshot.data.to_le_bytes().and_then(|expected| {
                    snapshot.data.to_le_bytes().map(|actual| expected == actual)
                });
                match same_bytes {
                    Ok(true)
                        if first.kind == kind
                            && first.snapshot.trainable == snapshot.trainable
                            && first.snapshot.shape == snapshot.shape
                            && first.snapshot.dtype == snapshot.dtype
                            && first.snapshot.version == snapshot.version => {}
                    Ok(_) => {
                        error = Some(training(
                            "tied compiled module state has inconsistent kind or snapshot",
                        ));
                    }
                    Err(err) => error = Some(err),
                }
                return;
            }
            states.insert(
                snapshot.identity,
                SealedModuleState {
                    name,
                    parameter: parameter.clone(),
                    kind,
                    source_trainable: snapshot.trainable,
                    snapshot,
                    trainable,
                    publication_value: None,
                },
            );
        });
        match error {
            Some(error) => Err(error),
            None => {
                let mut seal = Self {
                    visits,
                    states,
                    frozen_parameters: frozen_parameters.clone(),
                };
                seal.apply_frozen_parameters()?;
                for state in seal.states.values().filter(|state| state.trainable) {
                    next_version(state.snapshot.version)?;
                }
                Ok(seal)
            }
        }
    }

    fn apply_frozen_parameters(&mut self) -> Result<()> {
        let noncanonical_names = self
            .visits
            .iter()
            .filter(|visit| self.states[&visit.identity].name != visit.name)
            .map(|visit| visit.name.clone())
            .collect::<BTreeSet<_>>();
        for name in &self.frozen_parameters {
            if noncanonical_names.contains(name) {
                return Err(training(
                    "compiled AdamW frozen parameter is a noncanonical tied alias",
                ));
            }
            match self.states.values_mut().find(|state| state.name == *name) {
                Some(state) if state.trainable => state.trainable = false,
                Some(_) => {
                    return Err(training(
                        "compiled AdamW frozen parameter is not a trainable parameter",
                    ));
                }
                None => {
                    return Err(training("compiled AdamW frozen parameter name is unknown"));
                }
            }
        }
        if self.states.values().all(|state| !state.trainable) {
            return Err(training(
                "compiled module needs at least one effectively trainable parameter",
            ));
        }
        Ok(())
    }

    fn validate_unchanged(&self, module: &(impl Module + ?Sized)) -> Result<()> {
        let current = Self::capture(module, &self.frozen_parameters)?;
        if current.visits != self.visits || current.states.keys().ne(self.states.keys()) {
            return Err(training("owned compiled module topology changed"));
        }
        for (identity, expected) in &self.states {
            let actual = &current.states[identity];
            if actual.name != expected.name
                || actual.kind != expected.kind
                || actual.trainable != expected.trainable
                || actual.snapshot.shape != expected.snapshot.shape
                || actual.snapshot.dtype != expected.snapshot.dtype
                || actual.snapshot.version != expected.snapshot.version
                || actual.snapshot.input_name != expected.snapshot.input_name
                || actual.snapshot.data.to_le_bytes()? != expected.snapshot.data.to_le_bytes()?
            {
                return Err(training("owned compiled module state changed while sealed"));
            }
        }
        Ok(())
    }

    fn checkpoint_inventory(&self) -> (Vec<ModuleCheckpointState>, Vec<ModuleCheckpointVisit>) {
        let mut states = Vec::with_capacity(self.states.len());
        let mut visits = Vec::with_capacity(self.visits.len());
        let mut seen = BTreeSet::new();
        for visit in &self.visits {
            let state = &self.states[&visit.identity];
            visits.push(ModuleCheckpointVisit {
                name: visit.name.clone(),
                canonical_name: state.name.clone(),
            });
            if seen.insert(visit.identity) {
                states.push(ModuleCheckpointState {
                    name: state.name.clone(),
                    kind: match state.kind {
                        StateKind::Parameter => ModuleCheckpointStateKind::Parameter,
                        StateKind::Buffer => ModuleCheckpointStateKind::Buffer,
                    },
                    source_trainable: state.source_trainable,
                    policy_frozen: self.frozen_parameters.contains(&state.name),
                    value: (!state.trainable).then(|| {
                        state
                            .publication_value
                            .clone()
                            .unwrap_or_else(|| state.snapshot.data.clone())
                    }),
                });
            }
        }
        (states, visits)
    }

    fn apply_module_checkpoint(
        &mut self,
        checkpoint: &DecodedModuleAdamWCheckpoint,
    ) -> Result<BTreeMap<String, TensorData>> {
        let (current_states, current_visits) = self.checkpoint_inventory();
        if current_visits != checkpoint.visits || current_states.len() != checkpoint.states.len() {
            return Err(training("compiled module checkpoint topology mismatch"));
        }
        let optimizer_parameters =
            decode_adamw_checkpoint(checkpoint.optimizer.as_bytes())?.parameters;
        let mut immutable_values = BTreeMap::new();
        for (current, saved) in current_states.iter().zip(&checkpoint.states) {
            if current.name != saved.name
                || current.kind != saved.kind
                || current.source_trainable != saved.source_trainable
                || current.policy_frozen != saved.policy_frozen
                || current.trainable() != saved.trainable()
            {
                return Err(training(
                    "compiled module checkpoint state topology mismatch",
                ));
            }
            let saved_value = match &saved.value {
                Some(value) => value,
                None => &optimizer_parameters[&saved.name],
            };
            let state = self
                .states
                .values_mut()
                .find(|state| state.name == saved.name)
                .ok_or_else(|| training("compiled module checkpoint state is absent"))?;
            if saved_value.shape() != &state.snapshot.shape
                || saved_value.dtype() != state.snapshot.dtype
            {
                return Err(training(
                    "compiled module checkpoint state descriptor mismatch",
                ));
            }
            checked_bytes(saved_value)?;
            if let Some(value) = &saved.value {
                next_version(state.snapshot.version)?;
                state.publication_value = Some(value.clone());
                immutable_values.insert(saved.name.clone(), value.clone());
            }
        }
        Ok(immutable_values)
    }

    fn parameter_plan(&self, module: &(impl Module + ?Sized)) -> Result<ModuleParameterPlan> {
        let immutable_values = self
            .checkpoint_inventory()
            .0
            .into_iter()
            .filter_map(|state| state.value.map(|value| (state.name, value)))
            .collect::<BTreeMap<_, _>>();
        ModuleParameterPlan::new(module, &self.frozen_parameters)?
            .with_immutable_values(&immutable_values)
    }

    fn publish(
        &self,
        module: &(impl Module + ?Sized),
        parameters: &BTreeMap<String, TensorData>,
    ) -> Result<LoadReport> {
        self.validate_unchanged(module)?;
        let trainable = self
            .states
            .values()
            .filter(|state| state.trainable)
            .map(|state| (state.name.as_str(), state))
            .collect::<BTreeMap<_, _>>();
        if parameters.len() != trainable.len()
            || parameters
                .keys()
                .map(String::as_str)
                .ne(trainable.keys().copied())
        {
            return Err(training(
                "owned compiled module trainable parameter names changed",
            ));
        }

        let mut restores = Vec::with_capacity(self.states.len());
        let mut loaded_keys = Vec::with_capacity(trainable.len());
        for state in self.states.values() {
            let (data, restored_version) = if state.trainable {
                let value = &parameters[&state.name];
                if value.shape() != &state.snapshot.shape || value.dtype() != state.snapshot.dtype {
                    return Err(training(
                        "owned compiled module trainable parameter descriptor changed",
                    ));
                }
                loaded_keys.push(state.name.clone());
                (value.clone(), next_version(state.snapshot.version)?)
            } else {
                match &state.publication_value {
                    Some(value) => (value.clone(), next_version(state.snapshot.version)?),
                    None => (state.snapshot.data.clone(), state.snapshot.version),
                }
            };
            restores.push(ParameterRestore {
                parameter: state.parameter.clone(),
                data,
                expected_version: state.snapshot.version,
                restored_version,
            });
        }
        restore_parameters(restores)?;
        loaded_keys.sort();
        Ok(LoadReport {
            loaded_keys,
            ..LoadReport::default()
        })
    }
}

impl ModuleParameterPlan {
    fn new(module: &(impl Module + ?Sized), frozen_parameters: &BTreeSet<String>) -> Result<Self> {
        let mut entries = Vec::<ModuleParameterEntry>::new();
        let mut identities = BTreeMap::<ParameterId, usize>::new();
        let mut names = BTreeSet::new();
        let mut noncanonical_names = BTreeSet::new();
        let mut error = None;
        module.visit("", &mut |name, parameter, kind| {
            if error.is_some() {
                return;
            }
            if !names.insert(name.clone()) {
                error = Some(training("compiled module state names repeat"));
                return;
            }
            let identity = parameter.id();
            if let Some(&index) = identities.get(&identity) {
                let is_parameter = matches!(kind, StateKind::Parameter);
                if entries[index].source_trainable != (parameter.is_trainable() && is_parameter) {
                    error = Some(training(
                        "tied compiled module state has inconsistent trainability",
                    ));
                } else {
                    noncanonical_names.insert(name);
                }
                return;
            }
            match parameter.snapshot() {
                Ok(snapshot) => {
                    let trainable = snapshot.trainable && matches!(kind, StateKind::Parameter);
                    identities.insert(identity, entries.len());
                    entries.push(ModuleParameterEntry {
                        identity,
                        name,
                        value: snapshot.data,
                        source_trainable: trainable,
                        trainable,
                        policy_frozen: false,
                    });
                }
                Err(err) => error = Some(err),
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        for name in frozen_parameters {
            if noncanonical_names.contains(name) {
                return Err(training(
                    "compiled AdamW frozen parameter is a noncanonical tied alias",
                ));
            }
            match entries.iter_mut().find(|entry| entry.name == *name) {
                Some(entry) if entry.trainable => {
                    entry.trainable = false;
                    entry.policy_frozen = true;
                }
                Some(_) => {
                    return Err(training(
                        "compiled AdamW frozen parameter is not a trainable parameter",
                    ));
                }
                None => {
                    return Err(training("compiled AdamW frozen parameter name is unknown"));
                }
            }
        }
        if entries.iter().all(|entry| !entry.trainable) {
            return Err(training(
                "compiled module needs at least one effectively trainable parameter",
            ));
        }
        Ok(Self {
            entries,
            noncanonical_names,
        })
    }

    fn validate_weight_decay_exclusions(&self, config: &CompiledAdamWConfig) -> Result<()> {
        for name in &config.weight_decay_exclusions {
            if self.noncanonical_names.contains(name) {
                return Err(training(
                    "compiled AdamW weight-decay exclusion is a noncanonical tied alias",
                ));
            }
            match self
                .entries
                .iter()
                .find(|entry| entry.name == name.as_str())
            {
                Some(entry) if entry.trainable => {}
                Some(_) => {
                    return Err(training(
                        "compiled AdamW weight-decay exclusion is not a trainable parameter",
                    ));
                }
                None => {
                    return Err(training(
                        "compiled AdamW weight-decay exclusion name is unknown",
                    ));
                }
            }
        }
        Ok(())
    }

    fn with_immutable_values(mut self, values: &BTreeMap<String, TensorData>) -> Result<Self> {
        let expected = self
            .entries
            .iter()
            .filter(|entry| !entry.trainable)
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>();
        if expected != values.keys().map(String::as_str).collect::<BTreeSet<_>>() {
            return Err(training(
                "compiled module checkpoint immutable inventory mismatch",
            ));
        }
        for entry in self.entries.iter_mut().filter(|entry| !entry.trainable) {
            let value = &values[&entry.name];
            if value.shape() != entry.value.shape() || value.dtype() != entry.value.dtype() {
                return Err(training(
                    "compiled module checkpoint immutable descriptor mismatch",
                ));
            }
            entry.value = value.clone();
        }
        Ok(self)
    }

    fn initial_parameters(&self) -> Result<Vec<TrainingParameterInit>> {
        self.entries
            .iter()
            .filter(|entry| entry.trainable)
            .map(|entry| TrainingParameterInit::new(entry.name.clone(), entry.value.clone()))
            .collect()
    }

    fn lower<T>(
        &self,
        graph: &mut Graph,
        parameters: &BTreeMap<String, NodeId>,
        build: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        self.lower_impl(graph, parameters, None, build)
    }

    fn lower_with_frozen_parameter_nodes<T>(
        &self,
        graph: &mut Graph,
        parameters: &BTreeMap<String, NodeId>,
        frozen_parameter_nodes: &mut BTreeSet<NodeId>,
        build: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        self.lower_impl(graph, parameters, Some(frozen_parameter_nodes), build)
    }

    fn lower_impl<T>(
        &self,
        graph: &mut Graph,
        parameters: &BTreeMap<String, NodeId>,
        mut frozen_parameter_nodes: Option<&mut BTreeSet<NodeId>>,
        build: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        let mut overrides = BTreeMap::new();
        for entry in &self.entries {
            let node = if entry.trainable {
                parameters
                    .get(&entry.name)
                    .copied()
                    .ok_or_else(|| training("compiled module parameter set mismatch"))?
            } else {
                let node = graph.constant(entry.value.clone());
                if entry.policy_frozen
                    && let Some(nodes) = frozen_parameter_nodes.as_deref_mut()
                {
                    nodes.insert(node);
                }
                node
            };
            if graph.shape(node)? != entry.value.shape()
                || graph.dtype(node)? != entry.value.dtype()
                || graph.requires_grad(node)? != entry.trainable
            {
                return Err(training("compiled module parameter descriptor mismatch"));
            }
            overrides.insert(
                entry.identity,
                (node, entry.source_trainable, entry.trainable),
            );
        }
        graph.with_parameter_overrides(overrides, build)
    }
}

/// Static compilation policy for [`CpuCompiledMomentumSgd`].
#[derive(Clone, Debug)]
pub struct CompiledMomentumSgdConfig {
    momentum: f32,
    inputs: BTreeMap<String, (Shape, DType)>,
}

impl CompiledMomentumSgdConfig {
    /// Creates the source-style momentum rule `v = momentum*v + grad`.
    /// Tinygrad rejects only ordered negative momentum, so NaN and infinity
    /// remain ordinary graph constants rather than receiving an invented
    /// finite-value restriction here.
    pub fn new(momentum: f32) -> Result<Self> {
        if momentum < 0.0 {
            return Err(training(
                "compiled momentum-SGD momentum must be nonnegative",
            ));
        }
        Ok(Self {
            momentum,
            inputs: BTreeMap::new(),
        })
    }

    /// Adds one exact external input descriptor. Names are deterministic and
    /// may not overlap the session's private state/LR namespace.
    pub fn with_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, dtype)?;
        if self.inputs.insert(name, (shape, dtype)).is_some() {
            return Err(training("duplicate compiled training input name"));
        }
        Ok(self)
    }

    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&str, &Shape, DType)> {
        self.inputs
            .iter()
            .map(|(name, (shape, dtype))| (name.as_str(), shape, *dtype))
    }
}

/// An immutable learning-rate schedule captured by a compiled AdamW program.
///
/// The rate starts at `base` and is multiplied by `gamma` once for each
/// milestone less than or equal to the number of already completed optimizer
/// updates. Milestones are completed-update boundaries:
/// milestone one first changes the second update, independently of microbatch
/// accumulation.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledMultiStepLr {
    base: f32,
    gamma: f32,
    milestones: Vec<u64>,
}

impl CompiledMultiStepLr {
    pub fn new(base: f32, gamma: f32, milestones: impl IntoIterator<Item = u64>) -> Result<Self> {
        if !base.is_finite() || base < 0.0 {
            return Err(training(
                "compiled MultiStep learning-rate base must be finite and nonnegative",
            ));
        }
        if !gamma.is_finite() || gamma < 0.0 {
            return Err(training(
                "compiled MultiStep learning-rate gamma must be finite and nonnegative",
            ));
        }
        let milestones = milestones.into_iter().collect::<Vec<_>>();
        if milestones
            .iter()
            .any(|milestone| *milestone == 0 || *milestone == u64::MAX)
        {
            return Err(training(
                "compiled MultiStep learning-rate milestones must be positive and below u64::MAX",
            ));
        }
        if milestones.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(training(
                "compiled MultiStep learning-rate milestones must be strictly increasing",
            ));
        }
        let mut rate = base;
        for _ in &milestones {
            rate *= gamma;
            if !rate.is_finite() {
                return Err(training(
                    "compiled MultiStep learning-rate values must remain finite",
                ));
            }
        }
        Ok(Self {
            base,
            gamma,
            milestones,
        })
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    pub fn milestones(&self) -> &[u64] {
        &self.milestones
    }
}

#[derive(Clone, Debug, PartialEq)]
enum CompiledLearningRatePolicy {
    External,
    MultiStep(CompiledMultiStepLr),
}

impl CompiledLearningRatePolicy {
    fn require_external(&self) -> Result<()> {
        if matches!(self, Self::External) {
            Ok(())
        } else {
            Err(training(
                "compiled AdamW external learning-rate entrypoint requires external policy",
            ))
        }
    }

    fn require_scheduled(&self) -> Result<()> {
        if matches!(self, Self::MultiStep(_)) {
            Ok(())
        } else {
            Err(training(
                "compiled AdamW scheduled learning-rate entrypoint requires compiled MultiStep policy",
            ))
        }
    }
}

/// Static compilation policy for [`CpuCompiledAdamW`].
#[derive(Clone, Debug)]
pub struct CompiledAdamWConfig {
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    gradient_accumulation_steps: u64,
    token_weight_mask_input: Option<String>,
    max_gradient_norm: Option<f32>,
    clip_report: bool,
    window_loss_report: bool,
    loss_scale: f32,
    frozen_parameters: BTreeSet<String>,
    weight_decay_exclusions: BTreeSet<String>,
    inputs: BTreeMap<String, (Shape, DType)>,
    host_token_inputs: BTreeMap<String, Shape>,
    learning_rate: CompiledLearningRatePolicy,
}

impl CompiledAdamWConfig {
    pub fn new(beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Result<Self> {
        if !(0.0..1.0).contains(&beta1)
            || !(0.0..1.0).contains(&beta2)
            || !eps.is_finite()
            || eps <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return Err(training(
                "compiled AdamW requires beta1/beta2 in [0,1), positive finite epsilon, and finite nonnegative weight decay",
            ));
        }
        Ok(Self {
            beta1,
            beta2,
            eps,
            weight_decay,
            gradient_accumulation_steps: 1,
            token_weight_mask_input: None,
            max_gradient_norm: None,
            clip_report: false,
            window_loss_report: false,
            loss_scale: 1.0,
            frozen_parameters: BTreeSet::new(),
            weight_decay_exclusions: BTreeSet::new(),
            inputs: BTreeMap::new(),
            host_token_inputs: BTreeMap::new(),
            learning_rate: CompiledLearningRatePolicy::External,
        })
    }

    /// Accumulates gradients across exactly `steps` recurrent replays before
    /// committing one averaged AdamW update. Parameters, moments, the
    /// optimizer step, the partial gradient sums, and the accumulation cursor
    /// all remain inside the captured state frontier.
    pub fn with_gradient_accumulation(mut self, steps: u64) -> Result<Self> {
        if steps == 0 {
            return Err(training(
                "compiled AdamW gradient accumulation steps must be positive",
            ));
        }
        if let Some(mask_input) = &self.token_weight_mask_input {
            validate_token_weighted_accumulation(&self.inputs, mask_input, steps)?;
        }
        self.gradient_accumulation_steps = steps;
        Ok(self)
    }

    /// Configures an existing fixed F32 binary mask for compiler-owned token
    /// mean loss and valid-token-weighted gradient accumulation.
    ///
    /// This CPU-first opt-in requires accumulation and an explicit token-mean
    /// objective through [`CompiledAdamWPlan::compile_module_graph`] or its
    /// dropout variant. The compatibility token-mean constructor follows the
    /// same lowering. Compilation derives the scalar loss from per-token
    /// losses, then weights each normalized microbatch gradient by its valid
    /// count and divides by the whole window count immediately before clipping
    /// and AdamW. It adds one recurrent U64 count; mask padding layout remains
    /// a batch-level policy.
    pub fn with_token_weighted_gradient_accumulation(
        mut self,
        mask_input_name: impl Into<String>,
    ) -> Result<Self> {
        let mask_input_name = mask_input_name.into();
        if self.token_weight_mask_input.is_some() {
            return Err(training(
                "compiled AdamW token-weighted accumulation policy repeats",
            ));
        }
        validate_token_weighted_accumulation(
            &self.inputs,
            &mask_input_name,
            self.gradient_accumulation_steps,
        )?;
        self.token_weight_mask_input = Some(mask_input_name);
        Ok(self)
    }

    /// Clips the complete ordered parameter-gradient set to one global L2
    /// norm inside the compiled graph. With gradient accumulation, clipping is
    /// applied once to the averaged window immediately before AdamW updates;
    /// individual microbatch gradients are never clipped independently.
    /// The squared total is committed at F32 width before its square root so
    /// interpreter and strict-native CPU overflow semantics are identical; that
    /// storage boundary is part of the captured program identity.
    pub fn with_max_gradient_norm(mut self, max_norm: f32) -> Result<Self> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Err(training(
                "compiled AdamW maximum gradient norm must be positive and finite",
            ));
        }
        self.max_gradient_norm = Some(max_norm);
        Ok(self)
    }

    /// Requests CPU step and partial-flush results to report the complete
    /// pre-clip global gradient norm and the exact scale applied before AdamW.
    /// Accumulation-only steps and empty flushes report no completed window.
    /// Under [`CpuNonFinitePolicy::RejectTransition`], only reports belonging
    /// to a committing full window or explicit partial flush participate in
    /// admission. The default remains disabled and adds no graph outputs.
    pub fn with_clip_report(mut self) -> Self {
        self.clip_report = true;
        self
    }

    /// Retains the exact accumulated loss numerator inside the captured AdamW
    /// frontier and reports one aggregate mean only when a full or explicitly
    /// flushed window commits. Ordinary scalar objectives use equal
    /// microbatch weights; compiler-owned token means use the same validated
    /// token counts as gradient accumulation. `zero_grad` discards both
    /// gradients and the pending loss numerator atomically.
    pub fn with_window_loss_report(mut self) -> Self {
        self.window_loss_report = true;
        self
    }

    /// Scales the differentiation root by a fixed finite factor, then
    /// unscales the complete F32 parameter-gradient set before accumulation,
    /// clipping, and AdamW. The public loss remains the original unscaled
    /// scalar. A scale of one is canonical and adds no graph nodes.
    pub fn with_loss_scale(mut self, scale: f32) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(training(
                "compiled AdamW loss scale must be positive and finite",
            ));
        }
        self.loss_scale = scale;
        Ok(self)
    }

    /// Captures an immutable MultiStep learning-rate policy in the program.
    /// Scheduled CPU replay then uses the explicit no-learning-rate methods;
    /// the default remains a caller-supplied scalar on every replay.
    pub fn with_captured_multi_step_lr(mut self, schedule: CompiledMultiStepLr) -> Self {
        self.learning_rate = CompiledLearningRatePolicy::MultiStep(schedule);
        self
    }

    /// Freezes exact canonical module parameter names for this compilation.
    /// The policy is resolved by parameter identity without mutating the
    /// source module. Raw [`TrainingParameterInit`] compilation rejects a
    /// nonempty policy because it has no module topology to authenticate.
    pub fn with_frozen_parameters<I, S>(mut self, names: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in names {
            let name = name.into();
            validate_user_name(&name, "AdamW frozen parameter")?;
            if !self.frozen_parameters.insert(name) {
                return Err(training("duplicate compiled AdamW frozen parameter name"));
            }
        }
        Ok(self)
    }

    /// Excludes exact canonical trainable parameter names from decoupled
    /// weight decay. Names are accumulated across calls and duplicates are
    /// rejected; module compilation also rejects frozen state, buffers, tied
    /// aliases, and names outside the canonical trainable inventory.
    pub fn with_weight_decay_exclusions<I, S>(mut self, names: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in names {
            let name = name.into();
            validate_user_name(&name, "AdamW weight-decay exclusion")?;
            if !self.weight_decay_exclusions.insert(name) {
                return Err(training(
                    "duplicate compiled AdamW weight-decay exclusion name",
                ));
            }
        }
        Ok(self)
    }

    pub fn with_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, dtype)?;
        if self.inputs.insert(name, (shape, dtype)).is_some() {
            return Err(training("duplicate compiled training input name"));
        }
        Ok(self)
    }

    /// Declares one nonempty fixed-shape rank-two I32 token batch whose exact
    /// raw F32 Gather and first-order ScatterAdd VJP may be authenticated for
    /// status-free Metal replay. Module compilation may instead authenticate
    /// a sole forward Gather when its data is an exact policy-frozen parameter.
    /// The input name is declared atomically, so it collides with
    /// [`Self::with_input`] in either call order.
    pub fn with_host_token_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, DType::I32)?;
        if shape.rank() != 2 || shape.dims().contains(&0) {
            return Err(training(
                "compiled host token input must be nonempty fixed rank-two I32",
            ));
        }
        if self.inputs.contains_key(&name) || self.host_token_inputs.contains_key(&name) {
            return Err(training("duplicate compiled training input name"));
        }
        self.inputs
            .insert(name.clone(), (shape.clone(), DType::I32));
        self.host_token_inputs.insert(name, shape);
        Ok(self)
    }

    /// Declares the complete fixed external schema of a typed workload batch.
    pub fn with_input_batch<B>(mut self) -> Result<Self>
    where
        B: CompiledInputBatch,
    {
        for spec in B::schema() {
            self = match spec.policy {
                CompiledInputPolicy::External => {
                    self.with_input(spec.name, spec.shape.to_vec(), spec.dtype)?
                }
                CompiledInputPolicy::HostToken => {
                    self.with_host_token_input(spec.name, spec.shape.to_vec())?
                }
            };
        }
        Ok(self)
    }

    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Returns canonical frozen parameter names in deterministic sorted order.
    pub fn frozen_parameters(&self) -> impl ExactSizeIterator<Item = &str> {
        self.frozen_parameters.iter().map(String::as_str)
    }

    /// Returns canonical exclusion names in deterministic sorted order.
    pub fn weight_decay_exclusions(&self) -> impl ExactSizeIterator<Item = &str> {
        self.weight_decay_exclusions.iter().map(String::as_str)
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }

    /// Existing F32 input whose valid-token count weights each microbatch.
    pub fn token_weighted_gradient_accumulation_mask(&self) -> Option<&str> {
        self.token_weight_mask_input.as_deref()
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.max_gradient_norm
    }

    pub fn clip_report_enabled(&self) -> bool {
        self.clip_report
    }

    /// Whether completed-window loss aggregation is captured and reported.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.window_loss_report
    }

    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        match &self.learning_rate {
            CompiledLearningRatePolicy::External => None,
            CompiledLearningRatePolicy::MultiStep(schedule) => Some(schedule),
        }
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&str, &Shape, DType)> {
        self.inputs
            .iter()
            .map(|(name, (shape, dtype))| (name.as_str(), shape, *dtype))
    }

    /// Returns authenticated host-token declarations in lexical name order.
    pub fn host_token_inputs(&self) -> impl ExactSizeIterator<Item = (&str, &Shape)> {
        self.host_token_inputs
            .iter()
            .map(|(name, shape)| (name.as_str(), shape))
    }
}

/// Detached result of one successfully committed compiled training step.
#[derive(Clone, Debug)]
pub struct CompiledTrainingStepResult {
    loss: TensorData,
    outputs: BTreeMap<String, TensorData>,
    step: u64,
    capture_identity: u64,
    clip_report: Option<CompiledAdamWClipReport>,
    window_loss: Option<CompiledAdamWWindowLossValue>,
}

/// Detached outputs from one read-only evaluation of the live compiled
/// parameter frontier.
#[derive(Clone, Debug)]
pub struct CompiledEvaluationResult {
    loss: TensorData,
    outputs: BTreeMap<String, TensorData>,
    capture_identity: u64,
}

/// Preparation evidence for one strict-native CPU pure program.
///
/// Stable identities and cache counts describe compilation only. Wall time is
/// deliberately observational and does not participate in either identity.
#[derive(Clone, Debug)]
pub struct NativeCpuProgramPreparationReport {
    capture_identity: u64,
    native_identity: u64,
    vectorized: bool,
    native_item_count: usize,
    cache_hit_count: usize,
    cache_miss_count: usize,
    execution_plan: ExecutionPlanSummary,
    wall_time: Duration,
}

impl NativeCpuProgramPreparationReport {
    pub const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub const fn native_identity(&self) -> u64 {
        self.native_identity
    }

    pub const fn is_vectorized(&self) -> bool {
        self.vectorized
    }

    pub const fn native_item_count(&self) -> usize {
        self.native_item_count
    }

    pub const fn cache_hit_count(&self) -> usize {
        self.cache_hit_count
    }

    pub const fn cache_miss_count(&self) -> usize {
        self.cache_miss_count
    }

    /// Strict preparation never admits an interpreter fallback item.
    pub const fn fallback_count(&self) -> usize {
        0
    }

    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.execution_plan
    }

    pub const fn wall_time(&self) -> Duration {
        self.wall_time
    }
}

/// Complete preparation evidence for a native CPU AdamW session.
#[derive(Clone, Debug)]
pub struct NativeCpuCompiledAdamWPreparationReport {
    main: NativeCpuProgramPreparationReport,
    partial_flush: Option<NativeCpuProgramPreparationReport>,
    zero_grad: Option<NativeCpuProgramPreparationReport>,
    evaluation: Option<NativeCpuProgramPreparationReport>,
    recurrent_state_count: usize,
    recurrent_state_bytes: usize,
}

impl NativeCpuCompiledAdamWPreparationReport {
    pub const fn main(&self) -> &NativeCpuProgramPreparationReport {
        &self.main
    }

    pub const fn partial_flush(&self) -> Option<&NativeCpuProgramPreparationReport> {
        self.partial_flush.as_ref()
    }

    /// Preparation evidence for the captured state-only accumulation reset.
    pub const fn zero_grad(&self) -> Option<&NativeCpuProgramPreparationReport> {
        self.zero_grad.as_ref()
    }

    pub const fn evaluation(&self) -> Option<&NativeCpuProgramPreparationReport> {
        self.evaluation.as_ref()
    }

    pub const fn recurrent_state_count(&self) -> usize {
        self.recurrent_state_count
    }

    pub const fn recurrent_state_bytes(&self) -> usize {
        self.recurrent_state_bytes
    }
}

struct PreparedNativeCpuProgram {
    report: NativeCpuProgramPreparationReport,
    replay: PreparedRecurrentNativeReplay,
}

struct PreparedNativeCpuEvaluation {
    report: NativeCpuProgramPreparationReport,
    plan: PlannedNativeItems,
}

impl PreparedNativeCpuEvaluation {
    fn validate(&self, capture_identity: u64, capture: &CapturedSchedule) -> Result<()> {
        let native_identity = native_cpu_identity(
            capture_identity,
            self.plan.vectorized(),
            self.plan.schedule_cache_keys().iter().copied(),
        );
        if self.report.capture_identity != capture_identity
            || self.report.native_identity != native_identity
            || self.report.native_item_count != self.plan.item_count()
            || self.report.cache_hit_count != self.plan.cache_hit_count()
            || self.report.cache_miss_count != self.plan.cache_miss_count()
            || self.report.vectorized != self.plan.vectorized()
            || capture.items.iter().map(|item| item.cache_key).ne(self
                .plan
                .schedule_cache_keys()
                .iter()
                .copied())
        {
            return Err(training(
                "compiled native CPU evaluation preparation identity mismatch",
            ));
        }
        Ok(())
    }
}

/// Logical host traffic completed by one successful strict-native CPU replay.
///
/// External imports count only fallback owned copies into retained workspace
/// storage; supported dense F32/I32 inputs bind caller storage read-only for
/// the invocation instead. Recurrent bytes are borrowed directly from the
/// authoritative active and inactive host banks. None are host/device
/// transfers.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCpuReplayTraffic {
    external_input_import_count: u64,
    external_input_import_bytes: u64,
    borrowed_recurrent_input_bytes: u64,
    borrowed_recurrent_output_bytes: u64,
}

impl NativeCpuReplayTraffic {
    pub(crate) const fn new(
        external_input_import_count: u64,
        external_input_import_bytes: u64,
        borrowed_recurrent_input_bytes: u64,
        borrowed_recurrent_output_bytes: u64,
    ) -> Self {
        Self {
            external_input_import_count,
            external_input_import_bytes,
            borrowed_recurrent_input_bytes,
            borrowed_recurrent_output_bytes,
        }
    }

    pub const fn external_input_import_count(&self) -> u64 {
        self.external_input_import_count
    }

    pub const fn external_input_import_bytes(&self) -> u64 {
        self.external_input_import_bytes
    }

    pub const fn borrowed_recurrent_input_bytes(&self) -> u64 {
        self.borrowed_recurrent_input_bytes
    }

    pub const fn borrowed_recurrent_output_bytes(&self) -> u64 {
        self.borrowed_recurrent_output_bytes
    }
}

/// Truthful per-invocation evidence for strict-native CPU replay.
#[derive(Clone, Debug)]
pub struct NativeCpuRunReport {
    capture_identity: u64,
    native_identity: u64,
    vectorized: bool,
    successful_invocation: u64,
    native_item_count: usize,
    schedule_cache_keys: Vec<u64>,
    traffic: NativeCpuReplayTraffic,
    wall_time: Duration,
}

impl NativeCpuRunReport {
    pub const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub const fn native_identity(&self) -> u64 {
        self.native_identity
    }

    pub const fn is_vectorized(&self) -> bool {
        self.vectorized
    }

    pub const fn successful_invocation(&self) -> u64 {
        self.successful_invocation
    }

    pub const fn first_successful_invocation(&self) -> bool {
        self.successful_invocation == 1
    }

    pub const fn native_item_count(&self) -> usize {
        self.native_item_count
    }

    /// Strict replay never executes an interpreter fallback item.
    pub const fn fallback_count(&self) -> usize {
        0
    }

    pub fn schedule_cache_keys(&self) -> &[u64] {
        &self.schedule_cache_keys
    }

    pub const fn traffic(&self) -> &NativeCpuReplayTraffic {
        &self.traffic
    }

    pub const fn wall_time(&self) -> Duration {
        self.wall_time
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompiledInputPolicy {
    External,
    HostToken,
}

/// One fixed external input declaration supplied by a [`CompiledInputBatch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledInputSpec {
    name: &'static str,
    shape: &'static [usize],
    dtype: DType,
    policy: CompiledInputPolicy,
}

impl CompiledInputSpec {
    /// Declares one ordinary fixed-shape external input.
    pub const fn new(name: &'static str, shape: &'static [usize], dtype: DType) -> Self {
        Self {
            name,
            shape,
            dtype,
            policy: CompiledInputPolicy::External,
        }
    }

    /// Declares one fixed rank-two I32 token input eligible for the compiled
    /// training capture's authenticated host-index policy.
    pub const fn host_token(name: &'static str, shape: &'static [usize]) -> Self {
        Self {
            name,
            shape,
            dtype: DType::I32,
            policy: CompiledInputPolicy::HostToken,
        }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn shape(self) -> &'static [usize] {
        self.shape
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }
}

/// Converts one workload-owned batch into the exact named external inputs of a
/// compiled training or evaluation capture.
///
/// The declarative schema is shared by compilation and replay. Implementing
/// this trait once for a domain batch keeps names, shapes, dtypes, and host-token
/// policy at the workload boundary. Recurrent parameter, optimizer, and
/// workload state remain runtime-owned, and
/// [`CompiledTrainingRuntime::step_batch`] supplies the learning rate through
/// its separate typed scalar argument.
pub trait CompiledInputBatch {
    fn schema() -> &'static [CompiledInputSpec];

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>>;
}

impl CompiledEvaluationResult {
    pub fn loss(&self) -> &TensorData {
        &self.loss
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        &self.outputs
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs.get(name)
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }
}

/// Backend-neutral view of one successful read-only compiled evaluation.
pub trait CompiledEvaluation {
    fn loss(&self) -> &TensorData;

    fn outputs(&self) -> &BTreeMap<String, TensorData>;

    fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs().get(name)
    }

    fn capture_identity(&self) -> u64;
}

impl CompiledEvaluation for CompiledEvaluationResult {
    fn loss(&self) -> &TensorData {
        self.loss()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.outputs()
    }

    fn capture_identity(&self) -> u64 {
        self.capture_identity()
    }
}

/// Successful strict-Metal evaluation plus its exact stateless run evidence.
pub struct MetalCompiledEvaluationResult {
    inner: CompiledEvaluationResult,
    report: MetalDeviceRunReport,
}

impl MetalCompiledEvaluationResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl CompiledEvaluation for MetalCompiledEvaluationResult {
    fn loss(&self) -> &TensorData {
        self.loss()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.outputs()
    }

    fn capture_identity(&self) -> u64 {
        self.capture_identity()
    }
}

/// Read-only strict-native CPU evaluation plus its replay evidence.
pub struct NativeCpuCompiledEvaluationResult {
    inner: CompiledEvaluationResult,
    report: NativeCpuRunReport,
}

impl NativeCpuCompiledEvaluationResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &NativeCpuRunReport {
        &self.report
    }
}

impl CompiledEvaluation for NativeCpuCompiledEvaluationResult {
    fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }
}

/// Optional read-only evaluation capability for a compiled training session.
/// Evaluation observes the current parameter frontier without advancing or
/// mutating training, optimizer, accumulation, or workload state.
pub trait CompiledEvaluationRuntime {
    type Evaluation: CompiledEvaluation;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation>;

    /// Evaluates a domain batch after converting only its external bindings.
    fn evaluate_batch<B>(&mut self, batch: B) -> Result<Self::Evaluation>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.evaluate(batch.into_compiled_inputs()?)
    }

    fn evaluation_capture_identity(&self) -> Option<u64>;
}

impl CompiledTrainingStepResult {
    pub fn loss(&self) -> &TensorData {
        &self.loss
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        &self.outputs
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs.get(name)
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }
}

pub type CompiledMomentumSgdStepResult = CompiledTrainingStepResult;

/// Backend- and optimizer-neutral view of one committed compiled training step.
///
/// Concrete optimizer results may expose additional progress, while device
/// results may retain execution reports. Generic training loops can still
/// consume loss, named outputs, replay progress, and capture identity without
/// selecting either concern through an enum.
pub trait CompiledTrainingStep {
    fn loss(&self) -> &TensorData;

    fn outputs(&self) -> &BTreeMap<String, TensorData>;

    fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs().get(name)
    }

    fn step(&self) -> u64;

    fn capture_identity(&self) -> u64;
}

impl CompiledTrainingStep for CompiledTrainingStepResult {
    fn loss(&self) -> &TensorData {
        CompiledTrainingStepResult::loss(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        CompiledTrainingStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        CompiledTrainingStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        CompiledTrainingStepResult::capture_identity(self)
    }
}

/// One committed replay of a compiled AdamW program.
///
/// `step` counts microbatch replays. `optimizer_step` advances only when the
/// configured accumulation window commits, and `accumulation_index` reports
/// the number of retained microbatches toward the next update. `loss_weight`
/// is one for ordinary scalar-loss programs and the exact valid-token count for
/// compiler-owned token-mean programs.
#[derive(Clone, Debug)]
pub struct CompiledAdamWStepResult {
    inner: CompiledTrainingStepResult,
    optimizer_step: u64,
    accumulation_index: u64,
    loss_weight: u64,
    window_loss_report: Option<CompiledAdamWWindowLossReport>,
}

/// Completed-window evidence for compiled global gradient clipping.
///
/// Both values are exact F32 results from the captured graph. The scale is
/// one when clipping is disabled or the pre-clip norm is at or below the
/// configured limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWClipReport {
    pre_clip_global_norm_bits: u32,
    applied_scale_bits: u32,
}

/// Exact aggregate loss for one committed AdamW accumulation window.
///
/// `mean_loss` is the captured F32 recurrence result. `loss_weight` is the
/// microbatch count for ordinary scalar objectives and the valid-token count
/// for compiler-owned token means. Accumulation-only steps, discarded windows,
/// and empty flushes produce no report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWWindowLossReport {
    mean_loss_bits: u32,
    loss_weight: u64,
    microbatch_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompiledAdamWWindowLossValue {
    mean_loss_bits: u32,
    loss_weight: u64,
}

impl CompiledAdamWWindowLossReport {
    fn new(value: CompiledAdamWWindowLossValue, microbatch_count: u64) -> Self {
        Self {
            mean_loss_bits: value.mean_loss_bits,
            loss_weight: value.loss_weight,
            microbatch_count,
        }
    }

    /// Exact F32 mean produced by the captured recurrence.
    pub fn mean_loss(&self) -> f32 {
        f32::from_bits(self.mean_loss_bits)
    }

    /// Sum of microbatch or validated-token weights in this window.
    pub fn loss_weight(&self) -> u64 {
        self.loss_weight
    }

    /// Number of replays committed by this window.
    pub fn microbatch_count(&self) -> u64 {
        self.microbatch_count
    }

    /// Whether the captured aggregate mean is finite.
    pub fn is_finite(&self) -> bool {
        self.mean_loss().is_finite()
    }
}

impl CompiledAdamWClipReport {
    fn new(pre_clip_global_norm: f32, applied_scale: f32) -> Self {
        Self {
            pre_clip_global_norm_bits: pre_clip_global_norm.to_bits(),
            applied_scale_bits: applied_scale.to_bits(),
        }
    }

    pub fn pre_clip_global_norm(&self) -> f32 {
        f32::from_bits(self.pre_clip_global_norm_bits)
    }

    pub fn applied_scale(&self) -> f32 {
        f32::from_bits(self.applied_scale_bits)
    }

    /// Whether both captured values are finite.
    pub fn is_finite(&self) -> bool {
        self.pre_clip_global_norm().is_finite() && self.applied_scale().is_finite()
    }

    /// Whether clipping reduced this completed window's gradient.
    ///
    /// A non-finite report has no meaningful clipping outcome and returns
    /// `None` while preserving its exact captured F32 bits for diagnosis.
    pub fn did_clip(&self) -> Option<bool> {
        self.is_finite().then(|| self.applied_scale() < 1.0)
    }
}

impl CompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    pub fn accumulation_index(&self) -> u64 {
        self.accumulation_index
    }

    /// Exact weight of this step's normalized loss in an aggregate mean.
    ///
    /// Ordinary scalar-loss programs use one. Compiler-owned token-mean
    /// programs use the validated number of non-padding tokens in this replay.
    pub fn loss_weight(&self) -> u64 {
        self.loss_weight
    }

    /// Completed-window clipping evidence when reporting was requested.
    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report.as_ref()
    }

    /// Aggregate loss for the full window committed by this replay.
    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.window_loss_report.as_ref()
    }

    pub fn did_update(&self) -> bool {
        self.accumulation_index == 0
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }
}

/// AdamW-specific progress for one successfully committed training step.
///
/// Concrete runtimes may retain additional execution evidence. For example,
/// [`MetalCompiledAdamWStepResult`] also exposes its exact device run report.
pub trait CompiledAdamWStep: CompiledTrainingStep {
    fn optimizer_step(&self) -> u64;

    fn accumulation_index(&self) -> u64;

    /// Exact weight of this step's loss in the optimizer's aggregate mean.
    ///
    /// The default preserves ordinary scalar-loss and existing external
    /// implementations. Token-mean CPU results override it with the validated
    /// number of non-padding tokens in the replay.
    fn loss_weight(&self) -> u64 {
        1
    }

    /// Completed-window clipping evidence. Existing implementations and
    /// programs compiled without the opt-in report return `None`.
    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        None
    }

    /// Aggregate loss for a full window committed by this replay, when enabled.
    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        None
    }

    fn did_update(&self) -> bool {
        self.accumulation_index() == 0
    }
}

impl CompiledTrainingStep for CompiledAdamWStepResult {
    fn loss(&self) -> &TensorData {
        CompiledAdamWStepResult::loss(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        CompiledAdamWStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        CompiledAdamWStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        CompiledAdamWStepResult::capture_identity(self)
    }
}

impl CompiledAdamWStep for CompiledAdamWStepResult {
    fn optimizer_step(&self) -> u64 {
        CompiledAdamWStepResult::optimizer_step(self)
    }

    fn accumulation_index(&self) -> u64 {
        CompiledAdamWStepResult::accumulation_index(self)
    }

    fn loss_weight(&self) -> u64 {
        CompiledAdamWStepResult::loss_weight(self)
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        CompiledAdamWStepResult::clip_report(self)
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        CompiledAdamWStepResult::window_loss_report(self)
    }
}

/// One committed strict-native CPU AdamW step and its replay evidence.
pub struct NativeCpuCompiledAdamWStepResult {
    inner: CompiledAdamWStepResult,
    report: NativeCpuRunReport,
}

impl NativeCpuCompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    pub fn accumulation_index(&self) -> u64 {
        self.inner.accumulation_index()
    }

    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    /// Aggregate loss for the full window committed by this replay, when enabled.
    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &NativeCpuRunReport {
        &self.report
    }
}

impl CompiledTrainingStep for NativeCpuCompiledAdamWStepResult {
    fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    fn step(&self) -> u64 {
        self.inner.step()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }
}

impl CompiledAdamWStep for NativeCpuCompiledAdamWStepResult {
    fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    fn accumulation_index(&self) -> u64 {
        self.inner.accumulation_index()
    }

    fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    /// Aggregate loss for the full window committed by this replay, when enabled.
    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }
}

/// Outcome of explicitly discarding a compiled AdamW partial gradient window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWZeroGradResult {
    discarded_microbatches: u64,
}

impl CompiledAdamWZeroGradResult {
    /// Number of previously retained microbatches removed by this call.
    pub fn discarded_microbatches(&self) -> u64 {
        self.discarded_microbatches
    }

    /// Whether this call published a new recurrent frontier.
    pub fn did_discard(&self) -> bool {
        self.discarded_microbatches != 0
    }
}

/// Outcome of explicitly committing a non-full compiled AdamW window on CPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWFlushResult {
    flushed_microbatches: u64,
    optimizer_step: u64,
    clip_report: Option<CompiledAdamWClipReport>,
    window_loss_report: Option<CompiledAdamWWindowLossReport>,
}

impl CompiledAdamWFlushResult {
    /// Number of retained microbatches averaged by this call.
    pub fn flushed_microbatches(&self) -> u64 {
        self.flushed_microbatches
    }

    /// Whether this call published an AdamW update.
    pub fn did_update(&self) -> bool {
        self.flushed_microbatches != 0
    }

    /// Optimizer step after the call. Empty windows preserve the prior step.
    pub fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    /// Clipping evidence for a committed nonempty partial window.
    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.clip_report.as_ref()
    }

    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.window_loss_report.as_ref()
    }
}

/// Optimizer progress produced by one optional partial-window flush.
pub trait CompiledAdamWFlush {
    fn flushed_microbatches(&self) -> u64;

    fn did_update(&self) -> bool;

    fn optimizer_step(&self) -> u64;

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        None
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        None
    }
}

impl CompiledAdamWFlush for CompiledAdamWFlushResult {
    fn flushed_microbatches(&self) -> u64 {
        CompiledAdamWFlushResult::flushed_microbatches(self)
    }

    fn did_update(&self) -> bool {
        CompiledAdamWFlushResult::did_update(self)
    }

    fn optimizer_step(&self) -> u64 {
        CompiledAdamWFlushResult::optimizer_step(self)
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        CompiledAdamWFlushResult::clip_report(self)
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        CompiledAdamWFlushResult::window_loss_report(self)
    }
}

/// One strict-native CPU partial-window flush. Empty windows execute no
/// native program and therefore carry no run report.
pub struct NativeCpuCompiledAdamWFlushResult {
    inner: CompiledAdamWFlushResult,
    report: Option<NativeCpuRunReport>,
}

impl NativeCpuCompiledAdamWFlushResult {
    pub fn flushed_microbatches(&self) -> u64 {
        self.inner.flushed_microbatches()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }

    pub fn report(&self) -> Option<&NativeCpuRunReport> {
        self.report.as_ref()
    }
}

impl CompiledAdamWFlush for NativeCpuCompiledAdamWFlushResult {
    fn flushed_microbatches(&self) -> u64 {
        self.inner.flushed_microbatches()
    }

    fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AdamWProgress {
    replay_step: u64,
    optimizer_step: u64,
    accumulation_index: u64,
    discarded_microbatches: u64,
    flushed_window_count: u64,
    flushed_microbatch_count: u64,
    reset_transition_count: u64,
}

impl AdamWProgress {
    const INITIAL: Self = Self {
        replay_step: 0,
        optimizer_step: 0,
        accumulation_index: 0,
        discarded_microbatches: 0,
        flushed_window_count: 0,
        flushed_microbatch_count: 0,
        reset_transition_count: 0,
    };

    fn advance_replay(self, accumulation_steps: u64) -> Result<Self> {
        let replay_step = self
            .replay_step
            .checked_add(1)
            .ok_or_else(|| training("compiled training step overflow"))?;
        let next_index = self
            .accumulation_index
            .checked_add(1)
            .ok_or_else(|| training("compiled AdamW accumulation index overflow"))?;
        let (optimizer_step, accumulation_index) = if next_index == accumulation_steps {
            (
                self.optimizer_step
                    .checked_add(1)
                    .ok_or_else(|| training("compiled AdamW optimizer step overflow"))?,
                0,
            )
        } else {
            (self.optimizer_step, next_index)
        };
        let next = Self {
            replay_step,
            optimizer_step,
            accumulation_index,
            ..self
        };
        validate_adamw_progress(next, accumulation_steps)?;
        Ok(next)
    }

    fn cancel(self, accumulation_steps: u64) -> Result<(Self, CompiledAdamWZeroGradResult)> {
        if self.accumulation_index == 0 {
            return Ok((
                self,
                CompiledAdamWZeroGradResult {
                    discarded_microbatches: 0,
                },
            ));
        }
        let discarded = self.accumulation_index;
        let next = Self {
            accumulation_index: 0,
            discarded_microbatches: self
                .discarded_microbatches
                .checked_add(discarded)
                .ok_or_else(|| training("compiled AdamW discarded microbatch count overflow"))?,
            ..self
        };
        validate_adamw_progress(next, accumulation_steps)?;
        Ok((
            next,
            CompiledAdamWZeroGradResult {
                discarded_microbatches: discarded,
            },
        ))
    }

    fn record_reset_transition(mut self) -> Result<Self> {
        self.reset_transition_count = self
            .reset_transition_count
            .checked_add(1)
            .ok_or_else(|| training("compiled AdamW reset transition count overflow"))?;
        Ok(self)
    }

    fn flush_partial(self, accumulation_steps: u64) -> Result<(Self, CompiledAdamWFlushResult)> {
        if self.accumulation_index == 0 {
            return Ok((
                self,
                CompiledAdamWFlushResult {
                    flushed_microbatches: 0,
                    optimizer_step: self.optimizer_step,
                    clip_report: None,
                    window_loss_report: None,
                },
            ));
        }
        let flushed_microbatches = self.accumulation_index;
        let next = Self {
            optimizer_step: self
                .optimizer_step
                .checked_add(1)
                .ok_or_else(|| training("compiled AdamW optimizer step overflow"))?,
            accumulation_index: 0,
            flushed_window_count: self
                .flushed_window_count
                .checked_add(1)
                .ok_or_else(|| training("compiled AdamW partial flush count overflow"))?,
            flushed_microbatch_count: self
                .flushed_microbatch_count
                .checked_add(flushed_microbatches)
                .ok_or_else(|| training("compiled AdamW flushed microbatch count overflow"))?,
            ..self
        };
        validate_adamw_progress(next, accumulation_steps)?;
        Ok((
            next,
            CompiledAdamWFlushResult {
                flushed_microbatches,
                optimizer_step: next.optimizer_step,
                clip_report: None,
                window_loss_report: None,
            },
        ))
    }
}

trait CompiledOptimizerProgram {
    fn name(&self) -> &'static str;
    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)>;
    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>>;
    fn gradients(
        &self,
        graph: &mut Graph,
        loss: NodeId,
        targets: &[NodeId],
    ) -> Result<Vec<NodeId>> {
        graph.gradient_default(loss, targets)
    }
    fn lower_updates(
        &self,
        graph: &mut Graph,
        context: CompiledOptimizerLoweringContext<'_>,
    ) -> Result<CompiledOptimizerLowering>;
}

struct CompiledOptimizerLoweringContext<'a> {
    loss: NodeId,
    learning_rate: NodeId,
    inputs: &'a BTreeMap<String, NodeId>,
    parameters: &'a BTreeMap<String, NodeId>,
    gradients: &'a BTreeMap<String, NodeId>,
    states: &'a BTreeMap<RecurrentStateKey, NodeId>,
}

#[derive(Clone, Copy)]
struct CompiledAdamWClipNodes {
    pre_clip_global_norm: NodeId,
    applied_scale: NodeId,
}

#[derive(Clone, Copy)]
struct CompiledAdamWWindowLossNodes {
    mean_loss: NodeId,
    loss_weight: NodeId,
}

struct CompiledOptimizerLowering {
    updates: BTreeMap<RecurrentStateKey, NodeId>,
    clip_report: Option<CompiledAdamWClipNodes>,
    window_loss_report: Option<CompiledAdamWWindowLossNodes>,
}

struct ClippedGradients {
    gradients: BTreeMap<String, NodeId>,
    report: Option<CompiledAdamWClipNodes>,
}

struct MomentumProgram {
    config: CompiledMomentumSgdConfig,
}

struct AdamWProgram {
    config: CompiledAdamWConfig,
}

impl CompiledOptimizerProgram for MomentumProgram {
    fn name(&self) -> &'static str {
        "momentum-SGD"
    }

    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)> {
        &self.config.inputs
    }

    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>> {
        let mut specs = Vec::with_capacity(parameters.len() * 2);
        for (ordinal, (name, value)) in parameters.iter().enumerate() {
            specs.push(StateSpec::parameter(ordinal, name, value.clone()));
            specs.push(StateSpec::momentum(ordinal, name, value)?);
        }
        Ok(specs)
    }

    fn lower_updates(
        &self,
        graph: &mut Graph,
        context: CompiledOptimizerLoweringContext<'_>,
    ) -> Result<CompiledOptimizerLowering> {
        let CompiledOptimizerLoweringContext {
            learning_rate,
            parameters,
            gradients,
            states,
            ..
        } = context;
        let momentum = scalar_f32(graph, self.config.momentum)?;
        let mut updates = BTreeMap::new();
        for (name, parameter) in parameters {
            let momentum_key = RecurrentStateKey::momentum(name);
            let slot = states[&momentum_key];
            let retained = graph.mul(momentum, slot)?;
            let next_momentum = graph.add(retained, gradients[name])?;
            let scaled = graph.mul(learning_rate, next_momentum)?;
            let next_parameter = graph.sub(*parameter, scaled)?;
            validate_parameter_update(graph, *parameter, next_momentum)?;
            validate_parameter_update(graph, *parameter, next_parameter)?;
            updates.insert(momentum_key, next_momentum);
            updates.insert(RecurrentStateKey::parameter(name), next_parameter);
        }
        Ok(CompiledOptimizerLowering {
            updates,
            clip_report: None,
            window_loss_report: None,
        })
    }
}

impl CompiledOptimizerProgram for AdamWProgram {
    fn name(&self) -> &'static str {
        "AdamW"
    }

    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)> {
        &self.config.inputs
    }

    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>> {
        let accumulating = self.config.gradient_accumulation_steps > 1;
        let per_parameter = if accumulating { 4 } else { 3 };
        let mut specs = Vec::with_capacity(
            parameters.len() * per_parameter
                + 1
                + accumulating as usize
                + self.config.token_weight_mask_input.is_some() as usize
                + self.config.window_loss_report as usize,
        );
        for (ordinal, (name, value)) in parameters.iter().enumerate() {
            specs.push(StateSpec::parameter(ordinal, name, value.clone()));
            for state in [
                AdamWParameterState::FirstMoment,
                AdamWParameterState::SecondMoment,
            ] {
                specs.push(StateSpec::adamw_parameter(ordinal, name, value, state)?);
            }
            if accumulating {
                specs.push(StateSpec::adamw_parameter(
                    ordinal,
                    name,
                    value,
                    AdamWParameterState::GradientAccumulator,
                )?);
            }
        }
        specs.push(StateSpec::adamw_global(AdamWGlobalState::Step)?);
        if accumulating {
            specs.push(StateSpec::adamw_global(
                AdamWGlobalState::AccumulationIndex,
            )?);
        }
        if self.config.token_weight_mask_input.is_some() {
            specs.push(StateSpec::adamw_global(
                AdamWGlobalState::AccumulatedTokenCount,
            )?);
        }
        if self.config.window_loss_report {
            specs.push(StateSpec::adamw_global(
                AdamWGlobalState::AccumulatedLossNumerator,
            )?);
        }
        Ok(specs)
    }

    fn gradients(
        &self,
        graph: &mut Graph,
        loss: NodeId,
        targets: &[NodeId],
    ) -> Result<Vec<NodeId>> {
        if self.config.loss_scale == 1.0 {
            return graph.gradient_default(loss, targets);
        }
        let scale = scalar_f32(graph, self.config.loss_scale)?;
        let scaled_loss = graph.mul(loss, scale)?;
        graph
            .gradient_default(scaled_loss, targets)?
            .into_iter()
            .map(|gradient| graph.div(gradient, scale))
            .collect()
    }

    fn lower_updates(
        &self,
        graph: &mut Graph,
        context: CompiledOptimizerLoweringContext<'_>,
    ) -> Result<CompiledOptimizerLowering> {
        let CompiledOptimizerLoweringContext {
            loss,
            learning_rate,
            inputs,
            parameters,
            gradients,
            states,
        } = context;
        let learning_rate = lower_adamw_learning_rate(&self.config, graph, learning_rate, states)?;
        if self.config.gradient_accumulation_steps == 1 {
            let clipped = clip_gradients_by_global_norm(&self.config, graph, gradients)?;
            let mut updates = lower_adamw_update_candidates(
                &self.config,
                graph,
                learning_rate,
                parameters,
                &clipped.gradients,
                states,
            )?;
            let window_loss_report = if self.config.window_loss_report {
                let numerator_key =
                    RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator);
                let numerator = graph.add(states[&numerator_key], loss)?;
                let zero = scalar_f32(graph, 0.0)?;
                updates.insert(numerator_key, zero);
                Some(CompiledAdamWWindowLossNodes {
                    mean_loss: numerator,
                    loss_weight: graph.full_with_dtype(
                        Shape::from([]),
                        Scalar::U(1),
                        DType::U64,
                    )?,
                })
            } else {
                None
            };
            return Ok(CompiledOptimizerLowering {
                updates,
                clip_report: clipped.report,
                window_loss_report,
            });
        }

        let one_u64 = graph.full_with_dtype(Shape::from([]), Scalar::U(1), DType::U64)?;
        let zero_u64 = graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?;
        let threshold = graph.full_with_dtype(
            Shape::from([]),
            Scalar::U(self.config.gradient_accumulation_steps),
            DType::U64,
        )?;
        let accumulation_index_key =
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex);
        let step_key = RecurrentStateKey::adamw_global(AdamWGlobalState::Step);
        let next_index = graph.add(states[&accumulation_index_key], one_u64)?;
        let commit = graph.compare(CompareOp::Eq, next_index, threshold)?;
        let reset_index = graph.select(commit, zero_u64, next_index)?;
        let weighted_count =
            self.config
                .token_weight_mask_input
                .as_ref()
                .map(|mask_input| {
                    let mask = inputs.get(mask_input).copied().ok_or_else(|| {
                        training("compiled AdamW token-weight mask input is absent")
                    })?;
                    let batch_count = graph.sum_all(mask)?;
                    let batch_count_u64 = graph.cast(batch_count, DType::U64)?;
                    let count_key =
                        RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedTokenCount);
                    let total_count = graph.add(states[&count_key], batch_count_u64)?;
                    let divisor = graph.cast(total_count, DType::F32)?;
                    Ok::<_, Error>((batch_count, count_key, total_count, divisor))
                })
                .transpose()?;
        let divisor = match &weighted_count {
            Some((_, _, _, divisor)) => *divisor,
            None => scalar_f32(graph, self.config.gradient_accumulation_steps as f32)?,
        };
        let window_loss = if self.config.window_loss_report {
            let numerator_key =
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator);
            let contribution = match &weighted_count {
                Some((batch_count, ..)) => graph.mul(loss, *batch_count)?,
                None => loss,
            };
            let numerator = graph.add(states[&numerator_key], contribution)?;
            let mean_loss = graph.div(numerator, divisor)?;
            let loss_weight = match &weighted_count {
                Some((_, _, total_count, _)) => *total_count,
                None => next_index,
            };
            Some((numerator_key, numerator, mean_loss, loss_weight))
        } else {
            None
        };

        let mut averaged_gradients = BTreeMap::new();
        let mut accumulated_gradients = BTreeMap::new();
        for (name, gradient) in gradients {
            let gradient = match &weighted_count {
                Some((batch_count, ..)) => graph.mul(*gradient, *batch_count)?,
                None => *gradient,
            };
            let key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulated = graph.add(states[&key], gradient)?;
            let averaged = graph.div(accumulated, divisor)?;
            accumulated_gradients.insert(name.clone(), accumulated);
            averaged_gradients.insert(name.clone(), averaged);
        }

        let clipped = clip_gradients_by_global_norm(&self.config, graph, &averaged_gradients)?;

        let candidates = lower_adamw_update_candidates(
            &self.config,
            graph,
            learning_rate,
            parameters,
            &clipped.gradients,
            states,
        )?;
        let mut updates = BTreeMap::new();
        updates.insert(accumulation_index_key, reset_index);
        if let Some((_, count_key, total_count, _)) = weighted_count {
            updates.insert(count_key, graph.select(commit, zero_u64, total_count)?);
        }
        if let Some((key, numerator, ..)) = &window_loss {
            let zero = scalar_f32(graph, 0.0)?;
            updates.insert(key.clone(), graph.select(commit, zero, *numerator)?);
        }
        updates.insert(
            step_key.clone(),
            graph.select(commit, candidates[&step_key], states[&step_key])?,
        );
        for (name, parameter) in parameters {
            let first_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::FirstMoment);
            let second_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::SecondMoment);
            let accumulator_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulator_shape = graph.shape(accumulated_gradients[name])?.clone();
            let zero = graph.lazy_full_with_dtype(accumulator_shape, Scalar::I(0), DType::F32)?;
            let next_accumulator = graph.select(commit, zero, accumulated_gradients[name])?;
            let next_first = graph.select(commit, candidates[&first_key], states[&first_key])?;
            let next_second = graph.select(commit, candidates[&second_key], states[&second_key])?;
            let next_parameter = graph.select(
                commit,
                candidates[&RecurrentStateKey::parameter(name)],
                *parameter,
            )?;
            validate_parameter_update(graph, *parameter, next_accumulator)?;
            validate_parameter_update(graph, *parameter, next_first)?;
            validate_parameter_update(graph, *parameter, next_second)?;
            validate_parameter_update(graph, *parameter, next_parameter)?;
            updates.insert(accumulator_key, next_accumulator);
            updates.insert(first_key, next_first);
            updates.insert(second_key, next_second);
            updates.insert(RecurrentStateKey::parameter(name), next_parameter);
        }
        Ok(CompiledOptimizerLowering {
            updates,
            clip_report: clipped.report,
            window_loss_report: window_loss.map(|(_, _, mean_loss, loss_weight)| {
                CompiledAdamWWindowLossNodes {
                    mean_loss,
                    loss_weight,
                }
            }),
        })
    }
}

fn lower_adamw_learning_rate(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    external_learning_rate: NodeId,
    states: &BTreeMap<RecurrentStateKey, NodeId>,
) -> Result<NodeId> {
    let CompiledLearningRatePolicy::MultiStep(schedule) = &config.learning_rate else {
        return Ok(external_learning_rate);
    };
    let step_key = RecurrentStateKey::adamw_global(AdamWGlobalState::Step);
    let completed_step = states
        .get(&step_key)
        .copied()
        .ok_or_else(|| training("compiled AdamW optimizer step state is absent"))?;
    let mut learning_rate = scalar_f32(graph, schedule.base)?;
    if schedule.milestones.is_empty() {
        return Ok(learning_rate);
    }
    let gamma = scalar_f32(graph, schedule.gamma)?;
    for milestone in &schedule.milestones {
        let milestone =
            graph.full_with_dtype(Shape::from([]), Scalar::U(*milestone), DType::U64)?;
        let reached = graph.compare(CompareOp::Ge, completed_step, milestone)?;
        let decayed = graph.mul(learning_rate, gamma)?;
        learning_rate = graph.select(reached, decayed, learning_rate)?;
    }
    Ok(learning_rate)
}

fn clip_gradients_by_global_norm(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    gradients: &BTreeMap<String, NodeId>,
) -> Result<ClippedGradients> {
    if config.max_gradient_norm.is_none() && !config.clip_report {
        return Ok(ClippedGradients {
            gradients: gradients.clone(),
            report: None,
        });
    }

    let gradients = materialize_gradient_vector(graph, gradients)?;

    let mut squared_norms = Vec::with_capacity(gradients.len());
    for gradient in gradients.values() {
        let squared = graph.mul(*gradient, *gradient)?;
        squared_norms.push(graph.sum_all(squared)?);
    }
    if squared_norms.len() == 1 {
        // Graph reductions intentionally elide extent-one axes. A trailing
        // positive zero keeps the single-parameter case on the same explicit
        // F32 reduction boundary without changing this nonnegative sum.
        squared_norms.push(scalar_f32(graph, 0.0)?);
    }
    // Stack the stable per-parameter F32 subtotals and reduce them through one
    // typed kernel. Its output storage is the exact commitment before `sqrt`:
    // native replay cannot widen a left-deep scalar addition chain, and no
    // scalar Contiguous fusion rehearsal can absorb the gradient graph.
    let squared_norms = graph.stack_default(squared_norms)?;
    let total = graph.sum_all(squared_norms)?;
    let norm = graph.sqrt(total)?;
    let scale = match config.max_gradient_norm {
        Some(max_norm) => {
            let max_norm = scalar_f32(graph, max_norm)?;
            // max(norm, limit) makes the scale exactly one below the limit,
            // while a NaN norm remains the ordered lhs and therefore
            // propagates instead of being silently treated as finite.
            let denominator = graph.maximum(norm, max_norm)?;
            graph.div(max_norm, denominator)?
        }
        None => scalar_f32(graph, 1.0)?,
    };
    let gradients = if config.max_gradient_norm.is_some() {
        gradients
            .iter()
            .map(|(name, gradient)| Ok((name.clone(), graph.mul(*gradient, scale)?)))
            .collect::<Result<_>>()?
    } else {
        gradients.clone()
    };
    Ok(ClippedGradients {
        gradients,
        report: config.clip_report.then_some(CompiledAdamWClipNodes {
            pre_clip_global_norm: norm,
            applied_scale: scale,
        }),
    })
}

fn materialize_gradient_vector(
    graph: &mut Graph,
    gradients: &BTreeMap<String, NodeId>,
) -> Result<BTreeMap<String, NodeId>> {
    if gradients.is_empty() {
        return Err(training("compiled AdamW has no gradients to materialize"));
    }
    let mut flattened = Vec::with_capacity(gradients.len().max(2));
    let mut ranges = Vec::with_capacity(gradients.len());
    let mut offset = 0usize;
    for (name, gradient) in gradients {
        let shape = graph.shape(*gradient)?.clone();
        if graph.dtype(*gradient)? != DType::F32 {
            return Err(training("compiled AdamW gradient must be F32"));
        }
        let elements = shape.numel()?;
        let end = offset
            .checked_add(elements)
            .ok_or_else(|| training("compiled AdamW gradient vector overflow"))?;
        flattened.push(graph.reshape(*gradient, [elements])?);
        ranges.push((name.clone(), shape, offset, end));
        offset = end;
    }
    if flattened.len() == 1 {
        // Concat requires two inputs. One empty F32 leaf forces the same real
        // materialization owner for a single parameter without adding a lane.
        flattened.push(graph.constant(TensorData::new([0], Vec::<f32>::new())?));
    }
    let vector = graph.concat(flattened, 0)?;
    ranges
        .into_iter()
        .map(|(name, shape, start, end)| {
            let slice = graph.shrink(vector, vec![(start, end)])?;
            let gradient = graph.reshape(slice, shape)?;
            Ok((name, gradient))
        })
        .collect()
}

fn lower_adamw_update_candidates(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    learning_rate: NodeId,
    parameters: &BTreeMap<String, NodeId>,
    gradients: &BTreeMap<String, NodeId>,
    states: &BTreeMap<RecurrentStateKey, NodeId>,
) -> Result<BTreeMap<RecurrentStateKey, NodeId>> {
    let one_u64 = graph.full_with_dtype(Shape::from([]), Scalar::U(1), DType::U64)?;
    let step_key = RecurrentStateKey::adamw_global(AdamWGlobalState::Step);
    let next_step = graph.add(states[&step_key], one_u64)?;
    let step_f32 = graph.cast(next_step, DType::F32)?;
    let one = scalar_f32(graph, 1.0)?;
    let beta1 = scalar_f32(graph, config.beta1)?;
    let beta2 = scalar_f32(graph, config.beta2)?;
    let one_minus_beta1 = scalar_f32(graph, 1.0 - config.beta1)?;
    let one_minus_beta2 = scalar_f32(graph, 1.0 - config.beta2)?;
    let eps = scalar_f32(graph, config.eps)?;
    let weight_decay = scalar_f32(graph, config.weight_decay)?;
    let beta1_power = graph.pow(beta1, step_f32)?;
    let beta2_power = graph.pow(beta2, step_f32)?;
    let first_correction = graph.sub(one, beta1_power)?;
    let second_correction = graph.sub(one, beta2_power)?;
    let decay = graph.mul(learning_rate, weight_decay)?;
    let decay_factor = graph.sub(one, decay)?;

    let mut updates = BTreeMap::from([(step_key, next_step)]);
    for (name, parameter) in parameters {
        let gradient = gradients[name];
        let first_key = RecurrentStateKey::adamw_parameter(name, AdamWParameterState::FirstMoment);
        let second_key =
            RecurrentStateKey::adamw_parameter(name, AdamWParameterState::SecondMoment);
        let retained_first = graph.mul(beta1, states[&first_key])?;
        let fresh_first = graph.mul(one_minus_beta1, gradient)?;
        let next_first = graph.add(retained_first, fresh_first)?;
        let retained_second = graph.mul(beta2, states[&second_key])?;
        let gradient_squared = graph.mul(gradient, gradient)?;
        let fresh_second = graph.mul(one_minus_beta2, gradient_squared)?;
        let next_second = graph.add(retained_second, fresh_second)?;
        let corrected_first = graph.div(next_first, first_correction)?;
        let corrected_second = graph.div(next_second, second_correction)?;
        let root = graph.sqrt(corrected_second)?;
        let denominator = graph.add(root, eps)?;
        let normalized = graph.div(corrected_first, denominator)?;
        let decayed = if config.weight_decay_exclusions.contains(name) {
            *parameter
        } else {
            graph.mul(*parameter, decay_factor)?
        };
        let scaled = graph.mul(learning_rate, normalized)?;
        let next_parameter = graph.sub(decayed, scaled)?;
        validate_parameter_update(graph, *parameter, next_first)?;
        validate_parameter_update(graph, *parameter, next_second)?;
        validate_parameter_update(graph, *parameter, next_parameter)?;
        updates.insert(first_key, next_first);
        updates.insert(second_key, next_second);
        updates.insert(RecurrentStateKey::parameter(name), next_parameter);
    }
    Ok(updates)
}

fn scalar_f32(graph: &mut Graph, value: f32) -> Result<NodeId> {
    graph.full_with_dtype(Shape::from([]), Scalar::F(value as f64), DType::F32)
}

fn validate_parameter_update(graph: &Graph, parameter: NodeId, update: NodeId) -> Result<()> {
    if graph.shape(update)? != graph.shape(parameter)? || graph.dtype(update)? != DType::F32 {
        return Err(training("compiled optimizer update descriptor mismatch"));
    }
    Ok(())
}

/// One compiled momentum-SGD training program.
pub struct CpuCompiledMomentumSgd {
    inner: CpuCompiledTrainingProgram,
}

/// Resource-free compiled AdamW program ready for a concrete runtime.
///
/// Compilation owns graph construction, differentiation, scheduling, capture,
/// recurrent-state admission, and optional checkpoint restoration. Preparing
/// the plan then chooses CPU replay or strict Metal rendering without changing
/// the authenticated program or optimizer frontier.
#[derive(Clone)]
pub struct CompiledAdamWPlan {
    inner: CompiledTrainingPlan,
    partial_flush: Option<CompiledAdamWAuxiliaryPlan>,
    zero_grad: Option<CompiledAdamWAuxiliaryPlan>,
    program_identity: u64,
    gradient_accumulation_steps: u64,
    token_weight_mask_input: Option<String>,
    max_gradient_norm: Option<f32>,
    clip_report: bool,
    window_loss_report: bool,
    loss_scale: f32,
    progress: AdamWProgress,
    dropout: Option<CompiledDropoutState>,
    host_token_inputs: BTreeMap<String, Shape>,
    frozen_parameters: BTreeSet<String>,
    evaluation: Option<CompiledEvaluationPlan>,
    learning_rate: CompiledLearningRatePolicy,
}

/// Explicit differentiation objective returned by a module training builder.
///
/// [`Scalar`](Self::Scalar) is the already-normalized scalar loss used by the
/// ordinary compiled AdamW policy. [`TokenMean`](Self::TokenMean) is a
/// fixed-shape F32 tensor of per-token losses; compilation combines it with
/// the token mask configured on [`CompiledAdamWConfig`] and owns the resulting
/// masked mean as both the public loss and differentiation root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledAdamWObjective {
    Scalar(NodeId),
    TokenMean(NodeId),
}

impl CompiledAdamWObjective {
    pub const fn scalar(loss: NodeId) -> Self {
        Self::Scalar(loss)
    }

    pub const fn token_mean(losses: NodeId) -> Self {
        Self::TokenMean(losses)
    }

    pub const fn node(self) -> NodeId {
        match self {
            Self::Scalar(node) | Self::TokenMean(node) => node,
        }
    }
}

/// Compact result of building one compiled AdamW module graph.
///
/// The objective makes scalar-loss versus compiler-owned token-mean policy
/// explicit at the builder boundary. Named outputs retain their existing
/// replay behavior and capture identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAdamWGraph {
    objective: CompiledAdamWObjective,
    outputs: BTreeMap<String, NodeId>,
}

impl CompiledAdamWGraph {
    pub fn new(objective: CompiledAdamWObjective, outputs: BTreeMap<String, NodeId>) -> Self {
        Self { objective, outputs }
    }

    pub fn scalar(loss: NodeId, outputs: BTreeMap<String, NodeId>) -> Self {
        Self::new(CompiledAdamWObjective::Scalar(loss), outputs)
    }

    pub fn token_mean(losses: NodeId, outputs: BTreeMap<String, NodeId>) -> Self {
        Self::new(CompiledAdamWObjective::TokenMean(losses), outputs)
    }

    pub const fn objective(&self) -> CompiledAdamWObjective {
        self.objective
    }

    pub fn outputs(&self) -> &BTreeMap<String, NodeId> {
        &self.outputs
    }

    pub fn into_parts(self) -> (CompiledAdamWObjective, BTreeMap<String, NodeId>) {
        (self.objective, self.outputs)
    }
}

/// Resource-free AdamW plan paired with the exact module value used to build it.
///
/// The module is not exposed while the plan or its prepared session exists.
/// This prevents ordinary callers from accidentally treating its stale host
/// parameters as the active training frontier. Successful
/// [`CompiledModuleAdamWSession::finish`] publishes before returning it; the
/// explicit abort path returns the sealed host state without publication.
pub struct CompiledModuleAdamWPlan<M> {
    module: M,
    plan: CompiledAdamWPlan,
    seal: CompiledModuleSeal,
}

/// Prepared compiled AdamW session that owns its source module for the complete
/// replay lifecycle.
pub struct CompiledModuleAdamWSession<M, R> {
    module: M,
    runtime: R,
    seal: CompiledModuleSeal,
}

/// Recoverable compilation failure retaining the exact uncompiled module.
pub struct CompiledModuleAdamWCompileError<M> {
    module: M,
    source: Error,
}

/// Recoverable checkpoint-restore failure retaining the complete owned plan.
pub struct CompiledModuleAdamWRestoreError<M> {
    plan: Box<CompiledModuleAdamWPlan<M>>,
    source: Error,
}

impl<M> CompiledModuleAdamWRestoreError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn plan(&self) -> &CompiledModuleAdamWPlan<M> {
        &self.plan
    }

    pub fn into_plan(self) -> CompiledModuleAdamWPlan<M> {
        *self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWPlan<M>, Error) {
        (*self.plan, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleAdamWRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWRestoreError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleAdamWRestoreError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW checkpoint restore failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleAdamWRestoreError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<M> CompiledModuleAdamWCompileError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_module(self) -> M {
        self.module
    }

    pub fn into_parts(self) -> (M, Error) {
        (self.module, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleAdamWCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWCompileError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleAdamWCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW compilation failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleAdamWCompileError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Recoverable target-preparation failure retaining the unconsumed owned plan.
pub struct CompiledModuleAdamWPrepareError<M, E> {
    plan: CompiledModuleAdamWPlan<M>,
    source: E,
}

/// Recoverable evaluation-capture failure retaining the complete owned plan.
pub struct CompiledModuleAdamWEvaluationError<M> {
    plan: Box<CompiledModuleAdamWPlan<M>>,
    source: Error,
}

impl<M> CompiledModuleAdamWEvaluationError<M> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_plan(self) -> CompiledModuleAdamWPlan<M> {
        *self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWPlan<M>, Error) {
        (*self.plan, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleAdamWEvaluationError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWEvaluationError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleAdamWEvaluationError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW evaluation capture failed: {}",
            self.source
        )
    }
}

impl<M> std::error::Error for CompiledModuleAdamWEvaluationError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<M, E> CompiledModuleAdamWPrepareError<M, E> {
    pub fn source_error(&self) -> &E {
        &self.source
    }

    pub fn into_plan(self) -> CompiledModuleAdamWPlan<M> {
        self.plan
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWPlan<M>, E) {
        (self.plan, self.source)
    }
}

impl<M, E: fmt::Debug> fmt::Debug for CompiledModuleAdamWPrepareError<M, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWPrepareError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M, E: fmt::Display> fmt::Display for CompiledModuleAdamWPrepareError<M, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW preparation failed: {}",
            self.source
        )
    }
}

impl<M, E: std::error::Error + 'static> std::error::Error
    for CompiledModuleAdamWPrepareError<M, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Finalization failure retaining the intact owned module/session pair.
///
/// This covers both parameter-only [`CompiledModuleAdamWSession::finish`] and
/// checkpointed [`CompiledModuleAdamWSession::finish_with_checkpoint`] and
/// [`CompiledModuleAdamWSession::finish_with_module_checkpoint`] finalization.
/// The retained session remains available for inspection, retry, or recovery
/// without publication.
pub struct CompiledModuleAdamWFinishError<M, R> {
    session: Box<CompiledModuleAdamWSession<M, R>>,
    source: Error,
}

impl<M, R> CompiledModuleAdamWFinishError<M, R> {
    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn session(&self) -> &CompiledModuleAdamWSession<M, R> {
        &self.session
    }

    pub fn into_session(self) -> CompiledModuleAdamWSession<M, R> {
        *self.session
    }

    pub fn into_parts(self) -> (CompiledModuleAdamWSession<M, R>, Error) {
        (*self.session, self.source)
    }

    /// Discards the failed runtime frontier and returns the sealed host module
    /// exactly as it currently exists, without attempting publication again.
    pub fn into_module_without_publication(self) -> M {
        (*self.session).into_module_without_publication()
    }
}

impl<M, R> fmt::Debug for CompiledModuleAdamWFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledModuleAdamWFinishError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M, R> fmt::Display for CompiledModuleAdamWFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "owned compiled AdamW finalization failed: {}",
            self.source
        )
    }
}

impl<M, R> std::error::Error for CompiledModuleAdamWFinishError<M, R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// One compiled AdamW training program with recurrent first/second moments, a
/// graph-owned step counter, and capture-authenticated gradient policies.
pub struct CpuCompiledAdamW {
    inner: CpuCompiledTrainingProgram,
    partial_flush: Option<CompiledAdamWAuxiliaryPlan>,
    zero_grad: Option<CompiledAdamWAuxiliaryPlan>,
    gradient_accumulation_steps: u64,
    token_weight_mask_input: Option<String>,
    max_gradient_norm: Option<f32>,
    clip_report: bool,
    window_loss_report: bool,
    loss_scale: f32,
    progress: AdamWProgress,
    dropout: Option<CompiledDropoutState>,
    host_token_inputs: BTreeMap<String, Shape>,
    frozen_parameters: BTreeSet<String>,
    evaluation: Option<CpuCompiledEvaluation>,
    learning_rate: CompiledLearningRatePolicy,
    non_finite_policy: CpuNonFinitePolicy,
}

/// Strict-native CPU AdamW session prepared from the same authenticated plan
/// as [`CpuCompiledAdamW`]. Optimizer, progress, checkpoint, accumulation,
/// dropout, and evaluation ownership remain in the shared CPU core; only pure
/// schedule execution is replaced with strict native JIT replay.
pub struct NativeCpuCompiledAdamW<'a> {
    inner: CpuCompiledAdamW,
    executor: &'a CapturedReplayExecutor,
    main_replay: PreparedRecurrentNativeReplay,
    partial_flush_replay: Option<PreparedRecurrentNativeReplay>,
    zero_grad_replay: Option<PreparedRecurrentNativeReplay>,
    evaluation_replay: Option<PreparedNativeCpuEvaluation>,
    preparation: NativeCpuCompiledAdamWPreparationReport,
    successful_steps: u64,
    successful_flushes: u64,
    successful_zero_grads: u64,
    successful_evaluations: u64,
}

/// Resource-free Metal rendering of one compiled AdamW plan. Preparing it
/// uploads the plan's parameter, moment, and optimizer-step frontier into the
/// existing epoch-swapped Metal runtime.
pub struct MetalCompiledAdamWPlan {
    inner: MetalCompiledTrainingPlan,
    partial_flush: Option<MetalFixedStateTransitionPlan>,
    progress: AdamWProgress,
    flush_capture_identity: Option<u64>,
    gradient_accumulation_steps: u64,
    max_gradient_norm: Option<f32>,
    loss_scale: f32,
    dropout: Option<CompiledDropoutState>,
    frozen_parameters: BTreeSet<String>,
}

/// Optimizer-neutral strict-Metal rendering of one compiled training program.
/// Optimizer facades retain only their policy and progress around this shared
/// recurrent execution core.
struct MetalCompiledTrainingPlan {
    inner: MetalStatefulInferencePlan,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    program_identity: u64,
    evaluation: Option<(MetalFixedStateReadPlan, Vec<String>, u64)>,
}

/// Device-resident AdamW training session backed by one fixed Metal capture.
/// Parameters and optimizer slots remain in the double-buffered device state
/// frontier between calls. Batch inputs and learning rate cross the host
/// boundary on every step; requested outputs cross only when the caller uses
/// the observed [`MetalCompiledAdamW::step`] path.
pub struct MetalCompiledAdamW {
    inner: MetalCompiledTrainingProgram,
    partial_flush: Option<MetalFixedStateTransitionSession>,
    progress: AdamWProgress,
    flush_capture_identity: Option<u64>,
    gradient_accumulation_steps: u64,
    max_gradient_norm: Option<f32>,
    loss_scale: f32,
    dropout: Option<CompiledDropoutState>,
    frozen_parameters: BTreeSet<String>,
}

/// Optimizer-neutral owner of one prepared strict-Metal training program.
struct MetalCompiledTrainingProgram {
    session: MetalDeviceSession,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    program_identity: u64,
    scoreboard: Option<MetalScoreboardObserver>,
    evaluation: Option<(MetalFixedStateReadSession, Vec<String>, u64)>,
}

struct MetalCompiledTrainingRun {
    loss: TensorData,
    outputs: BTreeMap<String, TensorData>,
    report: MetalDeviceRunReport,
}

/// One committed Metal AdamW step plus its exact device execution report.
pub struct MetalCompiledAdamWStepResult {
    inner: CompiledAdamWStepResult,
    report: MetalDeviceRunReport,
}

/// One committed Metal AdamW step whose loss and named outputs remained on the
/// device. The exact replay and optimizer progress plus device report remain
/// available without manufacturing an observed [`CompiledTrainingStep`].
pub struct MetalCompiledAdamWCommitResult {
    progress: AdamWProgress,
    capture_identity: u64,
    report: MetalDeviceRunReport,
}

/// One committed strict-Metal partial-window flush and its exact device report.
pub struct MetalCompiledAdamWFlushResult {
    inner: CompiledAdamWFlushResult,
    report: Option<MetalDeviceRunReport>,
}

impl MetalCompiledAdamWFlushResult {
    pub fn flushed_microbatches(&self) -> u64 {
        self.inner.flushed_microbatches()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    /// Exact device report for a committed update. Empty flushes execute no
    /// device invocation and therefore return `None`.
    pub fn report(&self) -> Option<&MetalDeviceRunReport> {
        self.report.as_ref()
    }
}

impl CompiledAdamWFlush for MetalCompiledAdamWFlushResult {
    fn flushed_microbatches(&self) -> u64 {
        MetalCompiledAdamWFlushResult::flushed_microbatches(self)
    }

    fn did_update(&self) -> bool {
        MetalCompiledAdamWFlushResult::did_update(self)
    }

    fn optimizer_step(&self) -> u64 {
        MetalCompiledAdamWFlushResult::optimizer_step(self)
    }
}

impl MetalCompiledAdamWCommitResult {
    pub fn step(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn optimizer_step(&self) -> u64 {
        self.progress.optimizer_step
    }

    pub fn accumulation_index(&self) -> u64 {
        self.progress.accumulation_index
    }

    pub fn did_update(&self) -> bool {
        self.progress.accumulation_index == 0
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl MetalCompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    pub fn accumulation_index(&self) -> u64 {
        self.inner.accumulation_index()
    }

    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl CompiledTrainingStep for MetalCompiledAdamWStepResult {
    fn loss(&self) -> &TensorData {
        MetalCompiledAdamWStepResult::loss(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        MetalCompiledAdamWStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        MetalCompiledAdamWStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        MetalCompiledAdamWStepResult::capture_identity(self)
    }
}

impl CompiledAdamWStep for MetalCompiledAdamWStepResult {
    fn optimizer_step(&self) -> u64 {
        MetalCompiledAdamWStepResult::optimizer_step(self)
    }

    fn accumulation_index(&self) -> u64 {
        MetalCompiledAdamWStepResult::accumulation_index(self)
    }

    fn loss_weight(&self) -> u64 {
        MetalCompiledAdamWStepResult::loss_weight(self)
    }
}

/// Shared execution contract for one compiled training program.
///
/// Optimizer and backend implementations retain their concrete policy,
/// checkpoint, and device evidence through extension traits and inherent APIs.
/// This base interface owns only the common replay contract needed by a generic
/// training loop.
pub trait CompiledTrainingRuntime {
    type Step: CompiledTrainingStep;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step>;

    /// Replays one workload-owned batch with a rank-zero F32 learning rate.
    ///
    /// Batch conversion and the existing complete input validation both finish
    /// before CPU or device execution can mutate recurrent state.
    fn step_batch<B>(&mut self, batch: B, learning_rate: f32) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.step(
            batch.into_compiled_inputs()?,
            TensorData::scalar(learning_rate),
        )
    }

    fn step_count(&self) -> u64;

    fn capture_identity(&self) -> u64;

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;

    /// Explicitly publishes the current detached trainable-parameter frontier
    /// into any live module with the exact same canonical trainable schema.
    ///
    /// This does not synchronize frozen state, buffers, optimizer state, replay
    /// progress, or checkpoints. The source runtime remains unchanged if the
    /// snapshot or the module's one atomic replacement transaction fails.
    /// Optimizer runtimes with an authenticated compile-time freeze policy
    /// publish only their effective trainable frontier and preserve those
    /// policy-frozen host identities as well.
    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        let parameters = self.parameter_snapshots()?;
        module.load_trainable_parameters_exact(&parameters)
    }
}

fn publish_parameters_with_freeze_policy(
    module: &dyn Module,
    parameters: BTreeMap<String, TensorData>,
    frozen_parameters: &BTreeSet<String>,
) -> Result<LoadReport> {
    if frozen_parameters.is_empty() {
        return module.load_trainable_parameters_exact(&parameters);
    }
    CompiledModuleSeal::capture(module, frozen_parameters)?.publish(module, &parameters)
}

/// Portable checkpoint capability for a compiled training runtime.
///
/// Keeping persistence separate lets non-checkpointable optimizers implement
/// the common execution contract without inventing an empty checkpoint type.
pub trait CompiledCheckpointRuntime: CompiledTrainingRuntime {
    type Checkpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint>;
}

/// AdamW-specific policy and recurrent-state inspection.
///
/// CPU and Metal AdamW sessions share this extension and the same checkpoint
/// type while retaining their concrete step-result and device-report types.
pub trait CompiledAdamWRuntime:
    CompiledTrainingRuntime<Step: CompiledAdamWStep>
    + CompiledCheckpointRuntime<Checkpoint = CompiledAdamWCheckpoint>
{
    fn gradient_accumulation_steps(&self) -> u64;

    fn max_gradient_norm(&self) -> Option<f32>;

    fn loss_scale(&self) -> f32;

    /// Whether completed-window loss aggregation is captured and reported.
    fn window_loss_report_enabled(&self) -> bool {
        false
    }

    fn optimizer_step(&self) -> Result<u64>;

    fn accumulation_index(&self) -> Result<u64>;

    /// Atomically discards a retained partial gradient window. Empty windows
    /// are exact no-ops and successful microbatch replay progress never rewinds.
    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        Err(training(
            "compiled AdamW runtime does not support accumulation cancellation",
        ))
    }

    /// Stable identity of a separately captured accumulation-reset transition.
    /// Runtimes that retain historical host-side reset semantics return `None`.
    fn zero_grad_capture_identity(&self) -> Option<u64> {
        None
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>>;
}

/// Optional extension for committing an incomplete compiled AdamW window.
///
/// The transition consumes only the live parameter, moment, accumulator, and
/// optimizer-cursor frontier plus the explicit learning rate. It cannot reach
/// a training batch, forward/backward graph, or recurrent dropout state.
/// CPU executes the exact retained mixed capture. Strict Metal reuses its
/// authenticated state-only projection against the live epoch banks without
/// changing user code or staging gradients through the host.
pub trait CompiledAdamWFlushRuntime: CompiledAdamWRuntime {
    type Flush: CompiledAdamWFlush;

    /// Atomically averages the currently retained `k < N` gradients by `k`,
    /// clips once, performs one AdamW update, and resets the partial window.
    /// An empty window is an exact no-op.
    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush>;

    /// Stable identity of the authenticated auxiliary state transition.
    fn flush_capture_identity(&self) -> Option<u64>;
}

/// CPU capability for replaying AdamW with a captured learning-rate policy.
///
/// Metal deliberately does not implement this capability until it can admit
/// the same policy without weakening strict planning.
pub trait CompiledScheduledAdamWRuntime: CompiledAdamWRuntime {
    type ScheduledFlush: CompiledAdamWFlush;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr>;

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step>;

    fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<Self::Step>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush>;
}

/// Resource-free output of compiled training graph construction.
#[derive(Clone)]
struct CompiledTrainingPlan {
    capture: CapturedMixedSchedule,
    recurrent_capture: CapturedStatefulInference,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    clip_report: bool,
    window_loss_report: bool,
    parameter_buffers: BTreeMap<String, u64>,
    optimizer_buffers: BTreeMap<RecurrentStateKey, u64>,
    workload_buffers: BTreeMap<RecurrentStateKey, u64>,
    state_input_buffers: BTreeMap<String, u64>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    state_values: BTreeMap<RecurrentStateKey, TensorData>,
    state_versions: BTreeMap<RecurrentStateKey, u64>,
    frozen_parameter_nodes: BTreeSet<NodeId>,
    step: u64,
}

#[derive(Clone)]
struct CompiledAdamWAuxiliaryPlan {
    capture: CapturedMixedSchedule,
    recurrent_capture: CapturedStatefulInference,
    state_buffers: BTreeMap<RecurrentStateKey, u64>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    capture_identity: u64,
    clip_report: bool,
    window_loss_report: bool,
}

#[derive(Clone)]
struct CompiledEvaluationPlan {
    inference: crate::CapturedInference,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    parameter_inputs: BTreeMap<String, String>,
    capture_identity: u64,
}

#[derive(Clone)]
struct CpuCompiledEvaluation {
    plan: CompiledEvaluationPlan,
}

/// One static CPU training program with runtime-owned recurrent state.
struct CpuCompiledTrainingProgram {
    capture: CapturedMixedSchedule,
    recurrent_capture: CapturedStatefulInference,
    runtime: EffectRuntime,
    cursor: MixedReplayCursor,
    inputs: BTreeMap<String, (Shape, DType)>,
    output_names: Vec<String>,
    clip_report: bool,
    window_loss_report: bool,
    parameter_buffers: BTreeMap<String, u64>,
    optimizer_buffers: BTreeMap<RecurrentStateKey, u64>,
    workload_buffers: BTreeMap<RecurrentStateKey, u64>,
    state_input_buffers: BTreeMap<String, u64>,
    state_input_keys: BTreeMap<String, RecurrentStateKey>,
    frozen_parameter_nodes: BTreeSet<NodeId>,
    step: u64,
}

struct CpuAuxiliaryReplay {
    cursor: MixedReplayCursor,
    next_main_cursor: MixedReplayCursor,
    provided: BTreeMap<String, TensorData>,
}

struct CompiledAdamWAuxiliaryReports {
    clip_report: Option<CompiledAdamWClipReport>,
    window_loss: Option<CompiledAdamWWindowLossValue>,
}

/// Gives only terminal public aliases a concrete schedule owner before mixed
/// capture. RGSM stays owner-only, while ordinary already-materialized losses
/// and named outputs retain their existing graph and capture identities.
fn materialize_compiled_public_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
) -> Result<Vec<NodeId>> {
    let aliases = compiled_requested_aliases(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                graph.contiguous(*node)
            } else {
                Ok(*node)
            }
        })
        .collect()
}

/// Gives public values that coincide with recurrent inputs or successors a
/// distinct storage owner. Stateful capture deliberately rejects shared node
/// identity even when the value is otherwise already materialized.
fn materialize_compiled_recurrent_public_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
    state_links: &[InferenceStateLink],
) -> Result<Vec<NodeId>> {
    let requested = materialize_compiled_public_aliases(graph, requested)?;
    let state_nodes = state_links
        .iter()
        .flat_map(|link| [link.input(), link.output()])
        .collect::<BTreeSet<_>>();
    requested
        .into_iter()
        .map(|node| {
            if state_nodes.contains(&node) {
                let shape = graph.shape(node)?.clone();
                let dtype = graph.dtype(node)?;
                checked_descriptor(&shape, dtype)?;
                Ok(graph.push(crate::Op::Contiguous { input: node }, shape, dtype))
            } else {
                Ok(node)
            }
        })
        .collect()
}

fn compiled_requested_aliases(graph: &Graph, requested: &[NodeId]) -> Result<BTreeSet<NodeId>> {
    Ok(schedule_many(graph, requested)
        .map_err(schedule_error)?
        .requested_passthroughs
        .iter()
        .map(|alias| alias.requested)
        .collect())
}

fn compiled_unowned_requests(graph: &Graph, requested: &[NodeId]) -> Result<BTreeSet<NodeId>> {
    let preview = schedule_many(graph, requested).map_err(schedule_error)?;
    let owners = preview
        .items
        .iter()
        .flat_map(|item| item.outputs.iter())
        .map(|output| output.id)
        .collect::<BTreeSet<_>>();
    Ok(requested
        .iter()
        .copied()
        .filter(|node| !owners.contains(&(node.index() as u64)))
        .collect())
}

/// Recurrent outputs must name storage produced by the captured transition,
/// even when their value is a constant or an existing buffer alias. Insert an
/// explicit copy only for those passthroughs; ordinary computed successors
/// retain their original identity and schedule.
fn materialize_compiled_state_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
) -> Result<Vec<NodeId>> {
    let aliases = compiled_unowned_requests(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                let shape = graph.shape(*node)?.clone();
                let dtype = graph.dtype(*node)?;
                checked_descriptor(&shape, dtype)?;
                Ok(graph.push(crate::Op::Contiguous { input: *node }, shape, dtype))
            } else {
                Ok(*node)
            }
        })
        .collect()
}

impl CompiledTrainingPlan {
    /// Compiles one exact static training program.
    ///
    /// `build` receives the declared external inputs and detached parameter
    /// graph inputs. It returns one scalar F32 loss and deterministically named
    /// detached outputs. All parameter gradients are constructed by exactly
    /// one [`Graph::gradient_default`] traversal.
    fn compile<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_with_workload(
            optimizer,
            parameters,
            None,
            |graph, inputs, parameters, _| {
                let (loss, outputs) = build(graph, inputs, parameters)?;
                Ok((loss, outputs, None))
            },
        )
    }

    fn compile_with_workload<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        workload: Option<StateSpec>,
        build: F,
    ) -> Result<Self>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
            Option<NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>, Option<NodeId>)>,
    {
        let parameters = canonical_parameters(parameters)?;
        if parameters.is_empty() {
            return Err(training(format!(
                "compiled {} needs at least one parameter",
                optimizer.name()
            )));
        }
        if parameters
            .keys()
            .any(|name| optimizer.inputs().contains_key(name))
        {
            return Err(training(
                "compiled parameter and input names must be globally unique",
            ));
        }

        let mut graph = Graph::new();
        let inputs = optimizer
            .inputs()
            .iter()
            .map(|(name, (shape, dtype))| {
                (
                    name.clone(),
                    graph.input_dtype_requires_grad(name.clone(), shape.clone(), *dtype, false),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let learning_rate = graph.input_dtype_requires_grad(
            LEARNING_RATE_INPUT,
            Shape::from([]),
            DType::F32,
            false,
        );

        let mut specs = optimizer.state_specs(&parameters)?;
        let optimizer_spec_count = specs.len();
        if let Some(workload) = workload {
            if specs.iter().any(|spec| spec.key == workload.key) {
                return Err(training("compiled workload state key repeats"));
            }
            specs.push(workload);
        }
        let mut parameter_nodes = BTreeMap::new();
        let mut state_nodes = BTreeMap::new();
        let mut state_values = Vec::with_capacity(specs.len());
        let mut state_by_input = BTreeMap::new();
        let mut parameter_buffers = BTreeMap::new();
        let mut optimizer_buffers = BTreeMap::new();
        let mut workload_buffers = BTreeMap::new();
        let mut state_input_buffers = BTreeMap::new();
        let mut state_input_keys = BTreeMap::new();
        let optimizer_spec_count_u64 = u64::try_from(optimizer_spec_count)
            .map_err(|_| training("optimizer state count overflow"))?;
        for (ordinal, spec) in specs.iter().enumerate() {
            let ordinal = u64::try_from(ordinal).map_err(|_| training("parameter overflow"))?;
            let parameter_buffer = STATE_BUFFER_BASE
                .checked_add(ordinal)
                .ok_or_else(|| training("parameter buffer overflow"))?;
            let node = graph.input_dtype_requires_grad(
                spec.input_name.clone(),
                spec.value.shape().clone(),
                spec.value.dtype(),
                spec.requires_grad,
            );
            let state = state_for(parameter_buffer, &spec.value)?;
            state_nodes.insert(spec.key.clone(), node);
            state_by_input.insert(node, state);
            state_values.push((parameter_buffer, spec.value.clone()));
            state_input_buffers.insert(spec.input_name.clone(), parameter_buffer);
            state_input_keys.insert(spec.input_name.clone(), spec.key.clone());
            if let Some(name) = spec.key.parameter_name() {
                parameter_nodes.insert(name.to_string(), node);
                parameter_buffers.insert(name.to_string(), parameter_buffer);
            } else if ordinal < optimizer_spec_count_u64 {
                optimizer_buffers.insert(spec.key.clone(), parameter_buffer);
            } else {
                workload_buffers.insert(spec.key.clone(), parameter_buffer);
            }
        }
        if parameter_nodes.len() != parameters.len() {
            return Err(training("compiled optimizer omitted parameter state"));
        }

        let workload_node = specs
            .get(optimizer_spec_count)
            .map(|spec| state_nodes[&spec.key]);
        let (loss, outputs, workload_successor) =
            build(&mut graph, &inputs, &parameter_nodes, workload_node)?;
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            optimizer.inputs().keys().chain(parameters.keys()),
        )?;

        let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
        let gradients = optimizer.gradients(&mut graph, loss, &targets)?;
        if gradients.len() != targets.len() {
            return Err(training("compiled gradient target count mismatch"));
        }
        let gradients = parameter_nodes
            .keys()
            .cloned()
            .zip(gradients)
            .collect::<BTreeMap<_, _>>();
        let CompiledOptimizerLowering {
            mut updates,
            clip_report,
            window_loss_report,
        } = optimizer.lower_updates(
            &mut graph,
            CompiledOptimizerLoweringContext {
                loss,
                learning_rate,
                inputs: &inputs,
                parameters: &parameter_nodes,
                gradients: &gradients,
                states: &state_nodes,
            },
        )?;
        match (specs.get(optimizer_spec_count), workload_successor) {
            (Some(spec), Some(successor)) => {
                updates.insert(spec.key.clone(), successor);
            }
            (None, None) => {}
            _ => return Err(training("compiled workload successor set mismatch")),
        }
        if updates.len() != specs.len() || specs.iter().any(|spec| !updates.contains_key(&spec.key))
        {
            return Err(training("compiled optimizer successor set mismatch"));
        }
        let clip_requested = clip_report
            .into_iter()
            .flat_map(|report| [report.pre_clip_global_norm, report.applied_scale]);
        let window_loss_requested = window_loss_report
            .into_iter()
            .flat_map(|report| [report.mean_loss, report.loss_weight]);
        let state_links = specs
            .iter()
            .map(|spec| InferenceStateLink::new(state_nodes[&spec.key], updates[&spec.key]))
            .collect::<Vec<_>>();
        let public_requested = std::iter::once(loss)
            .chain(outputs.values().copied())
            .chain(clip_requested)
            .chain(window_loss_requested)
            .collect::<Vec<_>>();
        let public_requested = materialize_compiled_recurrent_public_aliases(
            &mut graph,
            &public_requested,
            &state_links,
        )?;
        let initial_state = specs
            .iter()
            .map(|spec| (spec.input_name.clone(), spec.value.clone()))
            .collect();
        let recurrent_capture = CapturedStatefulInference::from_graph(
            &graph,
            &public_requested,
            &state_links,
            initial_state,
        )
        .map_err(captured_inference_error)?;

        let public_output_count = public_requested.len();
        let mut requested = Vec::with_capacity(public_output_count + updates.len());
        requested.extend(public_requested);
        for spec in &specs {
            requested.push(updates[&spec.key]);
        }
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled pure prefix has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured =
            CapturedSchedule::capture(&graph, &pure, &requested[..public_output_count])
                .map_err(replay_error)?;
        if captured.requested.len() != public_output_count {
            return Err(training("compiled capture output count mismatch"));
        }

        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(updates.len());
        for (ordinal, spec) in specs.iter().enumerate() {
            let next = updates[&spec.key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let buffer = state_values[ordinal].0;
            let destination = effects
                .insert(buffer, spec.value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(spec.value.shape().clone(), spec.value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            let effect_index =
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?;
            effect_bindings.push(value_binding(&pure, next, effect_index)?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let states = effect_states(&effects)?;
        let capture =
            CapturedMixedSchedule::from_parts(captured, &mixed, states).map_err(replay_error)?;
        validate_external_binding_ownership(&capture, optimizer.inputs().keys())?;

        let output_names = outputs.keys().cloned().collect();
        Ok(Self {
            capture,
            recurrent_capture,
            inputs: optimizer.inputs().clone(),
            output_names,
            clip_report: clip_report.is_some(),
            window_loss_report: window_loss_report.is_some(),
            parameter_buffers,
            optimizer_buffers,
            workload_buffers,
            state_input_buffers,
            state_input_keys,
            state_values: specs
                .iter()
                .map(|spec| (spec.key.clone(), spec.value.clone()))
                .collect(),
            state_versions: specs.iter().map(|spec| (spec.key.clone(), 0)).collect(),
            frozen_parameter_nodes: BTreeSet::new(),
            step: 0,
        })
    }

    fn capture_identity(&self) -> Result<u64> {
        Ok(self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .capture_identity())
    }

    fn recurrent_capture(&self) -> Result<CapturedStatefulInference> {
        let initial_state = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                let value = self
                    .state_values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| training("compiled plan state value is absent"))?;
                Ok((input.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.recurrent_capture
            .clone()
            .with_initial_state(initial_state)
            .map_err(captured_inference_error)
    }

    fn metal_plan(
        &self,
        renderer: MetalRenderer,
        host_token_inputs: &BTreeMap<String, Shape>,
        evaluation: Option<CompiledEvaluationPlan>,
    ) -> Result<MetalCompiledTrainingPlan> {
        let recurrent = self.recurrent_capture()?;
        let recurrent = recurrent
            .with_authenticated_training_host_indices(
                host_token_inputs,
                &self.frozen_parameter_nodes,
            )
            .map_err(captured_inference_error)?;
        let inner = MetalStatefulInferencePlan::new(recurrent.clone(), renderer.clone()).map_err(
            |error| {
                let detail = if matches!(&error, MetalError::Unsupported(_)) {
                    recurrent
                        .capture()
                        .items
                        .iter()
                        .find_map(|item| {
                            renderer.render(&item.kernel).err().map(|item_error| {
                                format!(
                                    " at schedule item {} (node {}, {:?}): {item_error}",
                                    item.id,
                                    item.node.index(),
                                    item.kernel.operation()
                                )
                            })
                        })
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                training(format!("compiled Metal runtime: {error:?}{detail}"))
            },
        )?;
        let evaluation = evaluation
            .map(|evaluation| {
                let parameter_names = evaluation
                    .parameter_inputs
                    .values()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                MetalFixedStateReadPlan::new(
                    evaluation.inference,
                    renderer,
                    &inner,
                    &parameter_names,
                )
                .map(|plan| (plan, evaluation.output_names, evaluation.capture_identity))
                .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledTrainingPlan {
            inner,
            inputs: self.inputs.clone(),
            output_names: self.output_names.clone(),
            state_input_keys: self.state_input_keys.clone(),
            program_identity: self.capture_identity()?,
            evaluation,
        })
    }

    #[cfg(test)]
    fn restore_frontier(
        self,
        step: u64,
        values: BTreeMap<RecurrentStateKey, TensorData>,
    ) -> Result<Self> {
        let versions = values.keys().cloned().map(|key| (key, step)).collect();
        self.restore_frontier_with_versions(step, values, versions)
    }

    fn restore_frontier_with_versions(
        mut self,
        step: u64,
        values: BTreeMap<RecurrentStateKey, TensorData>,
        versions: BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<Self> {
        let expected = self
            .parameter_buffers
            .keys()
            .map(RecurrentStateKey::parameter)
            .chain(self.optimizer_buffers.keys().cloned())
            .chain(self.workload_buffers.keys().cloned())
            .collect::<BTreeSet<_>>();
        if values.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(training("compiled checkpoint state names mismatch"));
        }
        if versions.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(training("compiled checkpoint state versions mismatch"));
        }
        for (key, value) in &values {
            let current = self
                .state_values
                .get(key)
                .ok_or_else(|| training("compiled checkpoint state is absent"))?;
            if value.shape() != current.shape() || value.dtype() != current.dtype() {
                return Err(training("compiled checkpoint state descriptor mismatch"));
            }
            checked_bytes(value)?;
        }
        self.state_values = values;
        self.state_versions = versions;
        self.step = step;
        let frontier = self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .frontier()
            .iter()
            .cloned()
            .map(|mut state| {
                let key = self
                    .state_input_buffers
                    .iter()
                    .find_map(|(input, buffer)| {
                        (*buffer == state.buffer).then(|| self.state_input_keys[input].clone())
                    })
                    .ok_or_else(|| training("compiled checkpoint state buffer is absent"))?;
                state.version = self.state_versions[&key];
                Ok(state)
            })
            .collect::<Result<Vec<_>>>()?;
        MixedReplayCursor::resume(&self.capture, frontier).map_err(replay_error)?;
        Ok(self)
    }

    fn prepare_cpu(&self) -> Result<CpuCompiledTrainingProgram> {
        self.prepare_cpu_with_non_finite_policy(CpuNonFinitePolicy::Propagate)
    }

    fn prepare_cpu_with_non_finite_policy(
        &self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> Result<CpuCompiledTrainingProgram> {
        if non_finite_policy == CpuNonFinitePolicy::RejectTransition {
            validate_finite_tensors(self.state_values.values(), "prepared recurrent state")?;
        }
        let initial_states = self
            .state_input_buffers
            .iter()
            .map(|(input, buffer)| {
                let key = self
                    .state_input_keys
                    .get(input)
                    .ok_or_else(|| training("compiled plan state key is absent"))?;
                let value = self
                    .state_values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| training("compiled plan state value is absent"))?;
                Ok((*buffer, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_states(initial_states)
            .map_err(runtime_error)?;
        let cursor = self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?;
        let mut program = CpuCompiledTrainingProgram {
            capture: self.capture.clone(),
            recurrent_capture: self.recurrent_capture.clone(),
            runtime,
            cursor,
            inputs: self.inputs.clone(),
            output_names: self.output_names.clone(),
            clip_report: self.clip_report,
            window_loss_report: self.window_loss_report,
            parameter_buffers: self.parameter_buffers.clone(),
            optimizer_buffers: self.optimizer_buffers.clone(),
            workload_buffers: self.workload_buffers.clone(),
            state_input_buffers: self.state_input_buffers.clone(),
            state_input_keys: self.state_input_keys.clone(),
            frozen_parameter_nodes: self.frozen_parameter_nodes.clone(),
            step: 0,
        };
        if self.step != 0 || self.state_versions.values().any(|version| *version != 0) {
            program.restore_frontier(self.step, &self.state_values, &self.state_versions)?;
        }
        Ok(program)
    }
}

impl CompiledAdamWAuxiliaryPlan {
    fn compile_partial_flush(
        training_plan: &CompiledTrainingPlan,
        config: &CompiledAdamWConfig,
    ) -> Result<Self> {
        if config.gradient_accumulation_steps <= 1 {
            return Err(training(
                "compiled AdamW partial flush requires gradient accumulation",
            ));
        }

        let state_buffers = training_plan
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(
                training_plan
                    .optimizer_buffers
                    .iter()
                    .map(|(key, buffer)| (key.clone(), *buffer)),
            )
            .collect::<BTreeMap<_, _>>();
        let mut graph = Graph::new();
        let learning_rate = graph.input_dtype_requires_grad(
            LEARNING_RATE_INPUT,
            Shape::from([]),
            DType::F32,
            false,
        );
        let mut state_nodes = BTreeMap::new();
        let mut state_by_input = BTreeMap::new();
        let mut specs = Vec::with_capacity(state_buffers.len());
        for (input_name, key) in &training_plan.state_input_keys {
            let Some(buffer) = state_buffers.get(key).copied() else {
                continue;
            };
            let value = training_plan
                .state_values
                .get(key)
                .cloned()
                .ok_or_else(|| training("compiled partial flush state value is absent"))?;
            let node = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                false,
            );
            state_nodes.insert(key.clone(), node);
            state_by_input.insert(node, state_for(buffer, &value)?);
            specs.push((input_name.clone(), key.clone(), value, node, buffer));
        }
        if specs.len() != state_buffers.len() {
            return Err(training("compiled partial flush state schema differs"));
        }

        let parameters = state_nodes
            .iter()
            .filter_map(|(key, node)| key.parameter_name().map(|name| (name.to_owned(), *node)))
            .collect::<BTreeMap<_, _>>();
        if parameters.len() != training_plan.parameter_buffers.len() {
            return Err(training("compiled partial flush parameter schema differs"));
        }
        let accumulation_index_key =
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex);
        let index = state_nodes
            .get(&accumulation_index_key)
            .copied()
            .ok_or_else(|| training("compiled partial flush accumulation index is absent"))?;
        let token_count_key = config
            .token_weight_mask_input
            .as_ref()
            .map(|_| RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedTokenCount));
        let divisor = match &token_count_key {
            Some(key) => {
                let count = state_nodes.get(key).copied().ok_or_else(|| {
                    training("compiled partial flush accumulated token count is absent")
                })?;
                graph.cast(count, DType::F32)?
            }
            None => graph.cast(index, DType::F32)?,
        };
        let window_loss_report = if config.window_loss_report {
            let numerator_key =
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator);
            let numerator = state_nodes.get(&numerator_key).copied().ok_or_else(|| {
                training("compiled partial flush accumulated loss numerator is absent")
            })?;
            let loss_weight = match &token_count_key {
                Some(key) => state_nodes[key],
                None => index,
            };
            Some((
                numerator_key,
                CompiledAdamWWindowLossNodes {
                    mean_loss: graph.div(numerator, divisor)?,
                    loss_weight,
                },
            ))
        } else {
            None
        };
        let mut gradients = BTreeMap::new();
        for name in parameters.keys() {
            let accumulator_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulator = state_nodes
                .get(&accumulator_key)
                .copied()
                .ok_or_else(|| training("compiled partial flush accumulator is absent"))?;
            gradients.insert(name.clone(), graph.div(accumulator, divisor)?);
        }
        let clipped = clip_gradients_by_global_norm(config, &mut graph, &gradients)?;
        let learning_rate =
            lower_adamw_learning_rate(config, &mut graph, learning_rate, &state_nodes)?;
        let mut updates = lower_adamw_update_candidates(
            config,
            &mut graph,
            learning_rate,
            &parameters,
            &clipped.gradients,
            &state_nodes,
        )?;
        // Token-weighted flush divides by the retained count instead of the
        // accumulation index. Derive the exact reset from the old index so
        // the strict recurrent capture still owns every state input.
        let zero_index = if token_count_key.is_some() {
            graph.sub(index, index)?
        } else {
            graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?
        };
        updates.insert(accumulation_index_key, zero_index);
        if let Some(key) = token_count_key {
            let zero_count = graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?;
            updates.insert(key, zero_count);
        }
        if let Some((key, _)) = &window_loss_report {
            updates.insert(key.clone(), scalar_f32(&mut graph, 0.0)?);
        }
        for (name, parameter) in &parameters {
            let key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let zero = graph.lazy_full_with_dtype(
                graph.shape(*parameter)?.clone(),
                Scalar::I(0),
                DType::F32,
            )?;
            updates.insert(key, zero);
        }
        let successor_keys = specs
            .iter()
            .map(|(_, key, ..)| key.clone())
            .collect::<Vec<_>>();
        let successors = successor_keys
            .iter()
            .map(|key| updates[key])
            .collect::<Vec<_>>();
        let successors = materialize_compiled_state_aliases(&mut graph, &successors)?;
        for (key, successor) in successor_keys.into_iter().zip(successors) {
            updates.insert(key, successor);
        }
        if updates.len() != specs.len()
            || specs.iter().any(|(_, key, ..)| !updates.contains_key(key))
        {
            return Err(training("compiled partial flush successor schema differs"));
        }

        let state_links = specs
            .iter()
            .map(|(_, key, _, node, _)| InferenceStateLink::new(*node, updates[key]))
            .collect::<Vec<_>>();
        let initial_state = specs
            .iter()
            .map(|(input, _, value, _, _)| (input.clone(), value.clone()))
            .collect();
        let clip_requested = clipped
            .report
            .into_iter()
            .flat_map(|report| [report.pre_clip_global_norm, report.applied_scale]);
        let window_loss_requested = window_loss_report
            .iter()
            .flat_map(|(_, report)| [report.mean_loss, report.loss_weight]);
        let public_requested = clip_requested
            .chain(window_loss_requested)
            .collect::<Vec<_>>();
        let public_requested = materialize_compiled_recurrent_public_aliases(
            &mut graph,
            &public_requested,
            &state_links,
        )?;
        let recurrent_capture = CapturedStatefulInference::from_graph(
            &graph,
            &public_requested,
            &state_links,
            initial_state,
        )
        .map_err(captured_inference_error)?;

        let public_output_count = public_requested.len();
        let mut requested = public_requested;
        requested.extend(specs.iter().map(|(_, key, ..)| updates[key]));
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled partial flush has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured =
            CapturedSchedule::capture(&graph, &pure, &requested[..public_output_count])
                .map_err(replay_error)?;
        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(specs.len());
        for (ordinal, (_, key, value, _, buffer)) in specs.iter().enumerate() {
            let next = updates[key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let destination = effects
                .insert(*buffer, value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(value.shape().clone(), value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            effect_bindings.push(value_binding(
                &pure,
                next,
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?,
            )?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let capture = CapturedMixedSchedule::from_parts(captured, &mixed, effect_states(&effects)?)
            .map_err(replay_error)?;
        validate_external_binding_ownership(&capture, std::iter::empty::<&String>())?;
        let capture_identity = capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .capture_identity();
        Ok(Self {
            capture,
            recurrent_capture,
            state_buffers,
            state_input_keys: specs
                .into_iter()
                .map(|(input, key, ..)| (input, key))
                .collect(),
            capture_identity,
            clip_report: clipped.report.is_some(),
            window_loss_report: window_loss_report.is_some(),
        })
    }

    fn compile_zero_grad(training_plan: &CompiledTrainingPlan) -> Result<Self> {
        let state_buffers = training_plan
            .optimizer_buffers
            .iter()
            .filter(|(key, _)| key.is_accumulation_reset_state())
            .map(|(key, buffer)| (key.clone(), *buffer))
            .collect::<BTreeMap<_, _>>();
        if state_buffers.is_empty() {
            return Err(training(
                "compiled AdamW zero-grad requires gradient accumulation",
            ));
        }

        let mut graph = Graph::new();
        let mut state_by_input = BTreeMap::new();
        let mut specs = Vec::with_capacity(state_buffers.len());
        let mut successors = BTreeMap::new();
        for (input_name, key) in &training_plan.state_input_keys {
            let Some(buffer) = state_buffers.get(key).copied() else {
                continue;
            };
            let value = training_plan
                .state_values
                .get(key)
                .cloned()
                .ok_or_else(|| training("compiled zero-grad state value is absent"))?;
            let input = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                false,
            );
            // Retain the old state as an authenticated dependency without
            // deriving zero as `input - input`, which would preserve NaN/Inf.
            let false_condition = if value.dtype() == DType::F32 {
                let finite = graph.isfinite(input)?;
                let not_finite = graph.logical_not(finite)?;
                graph.logical_and(finite, not_finite)?
            } else if value.dtype() == DType::U64 {
                // U64 subtraction is defined modulo 2^64, so this remains
                // false for every value while retaining the state dependency.
                // Casting the difference avoids a native C self-comparison,
                // which Apple Clang rejects under -Wtautological-compare.
                let zero = graph.sub(input, input)?;
                graph.cast(zero, DType::Bool)?
            } else {
                return Err(training("compiled zero-grad state dtype is unsupported"));
            };
            let zero =
                graph.lazy_full_with_dtype(value.shape().clone(), Scalar::I(0), value.dtype())?;
            let successor = graph.select(false_condition, input, zero)?;
            state_by_input.insert(input, state_for(buffer, &value)?);
            successors.insert(key.clone(), successor);
            specs.push((input_name.clone(), key.clone(), value, input, buffer));
        }
        if specs.len() != state_buffers.len() {
            return Err(training("compiled zero-grad state schema differs"));
        }

        let successor_keys = specs
            .iter()
            .map(|(_, key, ..)| key.clone())
            .collect::<Vec<_>>();
        let materialized = materialize_compiled_state_aliases(
            &mut graph,
            &successor_keys
                .iter()
                .map(|key| successors[key])
                .collect::<Vec<_>>(),
        )?;
        for (key, successor) in successor_keys.into_iter().zip(materialized) {
            successors.insert(key, successor);
        }
        let state_links = specs
            .iter()
            .map(|(_, key, _, input, _)| InferenceStateLink::new(*input, successors[key]))
            .collect::<Vec<_>>();
        let initial_state = specs
            .iter()
            .map(|(input, _, value, _, _)| (input.clone(), value.clone()))
            .collect();
        let recurrent_capture =
            CapturedStatefulInference::from_graph(&graph, &[], &state_links, initial_state)
                .map_err(captured_inference_error)?;

        let requested = specs
            .iter()
            .map(|(_, key, ..)| successors[key])
            .collect::<Vec<_>>();
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled zero-grad has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured = CapturedSchedule::capture(&graph, &pure, &[]).map_err(replay_error)?;
        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(specs.len());
        for (ordinal, (_, key, value, _, buffer)) in specs.iter().enumerate() {
            let next = successors[key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let destination = effects
                .insert(*buffer, value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(value.shape().clone(), value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            effect_bindings.push(value_binding(
                &pure,
                next,
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?,
            )?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let capture = CapturedMixedSchedule::from_parts(captured, &mixed, effect_states(&effects)?)
            .map_err(replay_error)?;
        validate_external_binding_ownership(&capture, std::iter::empty::<&String>())?;
        let capture_identity = capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .capture_identity();
        Ok(Self {
            capture,
            recurrent_capture,
            state_buffers,
            state_input_keys: specs
                .into_iter()
                .map(|(input, key, ..)| (input, key))
                .collect(),
            capture_identity,
            clip_report: false,
            window_loss_report: false,
        })
    }

    fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    fn with_frontier(mut self, values: &BTreeMap<RecurrentStateKey, TensorData>) -> Result<Self> {
        let initial_state = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                Ok((
                    input.clone(),
                    values
                        .get(key)
                        .cloned()
                        .ok_or_else(|| training("compiled auxiliary frontier is absent"))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.recurrent_capture = self
            .recurrent_capture
            .with_initial_state(initial_state)
            .map_err(captured_inference_error)?;
        Ok(self)
    }
}

impl CompiledEvaluationPlan {
    fn compile_with_parameter_plan<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let mut graph = Graph::new();
        let inputs = training_plan
            .inner
            .inputs
            .iter()
            .map(|(name, (shape, dtype))| {
                (
                    name.clone(),
                    graph.input_dtype_requires_grad(name.clone(), shape.clone(), *dtype, false),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut parameters = BTreeMap::new();
        let mut parameter_inputs = BTreeMap::new();
        let mut residents = BTreeMap::new();
        for init in parameter_plan.initial_parameters()? {
            let key = RecurrentStateKey::parameter(init.name());
            let input_name = training_plan
                .inner
                .state_input_keys
                .iter()
                .find_map(|(input, candidate)| (candidate == &key).then(|| input.clone()))
                .ok_or_else(|| training("compiled evaluation parameter state is absent"))?;
            let value = training_plan
                .inner
                .state_values
                .get(&key)
                .cloned()
                .ok_or_else(|| training("compiled evaluation parameter value is absent"))?;
            let node = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                true,
            );
            parameters.insert(init.name().to_owned(), node);
            parameter_inputs.insert(init.name().to_owned(), input_name.clone());
            residents.insert(input_name, (node, value));
        }
        let (loss, outputs) = parameter_plan.lower(&mut graph, &parameters, |graph| {
            build(module, graph, &inputs)
        })?;
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            training_plan
                .inner
                .inputs
                .keys()
                .chain(parameter_inputs.keys()),
        )?;
        let requested = std::iter::once(loss)
            .chain(outputs.values().copied())
            .collect::<Vec<_>>();
        let requested = materialize_compiled_public_aliases(&mut graph, &requested)?;
        let inference =
            crate::CapturedInference::from_graph_residents(&graph, &requested, residents, &[])
                .map_err(captured_inference_error)?
                .with_authenticated_fixed_host_gathers(&training_plan.host_token_inputs)
                .map_err(captured_inference_error)?;
        let transient_names = inference
            .transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        if transient_names
            != training_plan
                .inner
                .inputs
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        {
            return Err(training("compiled evaluation input inventory differs"));
        }
        let capture_identity = inference.capture().identity;
        Ok(Self {
            inference,
            inputs: training_plan.inner.inputs.clone(),
            output_names: outputs.keys().cloned().collect(),
            parameter_inputs,
            capture_identity,
        })
    }

    fn bind(
        &self,
        mut inputs: BTreeMap<String, TensorData>,
        parameters: BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<String, TensorData>> {
        validate_evaluation_inputs(&self.inputs, &inputs)?;
        if parameters.keys().ne(self.parameter_inputs.keys()) {
            return Err(training("compiled evaluation parameter inventory differs"));
        }
        for (name, value) in parameters {
            let input = self
                .parameter_inputs
                .get(&name)
                .ok_or_else(|| training("compiled evaluation parameter is absent"))?;
            inputs.insert(input.clone(), value);
        }
        Ok(inputs)
    }

    fn evaluate(
        &self,
        inputs: BTreeMap<String, TensorData>,
        parameters: BTreeMap<String, TensorData>,
    ) -> Result<CompiledEvaluationResult> {
        let inputs = self.bind(inputs, parameters)?;
        let values = self
            .inference
            .capture()
            .replay(&inputs)
            .map_err(replay_error)?;
        evaluation_result(values, &self.output_names, self.capture_identity)
    }

    fn prepare_native(
        &self,
        parameters: BTreeMap<String, TensorData>,
        executor: &CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<PreparedNativeCpuEvaluation> {
        let started = Instant::now();
        let inputs = self.bind(zero_inputs(&self.inputs)?, parameters)?;
        let capture = self.inference.capture();
        let plan = executor
            .plan_native_items(capture, &inputs, vectorized)
            .map_err(replay_error)?;
        let execution_plan = ExecutionPlanSummary::from_capture(capture, true)
            .map_err(|error| training(format!("compiled native CPU summary: {error}")))?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: self.capture_identity,
            native_identity: native_cpu_identity(
                self.capture_identity,
                vectorized,
                capture.items.iter().map(|item| item.cache_key),
            ),
            vectorized,
            native_item_count: plan.item_count(),
            cache_hit_count: plan.cache_hit_count(),
            cache_miss_count: plan.cache_miss_count(),
            execution_plan,
            wall_time: started.elapsed(),
        };
        Ok(PreparedNativeCpuEvaluation { report, plan })
    }

    fn evaluate_native(
        &self,
        inputs: BTreeMap<String, TensorData>,
        parameters: BTreeMap<String, TensorData>,
        executor: &CapturedReplayExecutor,
        prepared: &mut PreparedNativeCpuEvaluation,
    ) -> Result<(CompiledEvaluationResult, NativeCpuRunReport)> {
        let inputs = self.bind(inputs, parameters)?;
        let started = Instant::now();
        let capture = self.inference.capture();
        prepared.validate(self.capture_identity, capture)?;
        let (values, traffic) = executor
            .execute_planned_native_items_observed(capture, &inputs, &mut prepared.plan)
            .map_err(replay_error)?;
        let outputs = values.requested(&capture.requested).map_err(replay_error)?;
        let schedule_cache_keys = prepared.plan.schedule_cache_keys().to_vec();
        let report = NativeCpuRunReport {
            capture_identity: self.capture_identity,
            native_identity: prepared.report.native_identity,
            vectorized: prepared.plan.vectorized(),
            successful_invocation: 0,
            native_item_count: prepared.plan.item_count(),
            schedule_cache_keys,
            traffic: native_cpu_replay_traffic(traffic),
            wall_time: started.elapsed(),
        };
        Ok((
            evaluation_result(outputs, &self.output_names, self.capture_identity)?,
            report,
        ))
    }
}

fn zero_inputs(inputs: &BTreeMap<String, (Shape, DType)>) -> Result<BTreeMap<String, TensorData>> {
    inputs
        .iter()
        .map(|(name, (shape, dtype))| {
            Ok((
                name.clone(),
                TensorData::zeros_with_dtype(shape.clone(), *dtype)?,
            ))
        })
        .collect()
}

fn native_cpu_identity(
    capture_identity: u64,
    vectorized: bool,
    cache_keys: impl IntoIterator<Item = u64>,
) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in crate::cpu_jit::RENDERER_VERSION
        .as_bytes()
        .iter()
        .chain(std::env::consts::ARCH.as_bytes())
        .chain(std::env::consts::OS.as_bytes())
    {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    for value in std::iter::once(capture_identity)
        .chain(std::iter::once(if vectorized { 1 } else { 0 }))
        .chain(cache_keys)
    {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn native_cpu_run_report(
    capture_identity: u64,
    trace: &NativeMixedReplayTrace,
    traffic: NativeReplayTraffic,
    successful_invocation: u64,
    wall_time: Duration,
) -> NativeCpuRunReport {
    NativeCpuRunReport {
        capture_identity,
        native_identity: trace.identity,
        vectorized: trace.vectorized,
        successful_invocation,
        native_item_count: trace.pure_item_cache_keys.len(),
        schedule_cache_keys: trace.pure_item_cache_keys.clone(),
        traffic: native_cpu_replay_traffic(traffic),
        wall_time,
    }
}

const fn native_cpu_replay_traffic(traffic: NativeReplayTraffic) -> NativeCpuReplayTraffic {
    NativeCpuReplayTraffic::new(
        traffic.external_input_import_count,
        traffic.external_input_import_bytes,
        traffic.borrowed_recurrent_input_bytes,
        traffic.borrowed_recurrent_output_bytes,
    )
}

fn evaluation_result(
    values: Vec<TensorData>,
    output_names: &[String],
    capture_identity: u64,
) -> Result<CompiledEvaluationResult> {
    if values.len() != 1 + output_names.len() {
        return Err(training("compiled evaluation output inventory differs"));
    }
    let mut values = values.into_iter();
    let loss = values
        .next()
        .expect("compiled evaluation output cardinality was checked");
    Ok(CompiledEvaluationResult {
        loss,
        outputs: output_names.iter().cloned().zip(values).collect(),
        capture_identity,
    })
}

fn take_compiled_clip_report(
    values: &mut impl Iterator<Item = TensorData>,
    enabled: bool,
) -> Option<CompiledAdamWClipReport> {
    enabled.then(|| {
        let norm = values
            .next()
            .expect("compiled clip-report norm cardinality was authenticated");
        let scale = values
            .next()
            .expect("compiled clip-report scale cardinality was authenticated");
        debug_assert_eq!(norm.shape(), &Shape::from([]));
        debug_assert_eq!(norm.dtype(), DType::F32);
        debug_assert_eq!(scale.shape(), &Shape::from([]));
        debug_assert_eq!(scale.dtype(), DType::F32);
        CompiledAdamWClipReport::new(norm.values()[0], scale.values()[0])
    })
}

fn take_compiled_window_loss_value(
    values: &mut impl Iterator<Item = TensorData>,
    enabled: bool,
) -> Option<CompiledAdamWWindowLossValue> {
    enabled.then(|| {
        let mean_loss = values
            .next()
            .expect("compiled window-loss mean cardinality was authenticated");
        let loss_weight = values
            .next()
            .expect("compiled window-loss weight cardinality was authenticated");
        debug_assert_eq!(mean_loss.shape(), &Shape::from([]));
        debug_assert_eq!(mean_loss.dtype(), DType::F32);
        debug_assert_eq!(loss_weight.shape(), &Shape::from([]));
        debug_assert_eq!(loss_weight.dtype(), DType::U64);
        CompiledAdamWWindowLossValue {
            mean_loss_bits: mean_loss.values()[0].to_bits(),
            loss_weight: loss_weight.scalar_at(0).as_u64(),
        }
    })
}

impl CpuCompiledTrainingProgram {
    fn prepare_native(
        &self,
        executor: &CapturedReplayExecutor,
        vectorized: bool,
        external_learning_rate: bool,
    ) -> Result<PreparedNativeCpuProgram> {
        let started = Instant::now();
        let mut provided = zero_inputs(&self.inputs)?;
        if external_learning_rate {
            provided.insert(
                LEARNING_RATE_INPUT.to_owned(),
                TensorData::zeros_with_dtype(Shape::from([]), DType::F32)?,
            );
        }
        let replay = self
            .capture
            .prepare_recurrent_native(&self.runtime, &self.cursor, &provided, executor, vectorized)
            .map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let report = NativeCpuProgramPreparationReport {
            capture_identity: self.capture_identity(),
            native_identity: trace.replay.identity,
            vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            execution_plan: self.recurrent_capture.execution_plan().clone(),
            wall_time: started.elapsed(),
        };
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    /// Executes one graph-free replay and atomically publishes every recurrent
    /// successor. The learning rate is an explicit rank-zero F32 input.
    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_inner(inputs, learning_rate, None)
    }

    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        self.step_inner_with_learning_rate(
            inputs,
            Some(learning_rate),
            CpuNonFinitePolicy::Propagate,
            true,
            injected_failure,
        )
    }

    fn step_inner_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        validate_commit_reports: bool,
        injected_failure: Option<u64>,
    ) -> Result<CompiledTrainingStepResult> {
        validate_training_inputs(&self.inputs, &inputs)?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, non_finite_policy)?;
        }
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| training("compiled training step overflow"))?;
        let mut provided = inputs;
        if let Some(learning_rate) = learning_rate {
            provided.insert(LEARNING_RATE_INPUT.to_string(), learning_rate);
        }
        let clip_report = self.clip_report;
        let clip_report_start = 1 + self.output_names.len();
        let window_loss_report = self.window_loss_report;
        let window_loss_report_start = clip_report_start + usize::from(clip_report) * 2;
        let replay = self
            .capture
            .replay_recurrent_checked(
                &mut self.runtime,
                &mut self.cursor,
                &provided,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(outputs, successors, non_finite_policy, true)?;
                    validate_staged_clip_report(
                        outputs,
                        clip_report_start,
                        clip_report && validate_commit_reports,
                        non_finite_policy,
                    )?;
                    validate_staged_window_loss_report(
                        outputs,
                        window_loss_report_start,
                        window_loss_report && validate_commit_reports,
                        non_finite_policy,
                    )
                },
            )
            .map_err(replay_error)?;
        debug_assert_eq!(
            replay.outputs.len(),
            1 + self.output_names.len()
                + usize::from(self.clip_report) * 2
                + usize::from(self.window_loss_report) * 2
        );
        let mut outputs = replay.outputs.into_iter();
        let loss = outputs
            .next()
            .expect("compiled output cardinality was validated before publication");
        let named_outputs = self
            .output_names
            .iter()
            .cloned()
            .zip(outputs.by_ref())
            .collect::<BTreeMap<_, _>>();
        let clip_report = take_compiled_clip_report(&mut outputs, self.clip_report);
        let window_loss = take_compiled_window_loss_value(&mut outputs, self.window_loss_report);
        debug_assert!(outputs.next().is_none());
        self.step = next_step;
        Ok(CompiledTrainingStepResult {
            loss,
            outputs: named_outputs,
            step: self.step,
            capture_identity: self.cursor.capture_identity(),
            clip_report,
            window_loss,
        })
    }

    fn step_native_inner_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        validate_commit_reports: bool,
        native: NativeReplayContext<'_>,
        injected_failure: Option<u64>,
    ) -> Result<(CompiledTrainingStepResult, NativeCpuRunReport)> {
        validate_training_inputs(&self.inputs, &inputs)?;
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, non_finite_policy)?;
        }
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| training("compiled training step overflow"))?;
        let mut provided = inputs;
        if let Some(learning_rate) = learning_rate {
            provided.insert(LEARNING_RATE_INPUT.to_string(), learning_rate);
        }
        let started = Instant::now();
        let clip_report = self.clip_report;
        let clip_report_start = 1 + self.output_names.len();
        let window_loss_report = self.window_loss_report;
        let window_loss_report_start = clip_report_start + usize::from(clip_report) * 2;
        let replay = self
            .capture
            .replay_recurrent_native_checked(
                &mut self.runtime,
                &mut self.cursor,
                &provided,
                native,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(
                        outputs,
                        successors.iter().copied(),
                        non_finite_policy,
                        true,
                    )?;
                    validate_staged_clip_report(
                        outputs,
                        clip_report_start,
                        clip_report && validate_commit_reports,
                        non_finite_policy,
                    )?;
                    validate_staged_window_loss_report(
                        outputs,
                        window_loss_report_start,
                        window_loss_report && validate_commit_reports,
                        non_finite_policy,
                    )
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let replay = replay.replay;
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            self.capture_identity(),
            native,
            traffic,
            next_step,
            started.elapsed(),
        );
        debug_assert_eq!(
            replay.outputs.len(),
            1 + self.output_names.len()
                + usize::from(self.clip_report) * 2
                + usize::from(self.window_loss_report) * 2
        );
        let mut outputs = replay.outputs.into_iter();
        let loss = outputs
            .next()
            .expect("compiled output cardinality was validated before publication");
        let named_outputs = self
            .output_names
            .iter()
            .cloned()
            .zip(outputs.by_ref())
            .collect();
        let clip_report = take_compiled_clip_report(&mut outputs, self.clip_report);
        let window_loss = take_compiled_window_loss_value(&mut outputs, self.window_loss_report);
        debug_assert!(outputs.next().is_none());
        self.step = next_step;
        Ok((
            CompiledTrainingStepResult {
                loss,
                outputs: named_outputs,
                step: self.step,
                capture_identity: self.cursor.capture_identity(),
                clip_report,
                window_loss,
            },
            report,
        ))
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn capture_identity(&self) -> u64 {
        self.cursor.capture_identity()
    }

    fn plan(&self) -> Result<CompiledTrainingPlan> {
        let state_frontier = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                let buffer = self
                    .state_input_buffers
                    .get(input)
                    .ok_or_else(|| training("compiled runtime state buffer is absent"))?;
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((key.clone(), (value, state.version)))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(CompiledTrainingPlan {
            capture: self.capture.clone(),
            recurrent_capture: self.recurrent_capture.clone(),
            inputs: self.inputs.clone(),
            output_names: self.output_names.clone(),
            clip_report: self.clip_report,
            window_loss_report: self.window_loss_report,
            parameter_buffers: self.parameter_buffers.clone(),
            optimizer_buffers: self.optimizer_buffers.clone(),
            workload_buffers: self.workload_buffers.clone(),
            state_input_buffers: self.state_input_buffers.clone(),
            state_input_keys: self.state_input_keys.clone(),
            state_values: state_frontier
                .iter()
                .map(|(key, (value, _))| (key.clone(), value.clone()))
                .collect(),
            state_versions: state_frontier
                .into_iter()
                .map(|(key, (_, version))| (key, version))
                .collect(),
            frozen_parameter_nodes: self.frozen_parameter_nodes.clone(),
            step: self.step,
        })
    }

    /// Returns independent owned parameter snapshots in canonical name order.
    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.snapshots(&self.parameter_buffers)
    }

    /// Current logical parameter versions. Every successful step advances all
    /// parameter and optimizer-state buffers exactly once.
    fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.versions(&self.parameter_buffers)
    }

    fn adamw_state_snapshots(
        &self,
        state: AdamWParameterState,
    ) -> Result<BTreeMap<String, TensorData>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.parameter_for_adamw_state(state)
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.snapshots(&buffers)
    }

    fn adamw_state_versions(&self, state: AdamWParameterState) -> Result<BTreeMap<String, u64>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.parameter_for_adamw_state(state)
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.versions(&buffers)
    }

    fn momentum_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.momentum_parameter_name()
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.snapshots(&buffers)
    }

    fn momentum_versions(&self) -> Result<BTreeMap<String, u64>> {
        let buffers = self
            .optimizer_buffers
            .iter()
            .filter_map(|(key, buffer)| {
                key.momentum_parameter_name()
                    .map(|name| (name.to_owned(), *buffer))
            })
            .collect::<BTreeMap<_, _>>();
        self.versions(&buffers)
    }

    fn global_snapshot(&self, state: AdamWGlobalState) -> Result<TensorData> {
        let buffer = self
            .optimizer_buffers
            .get(&RecurrentStateKey::adamw_global(state))
            .ok_or_else(|| training("compiled global optimizer state is absent"))?;
        let state = self.current_state(*buffer)?;
        Ok(self
            .runtime
            .snapshot(state)
            .map_err(runtime_error)?
            .tensor()
            .clone())
    }

    fn workload_snapshot(&self, key: &RecurrentStateKey) -> Result<TensorData> {
        let buffer = self
            .workload_buffers
            .get(key)
            .ok_or_else(|| training("compiled workload state is absent"))?;
        let state = self.current_state(*buffer)?;
        Ok(self
            .runtime
            .snapshot(state)
            .map_err(runtime_error)?
            .tensor()
            .clone())
    }

    fn restore_frontier(
        &mut self,
        step: u64,
        values: &BTreeMap<RecurrentStateKey, TensorData>,
        versions: &BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<()> {
        let buffers = self
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(
                self.optimizer_buffers
                    .iter()
                    .map(|(name, buffer)| (name.clone(), *buffer)),
            )
            .chain(
                self.workload_buffers
                    .iter()
                    .map(|(name, buffer)| (name.clone(), *buffer)),
            )
            .collect::<BTreeMap<_, _>>();
        if values.len() != buffers.len() || values.keys().ne(buffers.keys()) {
            return Err(training("compiled checkpoint state names mismatch"));
        }
        if versions.len() != buffers.len() || versions.keys().ne(buffers.keys()) {
            return Err(training("compiled checkpoint state versions mismatch"));
        }

        let mut snapshots = Vec::with_capacity(buffers.len());
        for (name, buffer) in buffers {
            let value = &values[&name];
            let current = self.current_state(buffer)?;
            if value.shape() != &current.shape || value.dtype() != current.dtype {
                return Err(training("compiled checkpoint state descriptor mismatch"));
            }
            checked_bytes(value)?;
            let mut state = current.clone();
            state.version = versions[&name];
            snapshots.push((state, value.clone()));
        }

        let frontier = snapshots
            .iter()
            .map(|(state, _)| state.clone())
            .collect::<Vec<_>>();
        let cursor = MixedReplayCursor::resume(&self.capture, frontier).map_err(replay_error)?;
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_snapshots(snapshots)
            .map_err(runtime_error)?;
        self.runtime = runtime;
        self.cursor = cursor;
        self.step = step;
        Ok(())
    }

    #[cfg(test)]
    fn replace_state_values(
        &mut self,
        step: u64,
        replacements: BTreeMap<RecurrentStateKey, TensorData>,
    ) -> Result<()> {
        let plan = self.plan()?;
        let mut values = plan.state_values;
        for (key, value) in replacements {
            let current = values
                .get(&key)
                .ok_or_else(|| training("compiled replacement state is absent"))?;
            if value.shape() != current.shape() || value.dtype() != current.dtype() {
                return Err(training("compiled replacement state descriptor mismatch"));
            }
            values.insert(key, value);
        }
        self.restore_frontier(step, &values, &plan.state_versions)
    }

    fn prepare_auxiliary_replay(
        &self,
        transition: &CompiledAdamWAuxiliaryPlan,
        learning_rate: Option<TensorData>,
    ) -> Result<CpuAuxiliaryReplay> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let selected_buffers = transition
            .state_buffers
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        let selected_frontier = self
            .cursor
            .frontier()
            .iter()
            .filter(|state| selected_buffers.contains(&state.buffer))
            .cloned()
            .collect::<Vec<_>>();
        if selected_frontier.len() != selected_buffers.len() {
            return Err(training("compiled auxiliary state frontier is incomplete"));
        }
        let cursor = MixedReplayCursor::resume(&transition.capture, selected_frontier)
            .map_err(replay_error)?;
        let next_frontier = self
            .cursor
            .frontier()
            .iter()
            .cloned()
            .map(|mut state| {
                if selected_buffers.contains(&state.buffer) {
                    state.version = state.version.checked_add(1).ok_or_else(|| {
                        training("compiled auxiliary state version would overflow")
                    })?;
                }
                Ok(state)
            })
            .collect::<Result<Vec<_>>>()?;
        let next_main_cursor =
            MixedReplayCursor::resume(&self.capture, next_frontier).map_err(replay_error)?;

        let mut provided = BTreeMap::new();
        if let Some(learning_rate) = learning_rate {
            provided.insert(LEARNING_RATE_INPUT.to_owned(), learning_rate);
        }
        Ok(CpuAuxiliaryReplay {
            cursor,
            next_main_cursor,
            provided,
        })
    }

    fn replay_auxiliary_transition(
        &mut self,
        transition: &CompiledAdamWAuxiliaryPlan,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWAuxiliaryReports> {
        let mut prepared = self.prepare_auxiliary_replay(transition, learning_rate)?;
        let clip_report = transition.clip_report;
        let window_loss_report = transition.window_loss_report;
        let window_loss_report_start = usize::from(clip_report) * 2;
        let replay = transition
            .capture
            .replay_recurrent_checked(
                &mut self.runtime,
                &mut prepared.cursor,
                &prepared.provided,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(outputs, successors, non_finite_policy, false)?;
                    validate_staged_clip_report(outputs, 0, clip_report, non_finite_policy)?;
                    validate_staged_window_loss_report(
                        outputs,
                        window_loss_report_start,
                        window_loss_report,
                        non_finite_policy,
                    )
                },
            )
            .map_err(replay_error)?;
        debug_assert_eq!(
            replay.outputs.len(),
            usize::from(transition.clip_report) * 2
                + usize::from(transition.window_loss_report) * 2
        );
        let mut outputs = replay.outputs.into_iter();
        let clip_report = take_compiled_clip_report(&mut outputs, transition.clip_report);
        let window_loss =
            take_compiled_window_loss_value(&mut outputs, transition.window_loss_report);
        debug_assert!(outputs.next().is_none());
        #[cfg(debug_assertions)]
        {
            let mut committed = replay.committed.clone();
            committed.sort_by_key(|state| state.buffer);
            debug_assert_eq!(committed, prepared.cursor.frontier());
        }
        self.cursor = prepared.next_main_cursor;
        Ok(CompiledAdamWAuxiliaryReports {
            clip_report,
            window_loss,
        })
    }

    fn prepare_native_auxiliary_transition(
        &self,
        transition: &CompiledAdamWAuxiliaryPlan,
        executor: &CapturedReplayExecutor,
        vectorized: bool,
        external_learning_rate: bool,
    ) -> Result<PreparedNativeCpuProgram> {
        let started = Instant::now();
        let learning_rate = external_learning_rate
            .then(|| TensorData::zeros_with_dtype(Shape::from([]), DType::F32))
            .transpose()?;
        let prepared = self.prepare_auxiliary_replay(transition, learning_rate)?;
        let replay = transition
            .capture
            .prepare_recurrent_native(
                &self.runtime,
                &prepared.cursor,
                &prepared.provided,
                executor,
                vectorized,
            )
            .map_err(replay_error)?;
        let trace = replay.preparation_trace();
        let report = NativeCpuProgramPreparationReport {
            capture_identity: transition.capture_identity(),
            native_identity: trace.replay.identity,
            vectorized,
            native_item_count: trace.item_count,
            cache_hit_count: trace.cache_hit_count,
            cache_miss_count: trace.cache_miss_count,
            execution_plan: transition.recurrent_capture.execution_plan().clone(),
            wall_time: started.elapsed(),
        };
        Ok(PreparedNativeCpuProgram { report, replay })
    }

    fn replay_auxiliary_transition_native(
        &mut self,
        transition: &CompiledAdamWAuxiliaryPlan,
        learning_rate: Option<TensorData>,
        non_finite_policy: CpuNonFinitePolicy,
        native: NativeReplayContext<'_>,
        successful_invocation: u64,
        injected_failure: Option<u64>,
    ) -> Result<(CompiledAdamWAuxiliaryReports, NativeCpuRunReport)> {
        let mut prepared = self.prepare_auxiliary_replay(transition, learning_rate)?;
        let started = Instant::now();
        let clip_report = transition.clip_report;
        let window_loss_report = transition.window_loss_report;
        let window_loss_report_start = usize::from(clip_report) * 2;
        let replay = transition
            .capture
            .replay_recurrent_native_checked(
                &mut self.runtime,
                &mut prepared.cursor,
                &prepared.provided,
                native,
                injected_failure,
                |outputs, successors| {
                    validate_staged_transition(
                        outputs,
                        successors.iter().copied(),
                        non_finite_policy,
                        false,
                    )?;
                    validate_staged_clip_report(outputs, 0, clip_report, non_finite_policy)?;
                    validate_staged_window_loss_report(
                        outputs,
                        window_loss_report_start,
                        window_loss_report,
                        non_finite_policy,
                    )
                },
            )
            .map_err(replay_error)?;
        let traffic = replay.traffic;
        let replay = replay.replay;
        debug_assert_eq!(
            replay.outputs.len(),
            usize::from(transition.clip_report) * 2
                + usize::from(transition.window_loss_report) * 2
        );
        let mut outputs = replay.outputs.into_iter();
        let clip_report = take_compiled_clip_report(&mut outputs, transition.clip_report);
        let window_loss =
            take_compiled_window_loss_value(&mut outputs, transition.window_loss_report);
        debug_assert!(outputs.next().is_none());
        let native = replay
            .native_trace
            .as_ref()
            .expect("strict-native recurrent replay returns a native trace");
        let report = native_cpu_run_report(
            transition.capture_identity(),
            native,
            traffic,
            successful_invocation,
            started.elapsed(),
        );
        #[cfg(debug_assertions)]
        {
            let mut committed = replay.committed.clone();
            committed.sort_by_key(|state| state.buffer);
            debug_assert_eq!(committed, prepared.cursor.frontier());
        }
        self.cursor = prepared.next_main_cursor;
        Ok((
            CompiledAdamWAuxiliaryReports {
                clip_report,
                window_loss,
            },
            report,
        ))
    }

    fn snapshots(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, TensorData>> {
        buffers
            .iter()
            .map(|(name, buffer)| {
                let state = self.current_state(*buffer)?;
                let value = self
                    .runtime
                    .snapshot(state)
                    .map_err(runtime_error)?
                    .tensor()
                    .clone();
                Ok((name.clone(), value))
            })
            .collect()
    }

    fn versions(&self, buffers: &BTreeMap<String, u64>) -> Result<BTreeMap<String, u64>> {
        buffers
            .iter()
            .map(|(name, buffer)| Ok((name.clone(), self.current_state(*buffer)?.version)))
            .collect()
    }

    fn current_state(&self, buffer: u64) -> Result<&BufferState> {
        self.cursor
            .frontier()
            .iter()
            .find(|state| state.buffer == buffer)
            .ok_or_else(|| training("compiled persistent state is absent"))
    }
}

impl CpuCompiledMomentumSgd {
    pub fn compile<F>(
        config: CompiledMomentumSgdConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let plan = CompiledTrainingPlan::compile(MomentumProgram { config }, parameters, build)?;
        Ok(Self {
            inner: plan.prepare_cpu()?,
        })
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner.step(inputs, learning_rate)
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn momentum_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.momentum_snapshots()
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn momentum_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.momentum_versions()
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledMomentumSgdStepResult> {
        self.inner
            .step_inner(inputs, learning_rate, injected_failure)
    }
}

impl CompiledTrainingRuntime for CpuCompiledMomentumSgd {
    type Step = CompiledMomentumSgdStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledMomentumSgd::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        CpuCompiledMomentumSgd::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        CpuCompiledMomentumSgd::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledMomentumSgd::parameter_snapshots(self)
    }
}

impl CompiledAdamWPlan {
    pub fn compile<F>(
        config: CompiledAdamWConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        if !config.frozen_parameters.is_empty() {
            return Err(training(
                "compiled AdamW raw parameters cannot resolve frozen parameter names",
            ));
        }
        Self::compile_parameters(config, parameters, build)
    }

    fn compile_parameters<F>(
        config: CompiledAdamWConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        reject_token_weighted_scalar_loss(&config)?;
        Self::compile_parameters_with_lowered_loss(config, parameters, build)
    }

    fn compile_parameters_with_lowered_loss<F>(
        config: CompiledAdamWConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        validate_weight_decay_exclusion_names(
            &config,
            parameters.iter().map(TrainingParameterInit::name),
        )?;
        let gradient_accumulation_steps = config.gradient_accumulation_steps;
        let token_weight_mask_input = config.token_weight_mask_input.clone();
        let max_gradient_norm = config.max_gradient_norm;
        let clip_report = config.clip_report;
        let window_loss_report = config.window_loss_report;
        let loss_scale = config.loss_scale;
        let host_token_inputs = config.host_token_inputs.clone();
        let frozen_parameters = config.frozen_parameters.clone();
        let learning_rate = config.learning_rate.clone();
        let inner = CompiledTrainingPlan::compile(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            build,
        )?;
        let partial_flush = (gradient_accumulation_steps > 1)
            .then(|| CompiledAdamWAuxiliaryPlan::compile_partial_flush(&inner, &config))
            .transpose()?;
        let zero_grad = (gradient_accumulation_steps > 1)
            .then(|| CompiledAdamWAuxiliaryPlan::compile_zero_grad(&inner))
            .transpose()?;
        let program_identity = inner.capture_identity()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            gradient_accumulation_steps,
            token_weight_mask_input,
            max_gradient_norm,
            clip_report,
            window_loss_report,
            loss_scale,
            progress: AdamWProgress::INITIAL,
            dropout: None,
            host_token_inputs,
            frozen_parameters,
            evaluation: None,
            learning_rate,
        })
    }

    /// Compiles an ordinary module forward without preparing a runtime.
    pub fn compile_module<M, F>(config: CompiledAdamWConfig, module: &M, build: F) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan =
            Self::compile_parameters(config, parameters, |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| build(module, graph, inputs),
                )
            })?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    /// Compiles a module through one explicit scalar-or-token-mean objective
    /// facade without preparing a runtime.
    ///
    /// The objective must agree with the configuration: ordinary configs
    /// accept [`CompiledAdamWObjective::Scalar`], while token-weighted configs
    /// accept [`CompiledAdamWObjective::TokenMean`]. The selected objective is
    /// lowered through the same capture path as the compatibility constructors.
    pub fn compile_module_graph<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        Self::compile_module_graph_parameters(config, module, parameter_plan, build)
    }

    fn compile_module_graph_parameters<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut plan = Self::compile_parameters_with_lowered_loss(
            config,
            parameters,
            |graph, inputs, parameters| {
                parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| {
                        let built = build(module, graph, inputs)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective(
                            &objective_config,
                            graph,
                            inputs,
                            objective,
                        )?;
                        Ok((loss, outputs))
                    },
                )
            },
        )?;
        plan.inner.frozen_parameter_nodes = frozen_parameter_nodes;
        Ok(plan)
    }

    /// Compiles module-bound AdamW with one device-resident Threefry block
    /// counter shared by the module's explicit residual-dropout calls.
    pub fn compile_module_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        reject_token_weighted_scalar_loss(&config)?;
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            build,
        )
    }

    /// Compiles a module with recurrent dropout through the unified explicit
    /// scalar-or-token-mean objective facade.
    pub fn compile_module_graph_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        let objective_config = config.clone();
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let built = build(module, graph, inputs, dropout)?;
                let (objective, outputs) = built.into_parts();
                let loss =
                    lower_compiled_adamw_objective(&objective_config, graph, inputs, objective)?;
                Ok((loss, outputs))
            },
        )
    }

    /// Compiles module-bound AdamW and derives its scalar differentiation root
    /// from fixed-shape per-token F32 losses and the configured token mask.
    ///
    /// The returned loss node must have exactly the mask input's descriptor.
    /// Capture owns `sum(mask * losses) / sum(mask)` as both the public loss and
    /// differentiation root, while the existing replay guard rejects an empty
    /// or malformed mask before recurrent state can advance.
    pub fn compile_token_mean_module_with_dropout<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let (mask_input, mask_shape) = token_mean_loss_descriptor(&config)?;
        let parameter_plan = ModuleParameterPlan::new(module, &config.frozen_parameters)?;
        let parameters = parameter_plan.initial_parameters()?;
        Self::compile_module_with_dropout_parameters(
            config,
            dropout,
            module,
            parameter_plan,
            parameters,
            |module, graph, inputs, dropout| {
                let (losses, outputs) = build(module, graph, inputs, dropout)?;
                let loss =
                    lower_token_mean_loss(graph, losses, inputs[mask_input.as_str()], &mask_shape)?;
                Ok((loss, outputs))
            },
        )
    }

    fn compile_module_with_dropout_parameters<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        parameter_plan: ModuleParameterPlan,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        parameter_plan.validate_weight_decay_exclusions(&config)?;
        let gradient_accumulation_steps = config.gradient_accumulation_steps;
        let token_weight_mask_input = config.token_weight_mask_input.clone();
        let max_gradient_norm = config.max_gradient_norm;
        let clip_report = config.clip_report;
        let window_loss_report = config.window_loss_report;
        let loss_scale = config.loss_scale;
        let host_token_inputs = config.host_token_inputs.clone();
        let frozen_parameters = config.frozen_parameters.clone();
        let learning_rate = config.learning_rate.clone();
        let workload = StateSpec::dropout_counter()?;
        let mut dropout_state = None;
        let mut frozen_parameter_nodes = BTreeSet::new();
        let mut inner = CompiledTrainingPlan::compile_with_workload(
            AdamWProgram {
                config: config.clone(),
            },
            parameters,
            Some(workload),
            |graph, inputs, parameters, counter| {
                let counter =
                    counter.ok_or_else(|| training("compiled dropout state is absent"))?;
                let mut provider = CompiledDropoutStream::new(counter, dropout);
                let (loss, outputs) = parameter_plan.lower_with_frozen_parameter_nodes(
                    graph,
                    parameters,
                    &mut frozen_parameter_nodes,
                    |graph| build(module, graph, inputs, &mut provider),
                )?;
                let (successor, state) = provider.finish(graph)?;
                dropout_state = Some(state);
                Ok((loss, outputs, Some(successor)))
            },
        )?;
        inner.frozen_parameter_nodes = frozen_parameter_nodes;
        let dropout = dropout_state
            .ok_or_else(|| training("compiled dropout configuration produced no state"))?;
        let partial_flush = (gradient_accumulation_steps > 1)
            .then(|| CompiledAdamWAuxiliaryPlan::compile_partial_flush(&inner, &config))
            .transpose()?;
        let zero_grad = (gradient_accumulation_steps > 1)
            .then(|| CompiledAdamWAuxiliaryPlan::compile_zero_grad(&inner))
            .transpose()?;
        let program_identity = inner.capture_identity()?;
        Ok(Self {
            inner,
            partial_flush,
            zero_grad,
            program_identity,
            gradient_accumulation_steps,
            token_weight_mask_input,
            max_gradient_norm,
            clip_report,
            window_loss_report,
            loss_scale,
            progress: AdamWProgress::INITIAL,
            dropout: Some(dropout),
            host_token_inputs,
            frozen_parameters,
            evaluation: None,
            learning_rate,
        })
    }

    /// Returns an independent plan whose recurrent frontier is restored from
    /// one checkpoint without rebuilding the Graph, derivatives, schedules,
    /// captures, partial-flush transition, or attached evaluation program.
    ///
    /// The checkpoint must authenticate this exact compiled program and its
    /// accumulation, dropout, frozen-parameter, input, clipping, loss-scaling,
    /// and partial-flush policies. The source plan remains unchanged on both
    /// success and failure, and each returned plan may be prepared or restored
    /// independently.
    pub fn restore_checkpoint(&self, checkpoint: &CompiledAdamWCheckpoint) -> Result<Self> {
        let decoded = decode_adamw_checkpoint(checkpoint.as_bytes())?;
        if self.gradient_accumulation_steps != decoded.accumulation_steps {
            return Err(training(
                "compiled AdamW checkpoint accumulation policy mismatch",
            ));
        }
        if self.window_loss_report != decoded.window_loss_report {
            return Err(training(
                "compiled AdamW checkpoint window-loss reporting policy mismatch",
            ));
        }
        match (
            self.token_weight_mask_input.as_ref(),
            decoded.accumulated_token_count,
        ) {
            (Some(mask_input), Some(count)) => validate_retained_token_count(
                &self.inner.inputs,
                mask_input,
                decoded.accumulation_index,
                count,
            )?,
            (None, None) => {}
            _ => {
                return Err(training(
                    "compiled AdamW checkpoint token-weighting policy mismatch",
                ));
            }
        }
        if self.capture_identity() != decoded.capture_identity {
            return Err(training(
                "compiled AdamW checkpoint capture identity mismatch",
            ));
        }
        if decoded.flush_capture_identity.is_some()
            && self.flush_capture_identity() != decoded.flush_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint partial flush capture identity mismatch",
            ));
        }
        if decoded.reset_capture_identity.is_some()
            && self.zero_grad_capture_identity() != decoded.reset_capture_identity
        {
            return Err(training(
                "compiled AdamW checkpoint zero-grad capture identity mismatch",
            ));
        }
        match (self.dropout, decoded.dropout_block_counter) {
            (None, None) => {}
            (None, Some(_)) => {
                return Err(training(
                    "compiled AdamW dropout checkpoint requires dropout restore",
                ));
            }
            (Some(_), None) => {
                return Err(training(
                    "compiled AdamW checkpoint has no dropout block counter",
                ));
            }
            (Some(dropout), Some(counter)) => {
                let expected = decoded
                    .replay_step
                    .checked_mul(dropout.blocks_per_replay)
                    .ok_or_else(|| training("compiled dropout counter progress overflows"))?;
                if counter != expected {
                    return Err(training(
                        "compiled dropout counter and replay progress diverged",
                    ));
                }
            }
        }

        let mut values = decoded
            .parameters
            .into_iter()
            .map(|(name, value)| (RecurrentStateKey::parameter(name), value))
            .collect::<BTreeMap<_, _>>();
        for (name, value) in decoded.first_moments {
            values.insert(
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::FirstMoment),
                value,
            );
        }
        for (name, value) in decoded.second_moments {
            values.insert(
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::SecondMoment),
                value,
            );
        }
        for (name, value) in decoded.gradient_accumulators {
            values.insert(
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator),
                value,
            );
        }
        values.insert(
            RecurrentStateKey::adamw_global(AdamWGlobalState::Step),
            TensorData::from_scalars(
                Shape::from([]),
                DType::U64,
                [Scalar::U(decoded.optimizer_step)],
            )?,
        );
        if decoded.accumulation_steps > 1 {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex),
                TensorData::from_scalars(
                    Shape::from([]),
                    DType::U64,
                    [Scalar::U(decoded.accumulation_index)],
                )?,
            );
        }
        if let Some(count) = decoded.accumulated_token_count {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedTokenCount),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(count)])?,
            );
        }
        if let Some(numerator) = decoded.accumulated_loss_numerator {
            values.insert(
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator),
                numerator,
            );
        }
        if let Some(counter) = decoded.dropout_block_counter {
            values.insert(
                RecurrentStateKey::dropout_counter(),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(counter)])?,
            );
        }

        let progress = AdamWProgress {
            replay_step: decoded.replay_step,
            optimizer_step: decoded.optimizer_step,
            accumulation_index: decoded.accumulation_index,
            discarded_microbatches: decoded.discarded_microbatches,
            flushed_window_count: decoded.flushed_window_count,
            flushed_microbatch_count: decoded.flushed_microbatch_count,
            reset_transition_count: decoded.reset_transition_count,
        };
        let optimizer_version = decoded
            .replay_step
            .checked_add(decoded.flushed_window_count)
            .ok_or_else(|| training("compiled AdamW checkpoint state version overflows"))?;
        let reset_version = optimizer_version
            .checked_add(decoded.reset_transition_count)
            .ok_or_else(|| training("compiled AdamW checkpoint reset state version overflows"))?;
        let versions = values
            .keys()
            .cloned()
            .map(|key| {
                let version = if self.inner.workload_buffers.contains_key(&key) {
                    decoded.replay_step
                } else if key.is_accumulation_reset_state() {
                    reset_version
                } else {
                    optimizer_version
                };
                (key, version)
            })
            .collect();

        let mut restored = self.clone();
        restored.inner =
            restored
                .inner
                .restore_frontier_with_versions(decoded.replay_step, values, versions)?;
        restored.partial_flush = restored
            .partial_flush
            .take()
            .map(|transition| transition.with_frontier(&restored.inner.state_values))
            .transpose()?;
        restored.zero_grad = restored
            .zero_grad
            .take()
            .map(|transition| transition.with_frontier(&restored.inner.state_values))
            .transpose()?;
        restored.progress = progress;
        Ok(restored)
    }

    /// Compatibility constructor that compiles an exact program and then
    /// restores its portable AdamW frontier before runtime preparation.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        if !config.frozen_parameters.is_empty() {
            return Err(training(
                "compiled AdamW raw parameters cannot resolve frozen parameter names",
            ));
        }
        let decoded = decode_adamw_checkpoint(checkpoint.as_bytes())?;
        let parameters = decoded
            .parameters
            .iter()
            .map(|(name, value)| TrainingParameterInit::new(name.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()?;
        Self::compile(config, parameters, build)?.restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor that compiles a module-bound program and then
    /// restores its portable frontier without preparing a runtime.
    pub fn compile_module_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_module(config, module, build)?.restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor that compiles the explicit residual-dropout
    /// program and then restores its optimizer and Threefry-counter frontier.
    pub fn compile_module_with_dropout_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_module_with_dropout(config, dropout, module, build)?
            .restore_checkpoint(checkpoint)
    }

    /// Compatibility constructor for a compiler-owned token-mean loss that
    /// restores its portable optimizer and Threefry-counter frontier.
    pub fn compile_token_mean_module_with_dropout_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_token_mean_module_with_dropout(config, dropout, module, build)?
            .restore_checkpoint(checkpoint)
    }

    /// Prepares graph-free CPU replay from this plan's exact frontier.
    pub fn prepare_cpu(&self) -> Result<CpuCompiledAdamW> {
        self.prepare_cpu_with_non_finite_policy(CpuNonFinitePolicy::Propagate)
    }

    fn prepare_cpu_with_non_finite_policy(
        &self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> Result<CpuCompiledAdamW> {
        Ok(CpuCompiledAdamW {
            inner: self
                .inner
                .prepare_cpu_with_non_finite_policy(non_finite_policy)?,
            partial_flush: self.partial_flush.clone(),
            zero_grad: self.zero_grad.clone(),
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            token_weight_mask_input: self.token_weight_mask_input.clone(),
            max_gradient_norm: self.max_gradient_norm,
            clip_report: self.clip_report,
            window_loss_report: self.window_loss_report,
            loss_scale: self.loss_scale,
            progress: self.progress,
            dropout: self.dropout,
            host_token_inputs: self.host_token_inputs.clone(),
            frozen_parameters: self.frozen_parameters.clone(),
            evaluation: self
                .evaluation
                .clone()
                .map(|plan| CpuCompiledEvaluation { plan }),
            learning_rate: self.learning_rate.clone(),
            non_finite_policy,
        })
    }

    /// Prepares strict-native CPU replay and compiles every attached pure
    /// program before exposing mutable session state.
    pub fn prepare_native_cpu<'a>(
        &self,
        target: &NativeCpuSessionTarget<'a>,
    ) -> Result<NativeCpuCompiledAdamW<'a>> {
        let inner = self.prepare_cpu_with_non_finite_policy(target.non_finite_policy())?;
        NativeCpuCompiledAdamW::prepare(inner, target.executor(), target.is_vectorized())
    }

    /// Prepares this authenticated plan through a concrete session target.
    ///
    /// The target's associated session and error keep backend-specific
    /// diagnostics statically available without a runtime backend enum or CPU
    /// fallback.
    pub fn prepare<'a, T>(
        &'a self,
        target: &T,
    ) -> std::result::Result<
        <T as SessionTarget<&'a Self>>::Session,
        <T as SessionTarget<&'a Self>>::Error,
    >
    where
        T: SessionTarget<&'a Self>,
    {
        target.prepare(self)
    }

    /// Renders the compiled program for strict Metal admission without
    /// creating device resources.
    pub fn metal_plan(&self, renderer: MetalRenderer) -> Result<MetalCompiledAdamWPlan> {
        if self.clip_report {
            return Err(training(
                "compiled AdamW clip reporting is currently CPU-only",
            ));
        }
        if self.window_loss_report {
            return Err(training(
                "compiled AdamW window-loss reporting is currently CPU-only",
            ));
        }
        if self.token_weight_mask_input.is_some() {
            return Err(training(
                "compiled AdamW token-weighted accumulation is currently CPU-only",
            ));
        }
        if matches!(
            &self.learning_rate,
            CompiledLearningRatePolicy::MultiStep(_)
        ) {
            return Err(training(
                "compiled MultiStep learning-rate policy is currently CPU-only",
            ));
        }
        let inner = self.inner.metal_plan(
            renderer.clone(),
            &self.host_token_inputs,
            self.evaluation.clone(),
        )?;
        let partial_flush = self
            .partial_flush
            .as_ref()
            .map(|transition| {
                MetalFixedStateTransitionPlan::new(
                    transition.recurrent_capture.clone(),
                    renderer,
                    &inner.inner,
                )
                .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledAdamWPlan {
            inner,
            partial_flush,
            progress: self.progress,
            flush_capture_identity: self.flush_capture_identity(),
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            max_gradient_norm: self.max_gradient_norm,
            loss_scale: self.loss_scale,
            dropout: self.dropout,
            frozen_parameters: self.frozen_parameters.clone(),
        })
    }

    pub fn capture_identity(&self) -> u64 {
        self.program_identity
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }

    pub fn token_weighted_gradient_accumulation_mask(&self) -> Option<&str> {
        self.token_weight_mask_input.as_deref()
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.max_gradient_norm
    }

    pub fn clip_report_enabled(&self) -> bool {
        self.clip_report
    }

    /// Whether completed-window loss aggregation is captured and reported.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.window_loss_report
    }

    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        match &self.learning_rate {
            CompiledLearningRatePolicy::External => None,
            CompiledLearningRatePolicy::MultiStep(schedule) => Some(schedule),
        }
    }

    /// Returns immutable logical work and recurrent-state facts without
    /// preparing a runtime or exposing the raw mixed capture.
    pub fn inspection(&self) -> Result<CompiledAdamWInspection> {
        let recurrent_state = checked_recurrent_state_extent(
            self.inner
                .state_values
                .values()
                .map(checked_bytes)
                .collect::<Result<Vec<_>>>()?,
        )?;
        let main = (
            self.capture_identity(),
            self.inner.recurrent_capture.execution_plan().clone(),
        );
        let partial_flush = self.partial_flush.as_ref().map(|transition| {
            (
                transition.capture_identity(),
                transition.recurrent_capture.execution_plan().clone(),
            )
        });
        let zero_grad = self.zero_grad.as_ref().map(|transition| {
            (
                transition.capture_identity(),
                transition.recurrent_capture.execution_plan().clone(),
            )
        });
        let evaluation = self.evaluation.as_ref().map(|evaluation| {
            (
                evaluation.capture_identity,
                evaluation.inference.execution_plan().clone(),
            )
        });
        Ok(CompiledAdamWInspection::new(
            self.step_count(),
            main,
            partial_flush,
            zero_grad,
            evaluation,
            recurrent_state,
        ))
    }

    /// Returns the explicit compiled dropout policy, when present.
    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.dropout.map(|dropout| dropout.config)
    }

    /// Number of Threefry U64 blocks reserved by each successful replay.
    pub fn dropout_blocks_per_replay(&self) -> Option<u64> {
        self.dropout.map(|dropout| dropout.blocks_per_replay)
    }

    /// Stable identity of the state-only flush capture, when accumulation is
    /// enabled.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.partial_flush
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    /// Stable identity of the captured state-only accumulation reset.
    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.zero_grad
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }
}

impl<M: Module> CompiledModuleAdamWPlan<M> {
    fn build_owned<F>(
        module: M,
        frozen_parameters: &BTreeSet<String>,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M) -> Result<CompiledAdamWPlan>,
    {
        let result: Result<(CompiledAdamWPlan, CompiledModuleSeal)> = (|| {
            let seal = CompiledModuleSeal::capture(&module, frozen_parameters)?;
            let plan = build(&module)?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal))
        })();
        match result {
            Ok((plan, seal)) => Ok(Self { module, plan, seal }),
            Err(source) => Err(CompiledModuleAdamWCompileError { module, source }),
        }
    }

    fn build_owned_from_module_checkpoint<F>(
        config: &CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, ModuleParameterPlan) -> Result<CompiledAdamWPlan>,
    {
        let result: Result<(CompiledAdamWPlan, CompiledModuleSeal)> = (|| {
            let decoded = decode_module_adamw_checkpoint(checkpoint.as_bytes())?;
            let mut seal = CompiledModuleSeal::capture(&module, &config.frozen_parameters)?;
            let immutable_values = seal.apply_module_checkpoint(&decoded)?;
            let parameter_plan = ModuleParameterPlan::new(&module, &config.frozen_parameters)?
                .with_immutable_values(&immutable_values)?;
            let plan = build(&module, parameter_plan)?
                .restore_checkpoint(checkpoint.optimizer_checkpoint())?;
            seal.validate_unchanged(&module)?;
            Ok((plan, seal))
        })();
        match result {
            Ok((plan, seal)) => Ok(Self { module, plan, seal }),
            Err(source) => Err(CompiledModuleAdamWCompileError { module, source }),
        }
    }

    /// Compiles AdamW from, and takes ownership of, one exact module value.
    pub fn compile<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module(config, module, build)
        })
    }

    /// Compiles the explicit recurrent-dropout workload while taking ownership
    /// of its exact module value.
    pub fn compile_with_dropout<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_with_dropout(config, dropout, module, build)
        })
    }

    /// Compiles and owns a module through the unified explicit objective
    /// facade.
    pub fn compile_graph<F>(
        config: CompiledAdamWConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph(config, module, build)
        })
    }

    /// Compiles and owns a recurrent-dropout module through the unified
    /// explicit objective facade.
    pub fn compile_graph_with_dropout<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let frozen_parameters = config.frozen_parameters.clone();
        Self::build_owned(module, &frozen_parameters, |module| {
            CompiledAdamWPlan::compile_module_graph_with_dropout(config, dropout, module, build)
        })
    }

    /// Recompiles a unified owned module program from a complete module
    /// checkpoint without first mutating the destination module.
    ///
    /// Saved frozen parameters and buffers are used as capture constants.
    /// Destination topology, ties, kinds, and source trainability must match;
    /// optimizer and immutable values are published together only by finish.
    pub fn compile_graph_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                CompiledAdamWPlan::compile_module_graph_parameters(
                    objective_config,
                    module,
                    parameter_plan,
                    build,
                )
            },
        )
    }

    /// Recompiles a recurrent-dropout unified owned module program from a
    /// complete module checkpoint without mutating the destination module.
    pub fn compile_graph_with_dropout_from_module_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledModuleAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<CompiledAdamWGraph>,
    {
        let objective_config = config.clone();
        Self::build_owned_from_module_checkpoint(
            &config,
            module,
            checkpoint,
            move |module, parameter_plan| {
                let parameters = parameter_plan.initial_parameters()?;
                let lower_config = objective_config.clone();
                CompiledAdamWPlan::compile_module_with_dropout_parameters(
                    objective_config,
                    dropout,
                    module,
                    parameter_plan,
                    parameters,
                    move |module, graph, inputs, dropout| {
                        let built = build(module, graph, inputs, dropout)?;
                        let (objective, outputs) = built.into_parts();
                        let loss = lower_compiled_adamw_objective(
                            &lower_config,
                            graph,
                            inputs,
                            objective,
                        )?;
                        Ok((loss, outputs))
                    },
                )
            },
        )
    }

    /// Compatibility constructor that compiles an owned module program and
    /// then restores its authenticated AdamW frontier before preparation.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        module: M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile(config, module, build).and_then(|plan| {
            plan.restore_checkpoint(checkpoint).map_err(|error| {
                let (plan, source) = error.into_parts();
                CompiledModuleAdamWCompileError {
                    module: plan.module,
                    source,
                }
            })
        })
    }

    /// Compatibility constructor that compiles the owned recurrent-dropout
    /// workload and then restores its complete optimizer/dropout frontier.
    pub fn compile_with_dropout_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        dropout: CompiledDropoutConfig,
        module: M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWCompileError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &mut dyn TrainingDropoutProvider,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_with_dropout(config, dropout, module, build).and_then(|plan| {
            plan.restore_checkpoint(checkpoint).map_err(|error| {
                let (plan, source) = error.into_parts();
                CompiledModuleAdamWCompileError {
                    module: plan.module,
                    source,
                }
            })
        })
    }

    /// Restores a checkpoint onto this already compiled owned program without
    /// rebuilding the Graph, derivatives, schedules, captures, partial-flush
    /// transition, or attached evaluation program.
    ///
    /// The returned owner contains an independent restored plan while keeping
    /// the same sealed module value. Failure retains this complete owner for
    /// inspection, retry, or preparation of its unchanged frontier.
    pub fn restore_checkpoint(
        mut self,
        checkpoint: &CompiledAdamWCheckpoint,
    ) -> std::result::Result<Self, CompiledModuleAdamWRestoreError<M>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleAdamWRestoreError {
                plan: Box::new(self),
                source,
            });
        }
        match self.plan.restore_checkpoint(checkpoint) {
            Ok(plan) => {
                self.plan = plan;
                Ok(self)
            }
            Err(source) => Err(CompiledModuleAdamWRestoreError {
                plan: Box::new(self),
                source,
            }),
        }
    }

    /// Attaches one read-only evaluation capture to this exact owned plan.
    /// The evaluator reuses the training input schema and live canonical
    /// trainable frontier; frozen parameters and buffers remain capture-owned
    /// constants. Failure retains the unconsumed plan for retry or recovery.
    pub fn with_evaluation<F>(
        mut self,
        build: F,
    ) -> std::result::Result<Self, CompiledModuleAdamWEvaluationError<M>>
    where
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        let result = (|| {
            if self.plan.evaluation.is_some() {
                return Err(training("compiled evaluation is already attached"));
            }
            self.seal.validate_unchanged(&self.module)?;
            let parameter_plan = self.seal.parameter_plan(&self.module)?;
            let evaluation = CompiledEvaluationPlan::compile_with_parameter_plan(
                &self.module,
                &self.plan,
                parameter_plan,
                build,
            )?;
            self.seal.validate_unchanged(&self.module)?;
            Ok(evaluation)
        })();
        match result {
            Ok(evaluation) => {
                self.plan.evaluation = Some(evaluation);
                Ok(self)
            }
            Err(source) => Err(CompiledModuleAdamWEvaluationError {
                plan: Box::new(self),
                source,
            }),
        }
    }

    /// Consumes this owner into a target-specific session. A preparation error
    /// retains the complete plan and module for inspection or retry.
    pub fn prepare<T>(
        self,
        target: &T,
    ) -> std::result::Result<<T as SessionTarget<Self>>::Session, <T as SessionTarget<Self>>::Error>
    where
        T: SessionTarget<Self>,
    {
        target.prepare(self)
    }

    pub fn capture_identity(&self) -> u64 {
        self.plan.capture_identity()
    }

    /// Returns the owned plan's immutable logical work and recurrent-state
    /// inspection without exposing its sealed module.
    pub fn inspection(&self) -> Result<CompiledAdamWInspection> {
        self.plan.inspection()
    }

    pub fn step_count(&self) -> u64 {
        self.plan.step_count()
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.plan.flush_capture_identity()
    }

    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.plan.dropout_config()
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.plan.captured_multi_step_lr()
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.plan
            .evaluation
            .as_ref()
            .map(|evaluation| evaluation.capture_identity)
    }

    /// Inspects strict Metal admission without exposing an independently
    /// preparable runtime path or releasing the owned module. Resource
    /// preparation still consumes this owner through [`Self::prepare`].
    pub fn metal_summary(&self, renderer: MetalRenderer) -> Result<MetalDeviceSessionSummary> {
        Ok(self.plan.metal_plan(renderer)?.summary().clone())
    }
}

impl<M, R> CompiledModuleAdamWSession<M, R> {
    /// Discards the compiled runtime frontier and returns the sealed host
    /// module without publishing any trained parameter values.
    pub fn into_module_without_publication(self) -> M {
        self.module
    }
}

impl<M: Module, R: CompiledScheduledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.runtime.captured_multi_step_lr()
    }

    pub fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<R::Step> {
        self.runtime.step_scheduled(inputs)
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<R::Step>
    where
        B: CompiledInputBatch,
    {
        self.runtime.step_batch_scheduled(batch)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<R::ScheduledFlush> {
        self.runtime.flush_partial_window_scheduled()
    }
}

impl<M: Module, R: CompiledTrainingRuntime> CompiledModuleAdamWSession<M, R> {
    /// Atomically publishes the runtime's exact trainable frontier and returns
    /// the owned module. The complete module topology, identities, versions,
    /// descriptors, and frozen/buffer bytes must still match the compile seal.
    /// A failure retains the intact session and can be recovered with
    /// [`CompiledModuleAdamWFinishError::into_session`].
    pub fn finish(self) -> std::result::Result<M, CompiledModuleAdamWFinishError<M, R>> {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let parameters = match self.runtime.parameter_snapshots() {
            Ok(parameters) => parameters,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { module, .. } = self;
        Ok(module)
    }
}

impl<M: Module, R: CompiledAdamWRuntime> CompiledModuleAdamWSession<M, R> {
    /// Snapshots the exact optimizer frontier together with the owned
    /// module's canonical immutable state and topology.
    ///
    /// This does not publish into or release the sealed host module. The
    /// embedded optimizer checkpoint is reused byte-for-byte across v1--v8;
    /// report-disabled programs retain their existing v1--v7 bytes.
    pub fn module_checkpoint(&self) -> Result<CompiledModuleAdamWCheckpoint> {
        self.seal.validate_unchanged(&self.module)?;
        let optimizer = self.runtime.checkpoint()?;
        self.seal.validate_unchanged(&self.module)?;
        let (states, visits) = self.seal.checkpoint_inventory();
        encode_module_adamw_checkpoint(&optimizer, &states, &visits)
    }

    /// Atomically publishes and returns the exact checkpointed AdamW frontier.
    ///
    /// The module seal is validated before snapshot work. One coherent
    /// checkpoint snapshot supplies both the returned resumable optimizer state
    /// and the parameter values published into the owned module, so strict
    /// device runtimes do not perform a second parameter-only read. Tied and
    /// policy-frozen identities retain the same publication rules as
    /// [`Self::finish`]. A checkpoint, decode, or publication failure retains
    /// the intact session in [`CompiledModuleAdamWFinishError`].
    pub fn finish_with_checkpoint(
        self,
    ) -> std::result::Result<(M, CompiledAdamWCheckpoint), CompiledModuleAdamWFinishError<M, R>>
    {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let checkpoint = match self.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let parameters = match decode_adamw_checkpoint(checkpoint.as_bytes()) {
            Ok(decoded) => decoded.parameters,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { module, .. } = self;
        Ok((module, checkpoint))
    }

    /// Atomically publishes and returns one complete module checkpoint built
    /// from the exact AdamW snapshot used for publication.
    ///
    /// The checkpoint retains canonical module topology, ties, frozen
    /// parameters, and buffers in addition to the optimizer frontier. The
    /// runtime is checkpointed exactly once; encoding and publication both use
    /// that same snapshot. A seal, checkpoint, encoding, decode, or publication
    /// failure retains the intact session in [`CompiledModuleAdamWFinishError`]
    /// for inspection or retry.
    pub fn finish_with_module_checkpoint(
        self,
    ) -> std::result::Result<(M, CompiledModuleAdamWCheckpoint), CompiledModuleAdamWFinishError<M, R>>
    {
        if let Err(source) = self.seal.validate_unchanged(&self.module) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let optimizer = match self.runtime.checkpoint() {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let (states, visits) = self.seal.checkpoint_inventory();
        let checkpoint = match encode_module_adamw_checkpoint(&optimizer, &states, &visits) {
            Ok(checkpoint) => checkpoint,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        let parameters = match decode_adamw_checkpoint(optimizer.as_bytes()) {
            Ok(decoded) => decoded.parameters,
            Err(source) => {
                return Err(CompiledModuleAdamWFinishError {
                    session: Box::new(self),
                    source,
                });
            }
        };
        if let Err(source) = self.seal.publish(&self.module, &parameters) {
            return Err(CompiledModuleAdamWFinishError {
                session: Box::new(self),
                source,
            });
        }
        let Self { module, .. } = self;
        Ok((module, checkpoint))
    }
}

impl<M> CompiledModuleAdamWSession<M, CpuCompiledAdamW> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.runtime.dropout_block_counter()
    }
}

impl<'a, M> CompiledModuleAdamWSession<M, NativeCpuCompiledAdamW<'a>> {
    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.runtime.dropout_block_counter()
    }

    /// Returns strict-native CPU preparation evidence without exposing the
    /// sealed module or mutable runtime internals.
    pub fn native_cpu_preparation_report(&self) -> &NativeCpuCompiledAdamWPreparationReport {
        self.runtime.preparation_report()
    }
}

impl<M: Module> CompiledModuleAdamWSession<M, MetalCompiledAdamW> {
    /// Strict Metal replay that commits the complete device state frontier
    /// without downloading loss or named outputs.
    pub fn step_without_host_outputs(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWCommitResult> {
        self.runtime
            .step_without_host_outputs(inputs, learning_rate)
    }

    /// Returns the sealed runtime's read-only Metal session evidence without
    /// exposing the owned module or mutable backend internals.
    pub fn metal_session(&self) -> &MetalDeviceSession {
        self.runtime.metal_session()
    }

    /// Returns the opt-in successful-step recorder attached during target
    /// preparation, when present.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.runtime.execution_scoreboard()
    }

    /// Snapshots the owned Metal runtime's successfully recorded prefix.
    pub fn execution_scoreboard_report(
        &self,
    ) -> std::result::Result<Option<MetalSessionScoreboardReport>, MetalScoreboardError> {
        self.runtime.execution_scoreboard_report()
    }

    /// Returns the first fail-soft scoreboard recording error, when recording
    /// has frozen.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.runtime.scoreboard_recording_error()
    }

    /// Returns preparation evidence for the two read-only active-bank
    /// evaluators, when evaluation was attached before preparation.
    pub fn evaluation_preparation_reports(
        &self,
    ) -> Option<[&crate::runtime::metal::MetalDevicePreparationReport; 2]> {
        self.runtime.evaluation_preparation_reports()
    }

    /// Returns deterministic resource/execution summaries for both read-only
    /// physical-bank evaluators.
    pub fn evaluation_summaries(&self) -> Option<[&MetalDeviceSessionSummary; 2]> {
        self.runtime.evaluation_summaries()
    }
}

impl CpuCompiledAdamW {
    pub fn compile<F>(
        config: CompiledAdamWConfig,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile(config, parameters, build)?.prepare_cpu()
    }

    /// Compiles an ordinary module forward against optimizer-owned recurrent
    /// parameter state.
    ///
    /// The builder receives only declared batch inputs: calls to
    /// [`crate::nn::Parameter::bind`] inside `module` resolve automatically to
    /// the compiled state frontier. Frozen parameters and buffers are captured
    /// as immutable constants, while tied parameter handles share one graph
    /// node and one AdamW state tuple. Names selected by
    /// [`CompiledAdamWConfig::with_frozen_parameters`] receive that same
    /// constant treatment without changing their host trainable flags.
    pub fn compile_module<M, F>(config: CompiledAdamWConfig, module: &M, build: F) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile_module(config, module, build)?.prepare_cpu()
    }

    /// Recompiles an exact program and restores its saved recurrent frontier.
    /// The build/configuration must reproduce the checkpoint's capture
    /// identity; all state is validated before the fresh runtime is replaced.
    pub fn compile_from_checkpoint<F>(
        config: CompiledAdamWConfig,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile_from_checkpoint(config, checkpoint, build)?.prepare_cpu()
    }

    /// Recompiles a module-bound program and restores its exact AdamW state.
    /// The module topology, frozen values, builder, and input descriptors must
    /// reproduce the authenticated capture identity before any restored state
    /// becomes visible.
    pub fn compile_module_from_checkpoint<M, F>(
        config: CompiledAdamWConfig,
        module: &M,
        checkpoint: &CompiledAdamWCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        CompiledAdamWPlan::compile_module_from_checkpoint(config, module, checkpoint, build)?
            .prepare_cpu()
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWStepResult> {
        self.learning_rate.require_external()?;
        self.step_with_learning_rate(inputs, Some(learning_rate), None)
    }

    /// Replays one batch using the immutable MultiStep rate captured in the
    /// program. This method accepts no host learning-rate value.
    pub fn step_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledAdamWStepResult> {
        self.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, None)
    }

    fn step_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        validate_training_inputs(&self.inner.inputs, &inputs)?;
        let loss_weight =
            validate_token_weight_mask(&inputs, self.token_weight_mask_input.as_deref())?;
        let next = self
            .progress
            .advance_replay(self.gradient_accumulation_steps)?;
        if let Some(dropout) = self.dropout {
            expected_dropout_counter(dropout, next.replay_step)?;
        }
        let mut result = self.inner.step_inner_with_learning_rate(
            inputs,
            learning_rate,
            self.non_finite_policy,
            next.accumulation_index == 0,
            injected_failure,
        )?;
        result.step = next.replay_step;
        self.progress = next;
        Ok(adamw_step_result(
            result,
            next,
            loss_weight,
            self.gradient_accumulation_steps,
        ))
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<CompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<CompiledEvaluationResult> {
        let evaluation = self
            .evaluation
            .as_ref()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        evaluation
            .plan
            .evaluate(inputs, self.inner.parameter_snapshots()?)
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation
            .as_ref()
            .map(|evaluation| evaluation.plan.capture_identity)
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }

    pub fn token_weighted_gradient_accumulation_mask(&self) -> Option<&str> {
        self.token_weight_mask_input.as_deref()
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.max_gradient_norm
    }

    pub fn clip_report_enabled(&self) -> bool {
        self.clip_report
    }

    /// Whether completed-window loss aggregation is captured and reported.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.window_loss_report
    }

    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        match &self.learning_rate {
            CompiledLearningRatePolicy::External => None,
            CompiledLearningRatePolicy::MultiStep(schedule) => Some(schedule),
        }
    }

    /// CPU-only admission policy selected when this runtime was prepared.
    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.non_finite_policy
    }

    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.dropout
            .map(|_| {
                Ok(self
                    .inner
                    .workload_snapshot(&RecurrentStateKey::dropout_counter())?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        Ok(self.progress.optimizer_step)
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::SecondMoment)
    }

    /// Partial F32 gradient sums retained between microbatches. The map is
    /// empty when accumulation is disabled (`steps == 1`).
    pub fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner
            .adamw_state_snapshots(AdamWParameterState::GradientAccumulator)
    }

    /// Number of microbatches currently retained toward the next update.
    pub fn accumulation_index(&self) -> Result<u64> {
        Ok(self.progress.accumulation_index)
    }

    /// Atomically clears a retained partial accumulation window. Parameters,
    /// moments, optimizer progress, and successful replay count are preserved.
    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_inner(None)
    }

    fn zero_grad_inner(
        &mut self,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWZeroGradResult> {
        let (next, result) = self.progress.cancel(self.gradient_accumulation_steps)?;
        if !result.did_discard() {
            return Ok(result);
        }
        let next = next.record_reset_transition()?;
        let transition = self
            .zero_grad
            .as_ref()
            .ok_or_else(|| training("compiled AdamW zero-grad capture is absent"))?;
        let reports = self.inner.replay_auxiliary_transition(
            transition,
            None,
            CpuNonFinitePolicy::Propagate,
            injected_failure,
        )?;
        debug_assert!(reports.clip_report.is_none());
        debug_assert!(reports.window_loss.is_none());
        self.progress = next;
        Ok(result)
    }

    #[cfg(test)]
    fn zero_grad_with_injected_failure(
        &mut self,
        injected_failure: u64,
    ) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_inner(Some(injected_failure))
    }

    /// Atomically commits a nonempty partial accumulation window through its
    /// separately authenticated state-only capture. Replay/dropout progress
    /// and all workload state remain unchanged.
    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWFlushResult> {
        self.learning_rate.require_external()?;
        self.flush_partial_window_with_learning_rate(Some(learning_rate), None)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<CompiledAdamWFlushResult> {
        self.learning_rate.require_scheduled()?;
        self.flush_partial_window_with_learning_rate(None, None)
    }

    fn flush_partial_window_with_learning_rate(
        &mut self,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWFlushResult> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let (next, mut result) = self
            .progress
            .flush_partial(self.gradient_accumulation_steps)?;
        if !result.did_update() {
            return Ok(result);
        }
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, self.non_finite_policy)?;
        }
        let transition = self
            .partial_flush
            .as_ref()
            .ok_or_else(|| training("compiled AdamW partial flush capture is absent"))?;
        let reports = self.inner.replay_auxiliary_transition(
            transition,
            learning_rate,
            self.non_finite_policy,
            injected_failure,
        )?;
        result.clip_report = reports.clip_report;
        result.window_loss_report = reports
            .window_loss
            .map(|value| CompiledAdamWWindowLossReport::new(value, result.flushed_microbatches));
        self.progress = next;
        Ok(result)
    }

    /// Stable identity of the state-only flush capture, when accumulation is
    /// enabled.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.partial_flush
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.zero_grad
            .as_ref()
            .map(CompiledAdamWAuxiliaryPlan::capture_identity)
    }

    pub fn parameter_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner.parameter_versions()
    }

    pub fn first_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner
            .adamw_state_versions(AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_versions(&self) -> Result<BTreeMap<String, u64>> {
        self.inner
            .adamw_state_versions(AdamWParameterState::SecondMoment)
    }

    /// Renders the identical loss/backward/AdamW capture for Metal, seeded
    /// from this session's currently committed recurrent state. Planning is
    /// resource-free; unsupported kernels fail before a device is touched.
    pub fn metal_plan(&self, renderer: MetalRenderer) -> Result<MetalCompiledAdamWPlan> {
        let inner = self.inner.plan()?;
        let partial_flush = self
            .partial_flush
            .clone()
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        let zero_grad = self
            .zero_grad
            .clone()
            .map(|transition| transition.with_frontier(&inner.state_values))
            .transpose()?;
        CompiledAdamWPlan {
            inner,
            partial_flush,
            zero_grad,
            program_identity: self.capture_identity(),
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            token_weight_mask_input: self.token_weight_mask_input.clone(),
            max_gradient_norm: self.max_gradient_norm,
            clip_report: self.clip_report,
            window_loss_report: self.window_loss_report,
            loss_scale: self.loss_scale,
            progress: self.progress,
            dropout: self.dropout,
            host_token_inputs: self.host_token_inputs.clone(),
            frozen_parameters: self.frozen_parameters.clone(),
            evaluation: self
                .evaluation
                .as_ref()
                .map(|evaluation| evaluation.plan.clone()),
            learning_rate: self.learning_rate.clone(),
        }
        .metal_plan(renderer)
    }

    /// Captures parameter values, both moment sets, the graph-owned optimizer
    /// step, and the exact compiled capture identity into deterministic bytes.
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        validate_adamw_progress(self.progress, self.gradient_accumulation_steps)?;
        validate_cpu_adamw_state(&self.inner, self.progress, self.gradient_accumulation_steps)?;
        let dropout_block_counter = self
            .dropout
            .map(|dropout| {
                let counter = self
                    .inner
                    .workload_snapshot(&RecurrentStateKey::dropout_counter())?
                    .scalar_at(0)
                    .as_u64();
                if counter != expected_dropout_counter(dropout, self.progress.replay_step)? {
                    return Err(training(
                        "compiled CPU dropout counter and replay progress diverged",
                    ));
                }
                Ok(counter)
            })
            .transpose()?;
        let accumulated_token_count = self
            .token_weight_mask_input
            .as_ref()
            .map(|_| {
                Ok(self
                    .inner
                    .global_snapshot(AdamWGlobalState::AccumulatedTokenCount)?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()?;
        if let (Some(mask_input), Some(count)) = (
            self.token_weight_mask_input.as_deref(),
            accumulated_token_count,
        ) {
            validate_retained_token_count(
                &self.inner.inputs,
                mask_input,
                self.progress.accumulation_index,
                count,
            )?;
        }
        let accumulated_loss_numerator = self
            .window_loss_report
            .then(|| {
                self.inner
                    .global_snapshot(AdamWGlobalState::AccumulatedLossNumerator)
            })
            .transpose()?;
        let bytes = encode_adamw_checkpoint(
            AdamWCheckpointProgress {
                capture_identity: self.capture_identity(),
                replay_step: self.progress.replay_step,
                optimizer_step: self.progress.optimizer_step,
                accumulation_steps: self.gradient_accumulation_steps,
                accumulation_index: self.progress.accumulation_index,
                discarded_microbatches: self.progress.discarded_microbatches,
                flushed_window_count: self.progress.flushed_window_count,
                flushed_microbatch_count: self.progress.flushed_microbatch_count,
                flush_capture_identity: self.flush_capture_identity(),
                dropout_block_counter,
                accumulated_token_count,
                window_loss_report: self.window_loss_report,
                reset_transition_count: self.progress.reset_transition_count,
                reset_capture_identity: (self.progress.reset_transition_count != 0)
                    .then(|| self.zero_grad_capture_identity())
                    .flatten(),
            },
            AdamWCheckpointTensors {
                parameters: self.parameter_snapshots()?,
                first_moments: self.first_moment_snapshots()?,
                second_moments: self.second_moment_snapshots()?,
                gradient_accumulators: self.gradient_accumulator_snapshots()?,
                accumulated_loss_numerator,
            },
        )?;
        CompiledAdamWCheckpoint::from_bytes(bytes)
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWStepResult> {
        self.step_with_learning_rate(inputs, Some(learning_rate), injected_failure)
    }

    #[cfg(test)]
    fn flush_partial_window_inner(
        &mut self,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWFlushResult> {
        self.flush_partial_window_with_learning_rate(Some(learning_rate), injected_failure)
    }
}

impl<'a> NativeCpuCompiledAdamW<'a> {
    fn prepare(
        inner: CpuCompiledAdamW,
        executor: &'a CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<Self> {
        let external_learning_rate =
            matches!(&inner.learning_rate, CompiledLearningRatePolicy::External);
        let main = inner
            .inner
            .prepare_native(executor, vectorized, external_learning_rate)?;
        let partial_flush = inner
            .partial_flush
            .as_ref()
            .map(|transition| {
                inner.inner.prepare_native_auxiliary_transition(
                    transition,
                    executor,
                    vectorized,
                    external_learning_rate,
                )
            })
            .transpose()?;
        let zero_grad = inner
            .zero_grad
            .as_ref()
            .map(|transition| {
                inner
                    .inner
                    .prepare_native_auxiliary_transition(transition, executor, vectorized, false)
            })
            .transpose()?;
        let evaluation = inner
            .evaluation
            .as_ref()
            .map(|evaluation| {
                evaluation
                    .plan
                    .prepare_native(inner.parameter_snapshots()?, executor, vectorized)
            })
            .transpose()?;
        let (recurrent_state_count, recurrent_state_bytes) = checked_recurrent_state_extent(
            inner
                .inner
                .cursor
                .frontier()
                .iter()
                .map(|state| state.bytes),
        )?;
        let PreparedNativeCpuProgram {
            report: main_report,
            replay: main_replay,
        } = main;
        let (partial_flush_report, partial_flush_replay) = partial_flush
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let (zero_grad_report, zero_grad_replay) = zero_grad
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let evaluation_report = evaluation.as_ref().map(|prepared| prepared.report.clone());
        Ok(Self {
            inner,
            executor,
            main_replay,
            partial_flush_replay,
            zero_grad_replay,
            evaluation_replay: evaluation,
            preparation: NativeCpuCompiledAdamWPreparationReport {
                main: main_report,
                partial_flush: partial_flush_report,
                zero_grad: zero_grad_report,
                evaluation: evaluation_report,
                recurrent_state_count,
                recurrent_state_bytes,
            },
            successful_steps: 0,
            successful_flushes: 0,
            successful_zero_grads: 0,
            successful_evaluations: 0,
        })
    }

    pub fn preparation_report(&self) -> &NativeCpuCompiledAdamWPreparationReport {
        &self.preparation
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.learning_rate.require_external()?;
        self.step_with_learning_rate(inputs, Some(learning_rate), None)
    }

    pub fn step_scheduled(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.inner.learning_rate.require_scheduled()?;
        self.step_with_learning_rate(inputs, None, None)
    }

    fn step_with_learning_rate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        validate_training_inputs(&self.inner.inner.inputs, &inputs)?;
        let loss_weight =
            validate_token_weight_mask(&inputs, self.inner.token_weight_mask_input.as_deref())?;
        let next = self
            .inner
            .progress
            .advance_replay(self.inner.gradient_accumulation_steps)?;
        if let Some(dropout) = self.inner.dropout {
            expected_dropout_counter(dropout, next.replay_step)?;
        }
        let successful_invocation = self
            .successful_steps
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU run count overflow"))?;
        let (mut result, mut report) = self.inner.inner.step_native_inner_with_learning_rate(
            inputs,
            learning_rate,
            self.inner.non_finite_policy,
            next.accumulation_index == 0,
            NativeReplayContext::new(self.executor, &mut self.main_replay),
            injected_failure,
        )?;
        result.step = next.replay_step;
        report.successful_invocation = successful_invocation;
        self.inner.progress = next;
        self.successful_steps = successful_invocation;
        Ok(NativeCpuCompiledAdamWStepResult {
            inner: adamw_step_result(
                result,
                next,
                loss_weight,
                self.inner.gradient_accumulation_steps,
            ),
            report,
        })
    }

    pub fn step_batch_scheduled<B>(&mut self, batch: B) -> Result<NativeCpuCompiledAdamWStepResult>
    where
        B: CompiledInputBatch,
    {
        self.step_scheduled(batch.into_compiled_inputs()?)
    }

    #[cfg(test)]
    fn step_inner(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWStepResult> {
        self.step_with_learning_rate(inputs, Some(learning_rate), injected_failure)
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<NativeCpuCompiledEvaluationResult> {
        let successful_invocation = self
            .successful_evaluations
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU evaluation count overflow"))?;
        let evaluation = self
            .inner
            .evaluation
            .as_ref()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        let prepared = self
            .evaluation_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU evaluation preparation is absent"))?;
        let (inner, mut report) = evaluation.plan.evaluate_native(
            inputs,
            self.inner.inner.parameter_snapshots()?,
            self.executor,
            prepared,
        )?;
        report.successful_invocation = successful_invocation;
        self.successful_evaluations = successful_invocation;
        Ok(NativeCpuCompiledEvaluationResult { inner, report })
    }

    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.inner.learning_rate.require_external()?;
        self.flush_partial_window_impl(Some(learning_rate), None)
    }

    pub fn flush_partial_window_scheduled(&mut self) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.inner.learning_rate.require_scheduled()?;
        self.flush_partial_window_impl(None, None)
    }

    fn flush_partial_window_impl(
        &mut self,
        learning_rate: Option<TensorData>,
        injected_failure: Option<u64>,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate(learning_rate)?;
        }
        let (next, mut result) = self
            .inner
            .progress
            .flush_partial(self.inner.gradient_accumulation_steps)?;
        if !result.did_update() {
            return Ok(NativeCpuCompiledAdamWFlushResult {
                inner: result,
                report: None,
            });
        }
        if let Some(learning_rate) = &learning_rate {
            validate_learning_rate_for_policy(learning_rate, self.inner.non_finite_policy)?;
        }
        let successful_invocation = self
            .successful_flushes
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU flush count overflow"))?;
        let transition = self
            .inner
            .partial_flush
            .as_ref()
            .ok_or_else(|| training("compiled AdamW partial flush capture is absent"))?;
        let prepared = self
            .partial_flush_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU partial flush preparation is absent"))?;
        let (reports, report) = self.inner.inner.replay_auxiliary_transition_native(
            transition,
            learning_rate,
            self.inner.non_finite_policy,
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            injected_failure,
        )?;
        result.clip_report = reports.clip_report;
        result.window_loss_report = reports
            .window_loss
            .map(|value| CompiledAdamWWindowLossReport::new(value, result.flushed_microbatches));
        self.inner.progress = next;
        self.successful_flushes = successful_invocation;
        Ok(NativeCpuCompiledAdamWFlushResult {
            inner: result,
            report: Some(report),
        })
    }

    #[cfg(test)]
    fn flush_partial_window_with_injected_failure(
        &mut self,
        learning_rate: TensorData,
        injected_failure: u64,
    ) -> Result<NativeCpuCompiledAdamWFlushResult> {
        self.flush_partial_window_impl(Some(learning_rate), Some(injected_failure))
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }

    pub fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.inner.zero_grad_capture_identity()
    }

    pub fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.inner.captured_multi_step_lr()
    }

    /// CPU-only admission policy selected when this runtime was prepared.
    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.inner.non_finite_policy
    }

    /// Explicit diagnostic snapshot of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.inner.dropout_block_counter()
    }

    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        self.inner.checkpoint()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_impl(None)
    }

    fn zero_grad_impl(
        &mut self,
        injected_failure: Option<u64>,
    ) -> Result<CompiledAdamWZeroGradResult> {
        let (next, result) = self
            .inner
            .progress
            .cancel(self.inner.gradient_accumulation_steps)?;
        if !result.did_discard() {
            return Ok(result);
        }
        let next = next.record_reset_transition()?;
        let successful_invocation = self
            .successful_zero_grads
            .checked_add(1)
            .ok_or_else(|| training("compiled native CPU zero-grad count overflow"))?;
        let transition = self
            .inner
            .zero_grad
            .as_ref()
            .ok_or_else(|| training("compiled AdamW zero-grad capture is absent"))?;
        let prepared = self
            .zero_grad_replay
            .as_mut()
            .ok_or_else(|| training("compiled native CPU zero-grad preparation is absent"))?;
        let (reports, _) = self.inner.inner.replay_auxiliary_transition_native(
            transition,
            None,
            CpuNonFinitePolicy::Propagate,
            NativeReplayContext::new(self.executor, prepared),
            successful_invocation,
            injected_failure,
        )?;
        debug_assert!(reports.clip_report.is_none());
        debug_assert!(reports.window_loss.is_none());
        self.inner.progress = next;
        self.successful_zero_grads = successful_invocation;
        Ok(result)
    }

    #[cfg(test)]
    fn zero_grad_with_injected_failure(
        &mut self,
        injected_failure: u64,
    ) -> Result<CompiledAdamWZeroGradResult> {
        self.zero_grad_impl(Some(injected_failure))
    }
}

impl CompiledTrainingRuntime for CpuCompiledAdamW {
    type Step = CompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        CpuCompiledAdamW::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        CpuCompiledAdamW::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::parameter_snapshots(self)
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.parameter_snapshots()?,
            &self.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for CpuCompiledAdamW {
    type Evaluation = CompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        CpuCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::evaluation_capture_identity(self)
    }
}

impl CompiledCheckpointRuntime for CpuCompiledAdamW {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        CpuCompiledAdamW::checkpoint(self)
    }
}

impl CompiledAdamWRuntime for CpuCompiledAdamW {
    fn gradient_accumulation_steps(&self) -> u64 {
        CpuCompiledAdamW::gradient_accumulation_steps(self)
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        CpuCompiledAdamW::max_gradient_norm(self)
    }

    fn loss_scale(&self) -> f32 {
        CpuCompiledAdamW::loss_scale(self)
    }

    fn window_loss_report_enabled(&self) -> bool {
        CpuCompiledAdamW::window_loss_report_enabled(self)
    }

    fn optimizer_step(&self) -> Result<u64> {
        CpuCompiledAdamW::optimizer_step(self)
    }

    fn accumulation_index(&self) -> Result<u64> {
        CpuCompiledAdamW::accumulation_index(self)
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        CpuCompiledAdamW::zero_grad(self)
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::zero_grad_capture_identity(self)
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::first_moment_snapshots(self)
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::second_moment_snapshots(self)
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        CpuCompiledAdamW::gradient_accumulator_snapshots(self)
    }
}

impl CompiledAdamWFlushRuntime for CpuCompiledAdamW {
    type Flush = CompiledAdamWFlushResult;

    fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<CompiledAdamWFlushResult> {
        CpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        CpuCompiledAdamW::flush_capture_identity(self)
    }
}

impl CompiledScheduledAdamWRuntime for CpuCompiledAdamW {
    type ScheduledFlush = CompiledAdamWFlushResult;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        CpuCompiledAdamW::captured_multi_step_lr(self)
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        CpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        CpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CompiledTrainingRuntime for NativeCpuCompiledAdamW<'_> {
    type Step = NativeCpuCompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.inner.parameter_snapshots()?,
            &self.inner.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for NativeCpuCompiledAdamW<'_> {
    type Evaluation = NativeCpuCompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        NativeCpuCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }
}

impl CompiledCheckpointRuntime for NativeCpuCompiledAdamW<'_> {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.inner.checkpoint()
    }
}

impl CompiledAdamWRuntime for NativeCpuCompiledAdamW<'_> {
    fn gradient_accumulation_steps(&self) -> u64 {
        self.inner.gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.inner.max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.inner.loss_scale()
    }

    fn window_loss_report_enabled(&self) -> bool {
        self.inner.window_loss_report_enabled()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.inner.optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.inner.accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.inner.zero_grad()
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.inner.zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.gradient_accumulator_snapshots()
    }
}

impl CompiledAdamWFlushRuntime for NativeCpuCompiledAdamW<'_> {
    type Flush = NativeCpuCompiledAdamWFlushResult;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        NativeCpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        self.inner.flush_capture_identity()
    }
}

impl CompiledScheduledAdamWRuntime for NativeCpuCompiledAdamW<'_> {
    type ScheduledFlush = NativeCpuCompiledAdamWFlushResult;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.inner.captured_multi_step_lr()
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        NativeCpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl<M, R> CompiledTrainingRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledTrainingRuntime,
{
    type Step = R::Step;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        self.runtime.step(inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        self.runtime.step_count()
    }

    fn capture_identity(&self) -> u64 {
        self.runtime.capture_identity()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime.parameter_snapshots()
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        self.runtime.publish_parameters(module)
    }
}

impl<M, R> CompiledCheckpointRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledCheckpointRuntime,
{
    type Checkpoint = R::Checkpoint;

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.runtime.checkpoint()
    }
}

impl<M, R> CompiledEvaluationRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledEvaluationRuntime,
{
    type Evaluation = R::Evaluation;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        self.runtime.evaluate(inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.runtime.evaluation_capture_identity()
    }
}

impl<M, R> CompiledAdamWRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWRuntime,
{
    fn gradient_accumulation_steps(&self) -> u64 {
        self.runtime.gradient_accumulation_steps()
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        self.runtime.max_gradient_norm()
    }

    fn loss_scale(&self) -> f32 {
        self.runtime.loss_scale()
    }

    fn window_loss_report_enabled(&self) -> bool {
        self.runtime.window_loss_report_enabled()
    }

    fn optimizer_step(&self) -> Result<u64> {
        self.runtime.optimizer_step()
    }

    fn accumulation_index(&self) -> Result<u64> {
        self.runtime.accumulation_index()
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        self.runtime.zero_grad()
    }

    fn zero_grad_capture_identity(&self) -> Option<u64> {
        self.runtime.zero_grad_capture_identity()
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime.first_moment_snapshots()
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime.second_moment_snapshots()
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.runtime.gradient_accumulator_snapshots()
    }
}

impl<M, R> CompiledAdamWFlushRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledAdamWFlushRuntime,
{
    type Flush = R::Flush;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        self.runtime.flush_partial_window(learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        self.runtime.flush_capture_identity()
    }
}

impl<M, R> CompiledScheduledAdamWRuntime for CompiledModuleAdamWSession<M, R>
where
    R: CompiledScheduledAdamWRuntime,
{
    type ScheduledFlush = R::ScheduledFlush;

    fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        self.runtime.captured_multi_step_lr()
    }

    fn step_scheduled(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        self.runtime.step_scheduled(inputs)
    }

    fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
        self.runtime.flush_partial_window_scheduled()
    }
}

fn adamw_step_result(
    mut inner: CompiledTrainingStepResult,
    progress: AdamWProgress,
    loss_weight: u64,
    gradient_accumulation_steps: u64,
) -> CompiledAdamWStepResult {
    if progress.accumulation_index != 0 {
        inner.clip_report = None;
        inner.window_loss = None;
    }
    let window_loss_report = inner
        .window_loss
        .map(|value| CompiledAdamWWindowLossReport::new(value, gradient_accumulation_steps));
    CompiledAdamWStepResult {
        inner,
        optimizer_step: progress.optimizer_step,
        accumulation_index: progress.accumulation_index,
        loss_weight,
        window_loss_report,
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for CpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu()
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for ConfiguredCpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu_with_non_finite_policy(self.non_finite_policy())
    }
}

impl<'executor> SessionTarget<&CompiledAdamWPlan> for NativeCpuSessionTarget<'executor> {
    type Session = NativeCpuCompiledAdamW<'executor>;
    type Error = Error;

    fn prepare(&self, plan: &CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_native_cpu(self)
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for MetalSessionTarget {
    type Session = MetalCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        let rendered = plan.metal_plan(self.renderer().clone())?;
        match self.scoreboard_context() {
            Some(context) => {
                rendered.prepare_with_scoreboard(self.device().clone(), context.clone())
            }
            None => rendered.prepare(self.device().clone()),
        }
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for CpuSessionTarget {
    type Session = CompiledModuleAdamWSession<M, CpuCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.seal.validate_unchanged(&plan.module) {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let runtime = match plan.plan.prepare_cpu() {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            module,
            runtime,
            seal,
        })
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for ConfiguredCpuSessionTarget {
    type Session = CompiledModuleAdamWSession<M, CpuCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.seal.validate_unchanged(&plan.module) {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let runtime = match plan
            .plan
            .prepare_cpu_with_non_finite_policy(self.non_finite_policy())
        {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            module,
            runtime,
            seal,
        })
    }
}

impl<'executor, M: Module> SessionTarget<CompiledModuleAdamWPlan<M>>
    for NativeCpuSessionTarget<'executor>
{
    type Session = CompiledModuleAdamWSession<M, NativeCpuCompiledAdamW<'executor>>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.seal.validate_unchanged(&plan.module) {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let runtime = match plan.plan.prepare_native_cpu(self) {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            module,
            runtime,
            seal,
        })
    }
}

impl<M: Module> SessionTarget<CompiledModuleAdamWPlan<M>> for MetalSessionTarget {
    type Session = CompiledModuleAdamWSession<M, MetalCompiledAdamW>;
    type Error = CompiledModuleAdamWPrepareError<M, Error>;

    fn prepare(
        &self,
        plan: CompiledModuleAdamWPlan<M>,
    ) -> std::result::Result<Self::Session, Self::Error> {
        if let Err(source) = plan.seal.validate_unchanged(&plan.module) {
            return Err(CompiledModuleAdamWPrepareError { plan, source });
        }
        let runtime = match <Self as SessionTarget<&CompiledAdamWPlan>>::prepare(self, &plan.plan) {
            Ok(runtime) => runtime,
            Err(source) => return Err(CompiledModuleAdamWPrepareError { plan, source }),
        };
        let CompiledModuleAdamWPlan {
            module,
            seal,
            plan: _,
        } = plan;
        Ok(CompiledModuleAdamWSession {
            module,
            runtime,
            seal,
        })
    }
}

impl MetalCompiledAdamWPlan {
    pub fn deployment_identity(&self) -> u64 {
        self.inner.inner.deployment_identity()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.program_identity
    }

    /// Stable identity of the exact mixed transition executed by CPU and
    /// represented by this strict-Metal state-only plan.
    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.max_gradient_norm
    }

    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    pub fn dropout_config(&self) -> Option<CompiledDropoutConfig> {
        self.dropout.map(|dropout| dropout.config)
    }

    pub fn dropout_blocks_per_replay(&self) -> Option<u64> {
        self.dropout.map(|dropout| dropout.blocks_per_replay)
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.inner.summary()
    }

    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.inner.rendered_items()
    }

    /// Creates all native resources and uploads the captured recurrent
    /// frontier once. No training step is executed during preparation.
    pub fn prepare(self, device: MetalDevice) -> Result<MetalCompiledAdamW> {
        self.prepare_inner(device, None)
    }

    /// Creates the persistent training session and binds an epoch-state
    /// scoreboard before the first step can execute.
    pub fn prepare_with_scoreboard(
        self,
        device: MetalDevice,
        context: MetalScoreboardContext,
    ) -> Result<MetalCompiledAdamW> {
        let recorder = MetalSessionScoreboard::new_epoch_state(context, &self.inner.inner);
        self.prepare_inner(device, Some(recorder))
    }

    fn prepare_inner(
        self,
        device: MetalDevice,
        recorder: Option<MetalSessionScoreboard>,
    ) -> Result<MetalCompiledAdamW> {
        let inner = self.inner.prepare(device.clone(), recorder)?;
        let partial_flush = self
            .partial_flush
            .map(|plan| {
                plan.prepare(device, &inner.session)
                    .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledAdamW {
            inner,
            partial_flush,
            progress: self.progress,
            flush_capture_identity: self.flush_capture_identity,
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            max_gradient_norm: self.max_gradient_norm,
            loss_scale: self.loss_scale,
            dropout: self.dropout,
            frozen_parameters: self.frozen_parameters,
        })
    }
}

impl MetalCompiledTrainingPlan {
    fn prepare(
        self,
        device: MetalDevice,
        recorder: Option<MetalSessionScoreboard>,
    ) -> Result<MetalCompiledTrainingProgram> {
        let session = self
            .inner
            .prepare(device.clone())
            .map_err(metal_training_error)?;
        let evaluation = self
            .evaluation
            .map(|(plan, output_names, capture_identity)| {
                plan.prepare(device.clone(), &session)
                    .map(|session| (session, output_names, capture_identity))
                    .map_err(metal_training_error)
            })
            .transpose()?;
        let scoreboard = recorder
            .map(|recorder| {
                MetalScoreboardObserver::bind(recorder, &session)
                    .map_err(|error| training(format!("compiled Metal scoreboard: {error}")))
            })
            .transpose()?;
        Ok(MetalCompiledTrainingProgram {
            session,
            inputs: self.inputs,
            output_names: self.output_names,
            state_input_keys: self.state_input_keys,
            program_identity: self.program_identity,
            scoreboard,
            evaluation,
        })
    }
}

impl MetalCompiledTrainingProgram {
    fn prepare_inputs(
        &self,
        mut inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<BTreeMap<String, TensorData>> {
        validate_step_inputs(&self.inputs, &inputs, &learning_rate)?;
        inputs.insert(LEARNING_RATE_INPUT.into(), learning_rate);
        Ok(inputs)
    }

    fn observe_committed_step(&mut self, run: &MetalDeviceRun) {
        if let Some(scoreboard) = &mut self.scoreboard {
            scoreboard.observe(run);
        }
    }

    fn run(&mut self, provided: &BTreeMap<String, TensorData>) -> Result<MetalCompiledTrainingRun> {
        let run = self.session.run(provided).map_err(metal_training_error)?;
        self.observe_committed_step(&run);
        let (outputs, report) = run.into_parts();
        debug_assert_eq!(outputs.len(), 1 + self.output_names.len());
        let mut outputs = outputs.into_iter();
        let loss = outputs
            .next()
            .expect("compiled Metal output cardinality was authenticated before preparation");
        let outputs = self.output_names.iter().cloned().zip(outputs).collect();
        Ok(MetalCompiledTrainingRun {
            loss,
            outputs,
            report,
        })
    }

    fn run_without_host_outputs(
        &mut self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<MetalDeviceRunReport> {
        let run = self
            .session
            .run_epoch_without_host_outputs(provided)
            .map_err(metal_training_error)?;
        debug_assert!(run.outputs().is_empty());
        debug_assert_eq!(run.report().output_count, 0);
        debug_assert_eq!(run.report().retained_d2h_calls, 0);
        debug_assert_eq!(run.report().retained_d2h_bytes, 0);
        self.observe_committed_step(&run);
        let (_, report) = run.into_parts();
        Ok(report)
    }

    fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<MetalCompiledEvaluationResult> {
        validate_evaluation_inputs(&self.inputs, &inputs)?;
        let (evaluation, output_names, capture_identity) = self
            .evaluation
            .as_mut()
            .ok_or_else(|| training("compiled evaluation is not attached"))?;
        let run = evaluation
            .run(self.session.state_epoch(), &inputs)
            .map_err(metal_training_error)?;
        let (values, report) = run.into_parts();
        let inner = evaluation_result(values, output_names, *capture_identity)?;
        Ok(MetalCompiledEvaluationResult { inner, report })
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        self.evaluation
            .as_ref()
            .map(|(_, _, capture_identity)| *capture_identity)
    }

    fn state_snapshots(&self) -> Result<BTreeMap<RecurrentStateKey, TensorData>> {
        let snapshots = self
            .session
            .state_snapshots()
            .map_err(metal_training_error)?;
        if snapshots.len() != self.state_input_keys.len()
            || snapshots.keys().ne(self.state_input_keys.keys())
        {
            return Err(training("compiled Metal state inventory mismatch"));
        }
        snapshots
            .into_iter()
            .map(|(input, value)| {
                let key = self
                    .state_input_keys
                    .get(&input)
                    .cloned()
                    .ok_or_else(|| training("compiled Metal state key is absent"))?;
                Ok((key, value))
            })
            .collect()
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        let state_inputs = self
            .session
            .state_inputs()
            .iter()
            .map(|input| (input.name.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        if state_inputs.len() != self.state_input_keys.len()
            || state_inputs
                .keys()
                .copied()
                .ne(self.state_input_keys.keys().map(String::as_str))
        {
            return Err(training("compiled Metal state inventory mismatch"));
        }
        let expected = self
            .state_input_keys
            .iter()
            .filter_map(|(input, key)| {
                key.parameter_name()
                    .map(|name| (input.clone(), name.to_owned()))
            })
            .collect::<BTreeMap<_, _>>();
        if expected.is_empty() {
            return Err(training("compiled Metal parameter inventory is empty"));
        }
        let requested = expected
            .keys()
            .map(|name| state_inputs[name.as_str()].desc.id)
            .collect::<BTreeSet<_>>();
        if requested.len() != expected.len() {
            return Err(training("compiled Metal parameter state identities repeat"));
        }
        let snapshots = self
            .session
            .state_snapshot_subset(&requested)
            .map_err(metal_training_error)?;
        if snapshots.len() != expected.len() || snapshots.keys().ne(expected.keys()) {
            return Err(training(
                "compiled Metal parameter snapshot inventory mismatch",
            ));
        }
        snapshots
            .into_iter()
            .map(|(input, value)| {
                let name = expected
                    .get(&input)
                    .cloned()
                    .ok_or_else(|| training("compiled Metal parameter snapshot is unknown"))?;
                Ok((name, value))
            })
            .collect()
    }
}

impl MetalCompiledAdamW {
    fn prepare_step(
        &self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<(AdamWProgress, BTreeMap<String, TensorData>)> {
        let inputs = self.inner.prepare_inputs(inputs, learning_rate)?;
        let next = self
            .progress
            .advance_replay(self.gradient_accumulation_steps)?;
        if let Some(dropout) = self.dropout {
            expected_dropout_counter(dropout, next.replay_step)?;
        }
        Ok((next, inputs))
    }

    pub fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWStepResult> {
        let (next, provided) = self.prepare_step(inputs, learning_rate)?;
        let MetalCompiledTrainingRun {
            loss,
            outputs,
            report,
        } = self.inner.run(&provided)?;
        self.progress = next;
        let inner = adamw_step_result(
            CompiledTrainingStepResult {
                loss,
                outputs,
                step: self.progress.replay_step,
                capture_identity: self.inner.program_identity,
                clip_report: None,
                window_loss: None,
            },
            self.progress,
            1,
            self.gradient_accumulation_steps,
        );
        Ok(MetalCompiledAdamWStepResult { inner, report })
    }

    /// Executes and commits the identical captured training program while
    /// leaving its loss and named outputs on the device. Batch inputs and the
    /// learning rate are still staged, the complete inactive state bank is
    /// produced, and successful replay/optimizer progress advances normally.
    /// Use [`Self::step`] whenever the caller needs to observe loss or outputs.
    pub fn step_without_host_outputs(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWCommitResult> {
        let (next, provided) = self.prepare_step(inputs, learning_rate)?;
        let report = self.inner.run_without_host_outputs(&provided)?;
        self.progress = next;
        Ok(MetalCompiledAdamWCommitResult {
            progress: self.progress,
            capture_identity: self.inner.program_identity,
            report,
        })
    }

    pub fn evaluate(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<MetalCompiledEvaluationResult> {
        self.inner.evaluate(inputs)
    }

    pub fn evaluation_capture_identity(&self) -> Option<u64> {
        self.inner.evaluation_capture_identity()
    }

    /// Returns preparation evidence for both stateless evaluators sharing the
    /// training session's physical parameter banks. Imported trainable state
    /// contributes zero resident or initial-state uploads.
    pub fn evaluation_preparation_reports(
        &self,
    ) -> Option<[&crate::runtime::metal::MetalDevicePreparationReport; 2]> {
        self.inner
            .evaluation
            .as_ref()
            .map(|(evaluation, _, _)| evaluation.preparation_reports())
    }

    pub fn evaluation_summaries(&self) -> Option<[&MetalDeviceSessionSummary; 2]> {
        self.inner
            .evaluation
            .as_ref()
            .map(|(evaluation, _, _)| evaluation.summaries())
    }

    pub fn step_count(&self) -> u64 {
        self.progress.replay_step
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.max_gradient_norm
    }

    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.program_identity
    }

    pub fn metal_session(&self) -> &MetalDeviceSession {
        &self.inner.session
    }

    /// Returns the opt-in successful-step recorder, when preparation enabled it.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.inner
            .scoreboard
            .as_ref()
            .map(MetalScoreboardObserver::recorder)
    }

    /// Returns a deterministic snapshot of all successfully observed steps.
    pub fn execution_scoreboard_report(
        &self,
    ) -> std::result::Result<Option<MetalSessionScoreboardReport>, MetalScoreboardError> {
        self.execution_scoreboard()
            .map(MetalSessionScoreboard::report)
            .transpose()
    }

    /// Returns the first fail-soft measurement error, if recording froze.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.inner
            .scoreboard
            .as_ref()
            .and_then(MetalScoreboardObserver::first_error)
    }

    /// Downloads every currently committed recurrent value once and returns
    /// it under the optimizer's semantic state keys.
    fn state_snapshots(&self) -> Result<BTreeMap<RecurrentStateKey, TensorData>> {
        self.inner.state_snapshots()
    }

    pub fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        self.inner.parameter_snapshots()
    }

    pub fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(self.state_snapshots()?, AdamWParameterState::FirstMoment)
    }

    pub fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(self.state_snapshots()?, AdamWParameterState::SecondMoment)
    }

    pub fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        metal_adamw_state_snapshots(
            self.state_snapshots()?,
            AdamWParameterState::GradientAccumulator,
        )
    }

    pub fn accumulation_index(&self) -> Result<u64> {
        Ok(self.progress.accumulation_index)
    }

    pub fn optimizer_step(&self) -> Result<u64> {
        Ok(self.progress.optimizer_step)
    }

    /// Explicit diagnostic download of the recurrent dropout counter.
    pub fn dropout_block_counter(&self) -> Result<Option<u64>> {
        self.dropout
            .map(|_| {
                Ok(self
                    .state_snapshots()?
                    .get(&RecurrentStateKey::dropout_counter())
                    .ok_or_else(|| training("compiled Metal dropout counter is absent"))?
                    .scalar_at(0)
                    .as_u64())
            })
            .transpose()
    }

    /// Clears a retained partial window entirely inside the epoch-swapped
    /// device frontier. No training run or host gradient download is performed.
    pub fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        let (next, result) = self.progress.cancel(self.gradient_accumulation_steps)?;
        if !result.did_discard() {
            return Ok(result);
        }
        let state_inputs = self
            .inner
            .session
            .state_inputs()
            .iter()
            .map(|input| (input.name.as_str(), &input.desc))
            .collect::<BTreeMap<_, _>>();
        let mut replacements = BTreeMap::new();
        for (input, key) in &self.inner.state_input_keys {
            if key.is_accumulation_reset_state() {
                let desc = state_inputs
                    .get(input.as_str())
                    .ok_or_else(|| training("compiled Metal reset state is absent"))?;
                replacements.insert(
                    input.clone(),
                    TensorData::zeros_with_dtype(desc.shape.clone(), desc.dtype)?,
                );
            }
        }
        let expected = self
            .inner
            .state_input_keys
            .values()
            .filter(|key| key.is_accumulation_reset_state())
            .count();
        if replacements.len() != expected
            || !replacements
                .values()
                .any(|value| value.dtype() == DType::U64)
        {
            return Err(training("compiled Metal reset state inventory mismatch"));
        }
        self.inner
            .session
            .replace_fixed_state(replacements)
            .map_err(metal_training_error)?;
        self.progress = next;
        Ok(result)
    }

    /// Commits a retained partial window through the separately rendered
    /// state-only capture while sharing the live epoch banks and queue.
    pub fn flush_partial_window(
        &mut self,
        learning_rate: TensorData,
    ) -> Result<MetalCompiledAdamWFlushResult> {
        validate_step_inputs(&BTreeMap::new(), &BTreeMap::new(), &learning_rate)?;
        let (next, result) = self
            .progress
            .flush_partial(self.gradient_accumulation_steps)?;
        if !result.did_update() {
            return Ok(MetalCompiledAdamWFlushResult {
                inner: result,
                report: None,
            });
        }
        let transition = self
            .partial_flush
            .as_mut()
            .ok_or_else(|| training("compiled Metal partial flush transition is absent"))?;
        let inputs = BTreeMap::from([(LEARNING_RATE_INPUT.to_owned(), learning_rate)]);
        let run = transition
            .run(&mut self.inner.session, &inputs)
            .map_err(metal_training_error)?;
        let (outputs, report) = run.into_parts();
        debug_assert!(outputs.is_empty());
        self.progress = next;
        Ok(MetalCompiledAdamWFlushResult {
            inner: result,
            report: Some(report),
        })
    }

    pub fn flush_capture_identity(&self) -> Option<u64> {
        self.flush_capture_identity
    }

    /// Preparation evidence for the state-only transition. Imported recurrent
    /// state must contribute zero initialization uploads.
    #[cfg(test)]
    pub(crate) fn flush_preparation_report(
        &self,
    ) -> Option<&crate::runtime::metal::MetalDevicePreparationReport> {
        self.partial_flush
            .as_ref()
            .map(MetalFixedStateTransitionSession::preparation_report)
    }

    /// Downloads one coherent active state bank and encodes the same portable
    /// checkpoint format accepted by [`CpuCompiledAdamW::compile_from_checkpoint`].
    pub fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        let states = self.state_snapshots()?;
        let optimizer_step = states
            .get(&RecurrentStateKey::adamw_global(AdamWGlobalState::Step))
            .ok_or_else(|| training("compiled Metal optimizer step is absent"))?
            .scalar_at(0)
            .as_u64();
        let accumulation_index = if self.gradient_accumulation_steps == 1 {
            0
        } else {
            states
                .get(&RecurrentStateKey::adamw_global(
                    AdamWGlobalState::AccumulationIndex,
                ))
                .ok_or_else(|| training("compiled Metal accumulation index is absent"))?
                .scalar_at(0)
                .as_u64()
        };
        validate_adamw_progress(self.progress, self.gradient_accumulation_steps)?;
        if optimizer_step != self.progress.optimizer_step
            || accumulation_index != self.progress.accumulation_index
        {
            return Err(training("compiled Metal AdamW progress state mismatch"));
        }
        let dropout_block_counter = self
            .dropout
            .map(|dropout| {
                let counter = states
                    .get(&RecurrentStateKey::dropout_counter())
                    .ok_or_else(|| training("compiled Metal dropout counter is absent"))?
                    .scalar_at(0)
                    .as_u64();
                if counter != expected_dropout_counter(dropout, self.progress.replay_step)? {
                    return Err(training(
                        "compiled Metal dropout counter and replay progress diverged",
                    ));
                }
                Ok(counter)
            })
            .transpose()?;
        let parameters = metal_parameter_snapshots(&states);
        let first_moments =
            metal_adamw_state_snapshots(states.clone(), AdamWParameterState::FirstMoment)?;
        let second_moments =
            metal_adamw_state_snapshots(states.clone(), AdamWParameterState::SecondMoment)?;
        let gradient_accumulators =
            metal_adamw_state_snapshots(states, AdamWParameterState::GradientAccumulator)?;
        CompiledAdamWCheckpoint::from_bytes(encode_adamw_checkpoint(
            AdamWCheckpointProgress {
                capture_identity: self.inner.program_identity,
                replay_step: self.progress.replay_step,
                optimizer_step: self.progress.optimizer_step,
                accumulation_steps: self.gradient_accumulation_steps,
                accumulation_index: self.progress.accumulation_index,
                discarded_microbatches: self.progress.discarded_microbatches,
                flushed_window_count: self.progress.flushed_window_count,
                flushed_microbatch_count: self.progress.flushed_microbatch_count,
                flush_capture_identity: self.flush_capture_identity,
                dropout_block_counter,
                accumulated_token_count: None,
                window_loss_report: false,
                reset_transition_count: 0,
                reset_capture_identity: None,
            },
            AdamWCheckpointTensors {
                parameters,
                first_moments,
                second_moments,
                gradient_accumulators,
                accumulated_loss_numerator: None,
            },
        )?)
    }
}

impl CompiledTrainingRuntime for MetalCompiledAdamW {
    type Step = MetalCompiledAdamWStepResult;

    fn step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        MetalCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn step_count(&self) -> u64 {
        MetalCompiledAdamW::step_count(self)
    }

    fn capture_identity(&self) -> u64 {
        MetalCompiledAdamW::capture_identity(self)
    }

    fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::parameter_snapshots(self)
    }

    fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
        publish_parameters_with_freeze_policy(
            module,
            self.parameter_snapshots()?,
            &self.frozen_parameters,
        )
    }
}

impl CompiledEvaluationRuntime for MetalCompiledAdamW {
    type Evaluation = MetalCompiledEvaluationResult;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        MetalCompiledAdamW::evaluate(self, inputs)
    }

    fn evaluation_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::evaluation_capture_identity(self)
    }
}

impl CompiledCheckpointRuntime for MetalCompiledAdamW {
    type Checkpoint = CompiledAdamWCheckpoint;

    fn checkpoint(&self) -> Result<CompiledAdamWCheckpoint> {
        MetalCompiledAdamW::checkpoint(self)
    }
}

impl CompiledAdamWRuntime for MetalCompiledAdamW {
    fn gradient_accumulation_steps(&self) -> u64 {
        MetalCompiledAdamW::gradient_accumulation_steps(self)
    }

    fn max_gradient_norm(&self) -> Option<f32> {
        MetalCompiledAdamW::max_gradient_norm(self)
    }

    fn loss_scale(&self) -> f32 {
        MetalCompiledAdamW::loss_scale(self)
    }

    fn optimizer_step(&self) -> Result<u64> {
        MetalCompiledAdamW::optimizer_step(self)
    }

    fn accumulation_index(&self) -> Result<u64> {
        MetalCompiledAdamW::accumulation_index(self)
    }

    fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        MetalCompiledAdamW::zero_grad(self)
    }

    fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::first_moment_snapshots(self)
    }

    fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::second_moment_snapshots(self)
    }

    fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
        MetalCompiledAdamW::gradient_accumulator_snapshots(self)
    }
}

impl CompiledAdamWFlushRuntime for MetalCompiledAdamW {
    type Flush = MetalCompiledAdamWFlushResult;

    fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        MetalCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn flush_capture_identity(&self) -> Option<u64> {
        MetalCompiledAdamW::flush_capture_identity(self)
    }
}

fn metal_parameter_snapshots(
    states: &BTreeMap<RecurrentStateKey, TensorData>,
) -> BTreeMap<String, TensorData> {
    states
        .iter()
        .filter_map(|(key, value)| {
            key.parameter_name()
                .map(|name| (name.to_owned(), value.clone()))
        })
        .collect()
}

fn metal_adamw_state_snapshots(
    states: BTreeMap<RecurrentStateKey, TensorData>,
    state: AdamWParameterState,
) -> Result<BTreeMap<String, TensorData>> {
    Ok(states
        .into_iter()
        .filter_map(|(key, value)| {
            key.parameter_for_adamw_state(state)
                .map(|name| (name.to_owned(), value))
        })
        .collect())
}

fn validate_adamw_progress(progress: AdamWProgress, accumulation_steps: u64) -> Result<()> {
    let AdamWProgress {
        replay_step,
        optimizer_step,
        accumulation_index,
        discarded_microbatches,
        flushed_window_count,
        flushed_microbatch_count,
        reset_transition_count,
    } = progress;
    if accumulation_steps == 0 || accumulation_index >= accumulation_steps {
        return Err(training(
            "compiled AdamW checkpoint accumulation progress is invalid",
        ));
    }
    if accumulation_steps == 1
        && (discarded_microbatches != 0
            || flushed_window_count != 0
            || flushed_microbatch_count != 0
            || reset_transition_count != 0)
    {
        return Err(training(
            "compiled AdamW checkpoint discarded progress is invalid",
        ));
    }
    if reset_transition_count > discarded_microbatches {
        return Err(training(
            "compiled AdamW checkpoint reset progress is invalid",
        ));
    }
    let maximum_flushed_microbatches = flushed_window_count
        .checked_mul(
            accumulation_steps
                .checked_sub(1)
                .ok_or_else(|| training("compiled AdamW checkpoint progress underflows"))?,
        )
        .ok_or_else(|| training("compiled AdamW checkpoint progress overflows"))?;
    if flushed_window_count > optimizer_step
        || (flushed_window_count == 0) != (flushed_microbatch_count == 0)
        || (flushed_window_count != 0
            && (flushed_microbatch_count < flushed_window_count
                || flushed_microbatch_count > maximum_flushed_microbatches))
    {
        return Err(training(
            "compiled AdamW checkpoint flushed progress is invalid",
        ));
    }
    let complete_windows = optimizer_step - flushed_window_count;
    let expected_replay = complete_windows
        .checked_mul(accumulation_steps)
        .and_then(|step| step.checked_add(flushed_microbatch_count))
        .and_then(|step| step.checked_add(accumulation_index))
        .and_then(|step| step.checked_add(discarded_microbatches))
        .ok_or_else(|| training("compiled AdamW checkpoint progress overflows"))?;
    if replay_step != expected_replay {
        return Err(training(
            "compiled AdamW checkpoint replay and optimizer progress diverged",
        ));
    }
    Ok(())
}

fn validate_cpu_adamw_state(
    inner: &CpuCompiledTrainingProgram,
    progress: AdamWProgress,
    accumulation_steps: u64,
) -> Result<()> {
    let optimizer_step = inner
        .global_snapshot(AdamWGlobalState::Step)?
        .scalar_at(0)
        .as_u64();
    let accumulation_index = if accumulation_steps == 1 {
        0
    } else {
        inner
            .global_snapshot(AdamWGlobalState::AccumulationIndex)?
            .scalar_at(0)
            .as_u64()
    };
    if optimizer_step != progress.optimizer_step
        || accumulation_index != progress.accumulation_index
    {
        return Err(training("compiled CPU AdamW progress state mismatch"));
    }
    Ok(())
}

fn canonical_parameters(
    parameters: impl IntoIterator<Item = TrainingParameterInit>,
) -> Result<BTreeMap<String, TensorData>> {
    let mut values = BTreeMap::new();
    for parameter in parameters {
        validate_user_name(&parameter.name, "parameter")?;
        if parameter.value.dtype() != DType::F32 {
            return Err(training("compiled training parameters must be F32"));
        }
        checked_bytes(&parameter.value)?;
        if values.insert(parameter.name, parameter.value).is_some() {
            return Err(training("duplicate compiled parameter name"));
        }
    }
    Ok(values)
}

fn validate_weight_decay_exclusion_names<'a>(
    config: &CompiledAdamWConfig,
    parameter_names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    if config.weight_decay_exclusions.is_empty() {
        return Ok(());
    }
    let parameter_names = parameter_names.into_iter().collect::<BTreeSet<_>>();
    if let Some(name) = config
        .weight_decay_exclusions
        .iter()
        .find(|name| !parameter_names.contains(name.as_str()))
    {
        return Err(training(format!(
            "compiled AdamW weight-decay exclusion name {name:?} is unknown"
        )));
    }
    Ok(())
}

fn validate_user_name(name: &str, kind: &str) -> Result<()> {
    if name.is_empty() || name == "loss" || name.starts_with(INTERNAL_PREFIX) {
        return Err(training(format!("invalid compiled {kind} name")));
    }
    Ok(())
}

fn checked_bytes(value: &TensorData) -> Result<usize> {
    value
        .len()
        .checked_mul(value.dtype().itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

fn checked_recurrent_state_extent(
    bytes: impl IntoIterator<Item = usize>,
) -> Result<(usize, usize)> {
    bytes
        .into_iter()
        .try_fold((0usize, 0usize), |(count, total), bytes| {
            Ok((
                count
                    .checked_add(1)
                    .ok_or_else(|| training("compiled recurrent state count overflows"))?,
                total
                    .checked_add(bytes)
                    .ok_or_else(|| training("compiled recurrent state bytes overflow"))?,
            ))
        })
}

fn checked_descriptor(shape: &Shape, dtype: DType) -> Result<usize> {
    shape
        .numel()
        .map_err(|_| training("compiled tensor element extent overflow"))?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| training("compiled tensor byte extent overflow"))
}

fn state_for(buffer: u64, value: &TensorData) -> Result<BufferState> {
    Ok(BufferState {
        buffer,
        version: 0,
        shape: value.shape().clone(),
        dtype: value.dtype(),
        bytes: checked_bytes(value)?,
    })
}

fn validate_loss(graph: &Graph, loss: NodeId) -> Result<()> {
    if graph.dtype(loss)? != DType::F32 || graph.shape(loss)? != &Shape::from([]) {
        return Err(training(
            "compiled training loss must be a rank-zero F32 scalar",
        ));
    }
    Ok(())
}

fn validate_evaluation_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    provided: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if expected.len() != provided.len() || expected.keys().ne(provided.keys()) {
        return Err(training("compiled evaluation input names mismatch"));
    }
    for (name, (shape, dtype)) in expected {
        let value = &provided[name];
        if value.shape() != shape || value.dtype() != *dtype {
            return Err(training(format!(
                "compiled evaluation input {name:?} descriptor mismatch"
            )));
        }
        checked_bytes(value)?;
    }
    Ok(())
}

fn validate_outputs<'a>(
    loss: NodeId,
    outputs: &BTreeMap<String, NodeId>,
    reserved_user_names: impl IntoIterator<Item = &'a String>,
) -> Result<()> {
    let mut nodes = BTreeSet::from([loss]);
    let reserved_user_names = reserved_user_names
        .into_iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for (name, node) in outputs {
        validate_user_name(name, "output")?;
        if name == "loss" || reserved_user_names.contains(name.as_str()) {
            return Err(training(
                "compiled output name collides with another user name",
            ));
        }
        if !nodes.insert(*node) {
            return Err(training("duplicate compiled output node"));
        }
    }
    Ok(())
}

fn validate_external_binding_ownership<'a>(
    capture: &CapturedMixedSchedule,
    configured_inputs: impl IntoIterator<Item = &'a String>,
) -> Result<()> {
    let mut external = configured_inputs
        .into_iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    external.insert(LEARNING_RATE_INPUT);
    for binding in &capture.state_bindings {
        let input = capture
            .schedule
            .inputs
            .iter()
            .find(|input| input.node == binding.input_node)
            .ok_or_else(|| training("compiled state input ABI is absent"))?;
        if external.contains(input.name.as_str()) {
            return Err(training(
                "compiled external input shadows persistent state binding",
            ));
        }
    }
    Ok(())
}

fn collect_state_bindings(
    schedule: &Schedule,
    states: &BTreeMap<NodeId, BufferState>,
) -> Result<Vec<ScheduleStateBinding>> {
    let mut bindings = Vec::new();
    let mut seen = BTreeSet::new();
    let mut bound_nodes = BTreeSet::new();
    for item in &schedule.items {
        for binding in &item.input_bindings {
            let Some(state) = states.get(&binding.input_node) else {
                continue;
            };
            if !seen.insert((item.id, binding.input_node)) {
                return Err(training("duplicate compiled state input binding"));
            }
            bound_nodes.insert(binding.input_node);
            bindings.push(ScheduleStateBinding {
                state: state.clone(),
                view: None,
                consumer_item: item.id,
                consumer_node: item.node,
                input_node: binding.input_node,
                desc: binding.desc.clone(),
                abi_index: binding.abi_index,
            });
        }
    }
    if bindings.is_empty() || states.keys().any(|node| !bound_nodes.contains(node)) {
        return Err(training("compiled state input is not reachable"));
    }
    Ok(bindings)
}

fn value_binding(
    schedule: &Schedule,
    node: NodeId,
    effect_item: u64,
) -> Result<ScheduleValueBinding> {
    let (producer_item, producer) = schedule
        .items
        .iter()
        .enumerate()
        .find(|(_, item)| item.primary_output().id == node.index() as u64)
        .ok_or_else(|| training("compiled update output is not materialized"))?;
    Ok(ScheduleValueBinding {
        producer_item: u64::try_from(producer_item)
            .map_err(|_| training("compiled producer index overflow"))?,
        producer_node: node,
        producer_output: producer.primary_output().clone(),
        abi_index: 0,
        effect_item,
        source_position: 0,
    })
}

fn effect_states(effects: &EffectGraph) -> Result<Vec<BufferState>> {
    let plan = effects.plan();
    plan.validate().map_err(effect_error)?;
    let mut states = BTreeMap::new();
    for step in plan.steps {
        for state in step.reads.into_iter().chain([step.write]) {
            states.insert((state.buffer, state.version), state);
        }
    }
    Ok(states.into_values().collect())
}

fn validate_step_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    actual: &BTreeMap<String, TensorData>,
    learning_rate: &TensorData,
) -> Result<()> {
    validate_training_inputs(expected, actual)?;
    validate_learning_rate(learning_rate)
}

fn validate_training_inputs(
    expected: &BTreeMap<String, (Shape, DType)>,
    actual: &BTreeMap<String, TensorData>,
) -> Result<()> {
    if actual.len() != expected.len() || actual.keys().ne(expected.keys()) {
        return Err(training("compiled training input names do not match"));
    }
    for (name, value) in actual {
        let (shape, dtype) = &expected[name];
        if value.shape() != shape || value.dtype() != *dtype {
            return Err(training("compiled training input descriptor mismatch"));
        }
        checked_bytes(value)?;
    }
    Ok(())
}

fn validate_token_weighted_accumulation(
    inputs: &BTreeMap<String, (Shape, DType)>,
    mask_input: &str,
    accumulation_steps: u64,
) -> Result<()> {
    if accumulation_steps <= 1 {
        return Err(training(
            "compiled AdamW token-weighted accumulation requires more than one step",
        ));
    }
    let (shape, dtype) = inputs
        .get(mask_input)
        .ok_or_else(|| training("compiled AdamW token-weight mask must name an existing input"))?;
    let mask_elements = shape.numel()?;
    if *dtype != DType::F32 || mask_elements == 0 {
        return Err(training(
            "compiled AdamW token-weight mask must be nonempty fixed-shape F32",
        ));
    }
    let mask_elements = u64::try_from(mask_elements)
        .map_err(|_| training("compiled AdamW token-weight mask element count overflows"))?;
    let maximum_count = mask_elements
        .checked_mul(accumulation_steps)
        .ok_or_else(|| training("compiled AdamW token-weight count bound overflows"))?;
    if maximum_count > MAX_EXACT_F32_INTEGER_COUNT {
        return Err(training(
            "compiled AdamW token-weight count must remain exactly representable in F32",
        ));
    }
    Ok(())
}

fn reject_token_weighted_scalar_loss(config: &CompiledAdamWConfig) -> Result<()> {
    if config.token_weight_mask_input.is_some() {
        return Err(training(
            "compiled AdamW token-weighted accumulation requires the token-mean-loss compile surface",
        ));
    }
    Ok(())
}

fn lower_compiled_adamw_objective(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    objective: CompiledAdamWObjective,
) -> Result<NodeId> {
    match objective {
        CompiledAdamWObjective::Scalar(loss) => {
            reject_token_weighted_scalar_loss(config)?;
            Ok(loss)
        }
        CompiledAdamWObjective::TokenMean(losses) => {
            let (mask_input, mask_shape) = token_mean_loss_descriptor(config)?;
            let mask = inputs.get(&mask_input).copied().ok_or_else(|| {
                training("compiled AdamW token-weight mask input is absent during compilation")
            })?;
            lower_token_mean_loss(graph, losses, mask, &mask_shape)
        }
    }
}

fn token_mean_loss_descriptor(config: &CompiledAdamWConfig) -> Result<(String, Shape)> {
    let mask_input = config.token_weight_mask_input.as_ref().ok_or_else(|| {
        training("compiled AdamW token-mean-loss compilation requires token weighting")
    })?;
    let (shape, dtype) = config
        .inputs
        .get(mask_input)
        .ok_or_else(|| training("compiled AdamW token-weight mask must name an existing input"))?;
    if *dtype != DType::F32 {
        return Err(training(
            "compiled AdamW token-weight mask must be nonempty fixed-shape F32",
        ));
    }
    Ok((mask_input.clone(), shape.clone()))
}

fn lower_token_mean_loss(
    graph: &mut Graph,
    losses: NodeId,
    mask: NodeId,
    expected_shape: &Shape,
) -> Result<NodeId> {
    if graph.dtype(losses)? != DType::F32 || graph.shape(losses)? != expected_shape {
        return Err(training(
            "compiled AdamW per-token losses must exactly match the token-weight mask descriptor",
        ));
    }
    if graph.dtype(mask)? != DType::F32 || graph.shape(mask)? != expected_shape {
        return Err(training(
            "compiled AdamW token-weight mask descriptor changed during compilation",
        ));
    }
    let weighted = graph.mul(losses, mask)?;
    let numerator = graph.sum_all(weighted)?;
    let denominator = graph.sum_all(mask)?;
    graph.div(numerator, denominator)
}

fn validate_token_weight_mask(
    inputs: &BTreeMap<String, TensorData>,
    mask_input: Option<&str>,
) -> Result<u64> {
    let Some(mask_input) = mask_input else {
        return Ok(1);
    };
    let mask = inputs
        .get(mask_input)
        .ok_or_else(|| training("compiled AdamW token-weight mask input is absent"))?;
    let mut valid_tokens = 0_u64;
    for index in 0..mask.shape().numel()? {
        let value = mask.scalar_at(index).as_f64();
        if !value.is_finite() || (value != 0.0 && value != 1.0) {
            return Err(training(
                "compiled AdamW token-weight mask must contain finite binary values",
            ));
        }
        valid_tokens = valid_tokens
            .checked_add(u64::from(value == 1.0))
            .ok_or_else(|| training("compiled AdamW token-weight count overflows"))?;
    }
    if valid_tokens == 0 {
        return Err(training(
            "compiled AdamW token-weight mask must contain at least one valid token",
        ));
    }
    Ok(valid_tokens)
}

fn validate_retained_token_count(
    inputs: &BTreeMap<String, (Shape, DType)>,
    mask_input: &str,
    accumulation_index: u64,
    count: u64,
) -> Result<()> {
    let (shape, _) = inputs
        .get(mask_input)
        .ok_or_else(|| training("compiled AdamW token-weight mask must name an existing input"))?;
    let mask_elements = u64::try_from(shape.numel()?)
        .map_err(|_| training("compiled AdamW token-weight mask element count overflows"))?;
    let maximum_count = mask_elements
        .checked_mul(accumulation_index)
        .ok_or_else(|| training("compiled AdamW retained token count bound overflows"))?;
    if count < accumulation_index || count > maximum_count {
        return Err(training(
            "compiled AdamW retained token count is inconsistent with progress",
        ));
    }
    Ok(())
}

fn validate_learning_rate(learning_rate: &TensorData) -> Result<()> {
    if learning_rate.shape() != &Shape::from([]) || learning_rate.dtype() != DType::F32 {
        return Err(training(
            "compiled training learning rate must be rank-zero F32",
        ));
    }
    checked_bytes(learning_rate)?;
    Ok(())
}

fn validate_learning_rate_for_policy(
    learning_rate: &TensorData,
    policy: CpuNonFinitePolicy,
) -> Result<()> {
    validate_learning_rate(learning_rate)?;
    if policy == CpuNonFinitePolicy::RejectTransition {
        validate_finite_tensors(std::iter::once(learning_rate), "external learning rate")?;
    }
    Ok(())
}

fn validate_staged_transition<'a>(
    outputs: &[TensorData],
    successors: impl IntoIterator<Item = &'a TensorData>,
    policy: CpuNonFinitePolicy,
    require_loss: bool,
) -> std::result::Result<(), String> {
    if policy == CpuNonFinitePolicy::Propagate {
        return Ok(());
    }
    if require_loss {
        let loss = outputs
            .first()
            .ok_or_else(|| "compiled CPU transition loss is absent".to_owned())?;
        if loss.shape() != &Shape::from([]) || loss.dtype() != DType::F32 {
            return Err("compiled CPU transition loss must be rank-zero F32".to_owned());
        }
        if has_non_finite_f32(std::iter::once(loss)) {
            return Err("compiled CPU transition has a non-finite loss".to_owned());
        }
    }
    if has_non_finite_f32(successors) {
        return Err("compiled CPU transition has a non-finite recurrent successor".to_owned());
    }
    Ok(())
}

fn validate_staged_clip_report(
    outputs: &[TensorData],
    start: usize,
    enabled: bool,
    policy: CpuNonFinitePolicy,
) -> std::result::Result<(), String> {
    if !enabled || policy == CpuNonFinitePolicy::Propagate {
        return Ok(());
    }
    let report = outputs
        .get(start..start + 2)
        .ok_or_else(|| "compiled CPU clip-report output inventory differs".to_owned())?;
    if report
        .iter()
        .any(|value| value.shape() != &Shape::from([]) || value.dtype() != DType::F32)
    {
        return Err("compiled CPU clip report must contain rank-zero F32 values".to_owned());
    }
    if has_non_finite_f32(report.iter()) {
        return Err("compiled CPU transition has a non-finite clip report".to_owned());
    }
    Ok(())
}

fn validate_staged_window_loss_report(
    outputs: &[TensorData],
    start: usize,
    enabled: bool,
    policy: CpuNonFinitePolicy,
) -> std::result::Result<(), String> {
    if !enabled || policy == CpuNonFinitePolicy::Propagate {
        return Ok(());
    }
    let report = outputs
        .get(start..start + 2)
        .ok_or_else(|| "compiled CPU window-loss output inventory differs".to_owned())?;
    if report[0].shape() != &Shape::from([]) || report[0].dtype() != DType::F32 {
        return Err("compiled CPU window loss must be rank-zero F32".to_owned());
    }
    if report[1].shape() != &Shape::from([]) || report[1].dtype() != DType::U64 {
        return Err("compiled CPU window-loss weight must be rank-zero U64".to_owned());
    }
    if has_non_finite_f32(std::iter::once(&report[0])) {
        return Err("compiled CPU transition has a non-finite window loss".to_owned());
    }
    if report[1].scalar_at(0).as_u64() == 0 {
        return Err("compiled CPU window-loss weight must be positive".to_owned());
    }
    Ok(())
}

fn validate_finite_tensors<'a>(
    tensors: impl IntoIterator<Item = &'a TensorData>,
    role: &str,
) -> Result<()> {
    if has_non_finite_f32(tensors) {
        return Err(training(format!(
            "compiled CPU transition has a non-finite {role}"
        )));
    }
    Ok(())
}

fn has_non_finite_f32<'a>(tensors: impl IntoIterator<Item = &'a TensorData>) -> bool {
    tensors.into_iter().any(|tensor| {
        tensor.dtype() == DType::F32 && tensor.values().iter().any(|value| !value.is_finite())
    })
}

fn schedule_error(error: impl std::fmt::Display) -> Error {
    training(format!("compiled schedule: {error}"))
}

fn replay_error(error: ReplayError) -> Error {
    training(format!("compiled replay: {error:?}"))
}

fn captured_inference_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled recurrent capture: {error:?}"))
}

fn metal_training_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled Metal runtime: {error:?}"))
}

fn runtime_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled persistent runtime: {error:?}"))
}

fn effect_error(error: impl std::fmt::Debug) -> Error {
    training(format!("compiled effect graph: {error:?}"))
}

fn training(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, LossOptions, Op, Parameter, cross_entropy};
    use std::{cell::Cell, collections::HashMap, rc::Rc};

    #[test]
    fn compiled_state_aliases_receive_explicit_capture_owners() {
        let mut graph = Graph::new();
        let input = graph.input_dtype_requires_grad("state", [], DType::F32, false);
        let constant = graph
            .full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)
            .unwrap();
        let expanded = graph
            .lazy_full_with_dtype(Shape::new([2]), Scalar::I(0), DType::F32)
            .unwrap();
        let one = graph
            .full_with_dtype(Shape::from([]), Scalar::F(1.0), DType::F32)
            .unwrap();
        let computed = graph.add(input, one).unwrap();
        let sources = [input, constant, expanded, computed];
        let owners = materialize_compiled_state_aliases(&mut graph, &sources).unwrap();

        for (source, owner) in sources[..3].iter().zip(&owners[..3]) {
            assert_ne!(owner, source);
            assert!(matches!(
                graph.op(*owner).unwrap(),
                Op::Contiguous { input } if input == source
            ));
        }
        assert_eq!(owners[3], computed);
        let scheduled = schedule_many(&graph, &owners).unwrap();
        let produced = scheduled
            .items
            .iter()
            .flat_map(|item| item.outputs.iter())
            .map(|output| output.id)
            .collect::<BTreeSet<_>>();
        assert!(
            owners
                .iter()
                .all(|owner| produced.contains(&(owner.index() as u64)))
        );
    }

    #[test]
    fn compiled_dropout_reserves_source_order_blocks_only_for_active_f32_draws() {
        let mut graph = Graph::new();
        let counter = graph.input_dtype_requires_grad("counter", [], DType::U64, false);
        let f32_three = graph.input_dtype_requires_grad("f32_three", [3], DType::F32, true);
        let f32_four = graph.input_dtype_requires_grad("f32_four", [4], DType::F32, true);
        let empty = graph.input_dtype_requires_grad("empty", [0], DType::F32, true);
        let integer = graph.input_dtype_requires_grad("integer", [3], DType::I32, false);
        let mut stream = CompiledDropoutStream::new(
            counter,
            CompiledDropoutConfig::new(CompiledDropoutKey([3, 7])),
        );

        assert_eq!(
            stream.dropout(&mut graph, f32_three, 0.0).unwrap(),
            f32_three
        );
        assert_eq!(stream.dropout(&mut graph, empty, 0.5).unwrap(), empty);
        let all_zero = stream.dropout(&mut graph, f32_three, 1.0).unwrap();
        assert_eq!(graph.shape(all_zero).unwrap(), &Shape::new([3]));
        assert!(stream.dropout(&mut graph, integer, 0.5).is_err());
        let first = stream.dropout(&mut graph, f32_three, 0.5).unwrap();
        let second = stream.dropout(&mut graph, f32_four, 0.25).unwrap();
        assert_eq!(graph.shape(first).unwrap(), &Shape::new([3]));
        assert_eq!(graph.shape(second).unwrap(), &Shape::new([4]));

        let (successor, state) = stream.finish(&mut graph).unwrap();
        assert_eq!(state.blocks_per_replay, 4);
        assert_eq!(state.config.key().words(), [3, 7]);
        assert_eq!(graph.dtype(successor).unwrap(), DType::U64);
        assert!(!graph.requires_grad(successor).unwrap());
        let loss = graph.sum_all(first).unwrap();
        let gradient = graph.gradient_default(loss, &[f32_three]).unwrap()[0];
        let bindings = HashMap::from([
            (
                "counter".into(),
                TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(0)]).unwrap(),
            ),
            (
                "f32_three".into(),
                TensorData::new([3], vec![1.0, 1.0, 1.0]).unwrap(),
            ),
        ]);
        let cpu = CpuBackend;
        assert_eq!(
            cpu.execute(&graph, gradient, &bindings).unwrap(),
            cpu.execute(&graph, first, &bindings).unwrap(),
            "for unit inputs and p=0.5, the VJP is exactly the realized mask/(1-p)"
        );
        assert_eq!(
            (0..graph.node_count())
                .filter(|index| matches!(graph.op(NodeId(*index)).unwrap(), Op::Threefry { .. }))
                .count(),
            2
        );
    }

    struct TiedFrozenModule {
        shared: Parameter,
        frozen: Parameter,
        buffer: Parameter,
    }

    impl TiedFrozenModule {
        fn new(frozen: [f32; 2]) -> Self {
            Self {
                shared: Parameter::new(TensorData::new([2], vec![0.25, -0.5]).unwrap(), true),
                frozen: Parameter::new(TensorData::new([2], frozen.to_vec()).unwrap(), false),
                buffer: Parameter::new(TensorData::scalar(3.0), false),
            }
        }
    }

    impl Module for TiedFrozenModule {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            assert!(prefix.is_empty());
            visitor("shared".into(), &self.shared, StateKind::Parameter);
            visitor("shared_alias".into(), &self.shared, StateKind::Parameter);
            visitor("frozen".into(), &self.frozen, StateKind::Parameter);
            visitor("buffer".into(), &self.buffer, StateKind::Buffer);
        }
    }

    #[derive(Debug)]
    struct FinishRaceModule {
        weight: Parameter,
        finish_visits: Cell<u64>,
        race_after_second_visit: Cell<bool>,
    }

    struct CheckpointCountingRuntime {
        inner: CpuCompiledAdamW,
        checkpoint_calls: Rc<Cell<u64>>,
    }

    impl CompiledTrainingRuntime for CheckpointCountingRuntime {
        type Step = CompiledAdamWStepResult;

        fn step(
            &mut self,
            inputs: BTreeMap<String, TensorData>,
            learning_rate: TensorData,
        ) -> Result<Self::Step> {
            self.inner.step(inputs, learning_rate)
        }

        fn step_count(&self) -> u64 {
            self.inner.step_count()
        }

        fn capture_identity(&self) -> u64 {
            self.inner.capture_identity()
        }

        fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
            self.inner.parameter_snapshots()
        }

        fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
            CompiledTrainingRuntime::publish_parameters(&self.inner, module)
        }
    }

    impl CompiledCheckpointRuntime for CheckpointCountingRuntime {
        type Checkpoint = CompiledAdamWCheckpoint;

        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            self.checkpoint_calls
                .set(self.checkpoint_calls.get().saturating_add(1));
            self.inner.checkpoint()
        }
    }

    impl CompiledAdamWRuntime for CheckpointCountingRuntime {
        fn gradient_accumulation_steps(&self) -> u64 {
            self.inner.gradient_accumulation_steps()
        }

        fn max_gradient_norm(&self) -> Option<f32> {
            self.inner.max_gradient_norm()
        }

        fn loss_scale(&self) -> f32 {
            self.inner.loss_scale()
        }

        fn optimizer_step(&self) -> Result<u64> {
            self.inner.optimizer_step()
        }

        fn accumulation_index(&self) -> Result<u64> {
            self.inner.accumulation_index()
        }

        fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
            self.inner.zero_grad()
        }

        fn zero_grad_capture_identity(&self) -> Option<u64> {
            self.inner.zero_grad_capture_identity()
        }

        fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
            self.inner.first_moment_snapshots()
        }

        fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
            self.inner.second_moment_snapshots()
        }

        fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
            self.inner.gradient_accumulator_snapshots()
        }
    }

    impl FinishRaceModule {
        fn new() -> Self {
            Self {
                weight: Parameter::new(TensorData::new([2], vec![0.25, -0.5]).unwrap(), true),
                finish_visits: Cell::new(0),
                race_after_second_visit: Cell::new(false),
            }
        }

        fn arm_finish_race(&self) {
            self.finish_visits.set(0);
            self.race_after_second_visit.set(true);
        }
    }

    impl Module for FinishRaceModule {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            assert!(prefix.is_empty());
            let visit = self.finish_visits.get() + 1;
            self.finish_visits.set(visit);
            visitor("weight".into(), &self.weight, StateKind::Parameter);
            if self.race_after_second_visit.get() && visit == 2 {
                self.weight.replace(self.weight.value().unwrap()).unwrap();
                self.race_after_second_visit.set(false);
            }
        }
    }

    fn build_finish_race(
        module: &FinishRaceModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let weight = module.weight.bind(graph)?;
        let output = graph.mul(weight, inputs["x"])?;
        Ok((graph.sum_all(output)?, BTreeMap::new()))
    }

    struct FineTuneModule {
        base: Parameter,
        adapter: Parameter,
        frozen: Parameter,
    }

    impl FineTuneModule {
        fn new() -> Self {
            Self {
                base: Parameter::new(TensorData::new([2], vec![0.25, -0.5]).unwrap(), true),
                adapter: Parameter::new(TensorData::new([2], vec![0.1, 0.2]).unwrap(), true),
                frozen: Parameter::new(TensorData::new([2], vec![1.0, -1.0]).unwrap(), false),
            }
        }
    }

    impl Module for FineTuneModule {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            assert!(prefix.is_empty());
            visitor("base".into(), &self.base, StateKind::Parameter);
            visitor("base_alias".into(), &self.base, StateKind::Parameter);
            visitor("adapter".into(), &self.adapter, StateKind::Parameter);
            visitor("frozen".into(), &self.frozen, StateKind::Parameter);
        }
    }

    fn build_fine_tune(
        module: &FineTuneModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let base = module.base.bind(graph)?;
        assert_eq!(module.base.bind(graph)?, base);
        let adapter = module.adapter.bind(graph)?;
        let frozen = module.frozen.bind(graph)?;
        let scaled = graph.mul(inputs["x"], base)?;
        let adapted = graph.add(scaled, adapter)?;
        let output = graph.add(adapted, frozen)?;
        let squared = graph.square(output)?;
        let loss = graph.reduce(squared, crate::ReduceKind::Mean, None, false)?;
        Ok((loss, BTreeMap::from([("output".into(), output)])))
    }

    fn build_fine_tune_clip(
        module: &FineTuneModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let base = module.base.bind(graph)?;
        assert_eq!(module.base.bind(graph)?, base);
        let adapter = module.adapter.bind(graph)?;
        let scaled = graph.mul(inputs["x"], base)?;
        let output = graph.add(scaled, adapter)?;
        Ok((graph.mean_default(output)?, BTreeMap::new()))
    }

    fn assert_parameter_snapshot_eq(actual: &ParameterSnapshot, expected: &ParameterSnapshot) {
        assert_eq!(actual.data, expected.data);
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.dtype, expected.dtype);
        assert_eq!(actual.version, expected.version);
        assert_eq!(actual.identity, expected.identity);
        assert_eq!(actual.trainable, expected.trainable);
        assert_eq!(actual.input_name, expected.input_name);
    }

    fn build_tied_dropout(
        module: &TiedFrozenModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let dropped = dropout.dropout(graph, inputs["x"], 0.5)?;
        let shared = module.shared.bind(graph)?;
        let frozen = module.frozen.bind(graph)?;
        let scaled = graph.mul(dropped, shared)?;
        let output = graph.add(scaled, frozen)?;
        let squared = graph.square(output)?;
        let loss = graph.sum_all(squared)?;
        Ok((loss, BTreeMap::from([("output".into(), output)])))
    }

    fn build_double_dropout(
        module: &TiedFrozenModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let first = dropout.dropout(graph, inputs["x"], 0.5)?;
        let second = dropout.dropout(graph, first, 0.5)?;
        let shared = module.shared.bind(graph)?;
        let output = graph.mul(second, shared)?;
        let loss = graph.sum_all(output)?;
        Ok((loss, BTreeMap::from([("output".into(), output)])))
    }

    fn build_tied_dropout_with_input_guard(
        module: &TiedFrozenModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let (loss, outputs) = build_tied_dropout(module, graph, inputs, dropout)?;
        let reciprocal = graph.reciprocal(inputs["x"])?;
        let reciprocal_sum = graph.sum_all(reciprocal)?;
        Ok((graph.add(loss, reciprocal_sum)?, outputs))
    }

    #[test]
    fn compiled_dropout_effect_failure_retry_and_zero_grad_preserve_counter_contract() {
        let config = module_config().with_gradient_accumulation(2).unwrap();
        let key = CompiledDropoutConfig::new(CompiledDropoutKey([11, 13]));
        let module = TiedFrozenModule::new([0.1, -0.2]);
        let reference_module = TiedFrozenModule::new([0.1, -0.2]);
        let mut candidate = CompiledAdamWPlan::compile_module_with_dropout(
            config.clone(),
            key,
            &module,
            build_tied_dropout,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap();
        let mut reference = CompiledAdamWPlan::compile_module_with_dropout(
            config,
            key,
            &reference_module,
            build_tied_dropout,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap();
        let inputs =
            || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
        let before = candidate.checkpoint().unwrap();
        assert!(
            candidate
                .step_inner(inputs(), TensorData::scalar(0.01), Some(0))
                .is_err()
        );
        assert_eq!(candidate.dropout_block_counter().unwrap(), Some(0));
        assert_eq!(candidate.checkpoint().unwrap(), before);
        let expected = reference.step(inputs(), TensorData::scalar(0.01)).unwrap();
        let actual = candidate.step(inputs(), TensorData::scalar(0.01)).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(candidate.dropout_block_counter().unwrap(), Some(1));
        assert!(candidate.zero_grad().unwrap().did_discard());
        assert_eq!(candidate.dropout_block_counter().unwrap(), Some(1));
        assert_eq!(candidate.step_count(), 1);
    }

    #[test]
    fn compiled_dropout_checkpoint_v4_requires_exact_restore_policy_and_counter() {
        let key = CompiledDropoutConfig::new(CompiledDropoutKey([23, 29]));
        let module = TiedFrozenModule::new([0.1, -0.2]);
        let mut runtime = CompiledAdamWPlan::compile_module_with_dropout(
            module_config(),
            key,
            &module,
            build_tied_dropout,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap();
        runtime
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        let checkpoint = runtime.checkpoint().unwrap();
        let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V4);
        let info = checkpoint.info();
        assert_eq!(info.capture_identity(), runtime.capture_identity());
        assert_eq!(info.replay_step(), 1);
        assert_eq!(info.optimizer_step(), 1);
        assert_eq!(info.gradient_accumulation_steps(), 1);
        assert_eq!(info.accumulation_index(), 0);
        assert_eq!(info.discarded_microbatches(), 0);
        assert_eq!(info.flushed_window_count(), 0);
        assert_eq!(info.flushed_microbatch_count(), 0);
        assert_eq!(info.flush_capture_identity(), None);
        assert_eq!(info.dropout_block_counter(), Some(1));

        let fresh = TiedFrozenModule::new([0.1, -0.2]);
        assert!(
            CompiledAdamWPlan::compile_module_from_checkpoint(
                module_config(),
                &fresh,
                &checkpoint,
                build_tied_frozen,
            )
            .is_err()
        );
        assert!(
            CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
                module_config(),
                CompiledDropoutConfig::new(CompiledDropoutKey([23, 30])),
                &fresh,
                &checkpoint,
                build_tied_dropout,
            )
            .is_err()
        );
        assert!(
            CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
                module_config(),
                key,
                &fresh,
                &checkpoint,
                build_double_dropout,
            )
            .is_err()
        );

        let decoded = decode_adamw_checkpoint(checkpoint.as_bytes()).unwrap();
        let malformed = CompiledAdamWCheckpoint::from_bytes(
            encode_adamw_checkpoint(
                AdamWCheckpointProgress {
                    capture_identity: decoded.capture_identity,
                    replay_step: decoded.replay_step,
                    optimizer_step: decoded.optimizer_step,
                    accumulation_steps: decoded.accumulation_steps,
                    accumulation_index: decoded.accumulation_index,
                    discarded_microbatches: decoded.discarded_microbatches,
                    flushed_window_count: decoded.flushed_window_count,
                    flushed_microbatch_count: decoded.flushed_microbatch_count,
                    flush_capture_identity: decoded.flush_capture_identity,
                    dropout_block_counter: decoded.dropout_block_counter.map(|value| value + 1),
                    accumulated_token_count: decoded.accumulated_token_count,
                    window_loss_report: decoded.window_loss_report,
                    reset_transition_count: decoded.reset_transition_count,
                    reset_capture_identity: decoded.reset_capture_identity,
                },
                AdamWCheckpointTensors {
                    parameters: decoded.parameters,
                    first_moments: decoded.first_moments,
                    second_moments: decoded.second_moments,
                    gradient_accumulators: decoded.gradient_accumulators,
                    accumulated_loss_numerator: decoded.accumulated_loss_numerator,
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert!(
            CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
                module_config(),
                key,
                &fresh,
                &malformed,
                build_tied_dropout,
            )
            .is_err()
        );

        let ordinary = CpuCompiledAdamW::compile_module(module_config(), &fresh, build_tied_frozen)
            .unwrap()
            .checkpoint()
            .unwrap();
        let (_, metadata) = load_safetensors(ordinary.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V1);
        let ordinary_info = ordinary.info();
        assert_eq!(ordinary_info.replay_step(), 0);
        assert_eq!(ordinary_info.optimizer_step(), 0);
        assert_eq!(ordinary_info.gradient_accumulation_steps(), 1);
        assert_eq!(ordinary_info.accumulation_index(), 0);
        assert_eq!(ordinary_info.discarded_microbatches(), 0);
        assert_eq!(ordinary_info.flushed_window_count(), 0);
        assert_eq!(ordinary_info.flushed_microbatch_count(), 0);
        assert_eq!(ordinary_info.flush_capture_identity(), None);
        assert_eq!(ordinary_info.dropout_block_counter(), None);
        assert_eq!(ordinary_info.accumulated_token_count(), None);
        assert_eq!(ordinary_info.reset_transition_count(), 0);
        assert_eq!(ordinary_info.reset_capture_identity(), None);
        assert!(
            CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
                module_config(),
                key,
                &fresh,
                &ordinary,
                build_tied_dropout,
            )
            .is_err()
        );
        let mut accumulated = CpuCompiledAdamW::compile_module(
            module_config().with_gradient_accumulation(2).unwrap(),
            &fresh,
            build_tied_frozen,
        )
        .unwrap();
        accumulated
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        let v2 = accumulated.checkpoint().unwrap();
        accumulated.zero_grad().unwrap();
        let decoded =
            decode_adamw_checkpoint(accumulated.checkpoint().unwrap().as_bytes()).unwrap();
        let v3 = CompiledAdamWCheckpoint::from_bytes(
            encode_adamw_checkpoint(
                AdamWCheckpointProgress {
                    capture_identity: decoded.capture_identity,
                    replay_step: decoded.replay_step,
                    optimizer_step: decoded.optimizer_step,
                    accumulation_steps: decoded.accumulation_steps,
                    accumulation_index: decoded.accumulation_index,
                    discarded_microbatches: decoded.discarded_microbatches,
                    flushed_window_count: decoded.flushed_window_count,
                    flushed_microbatch_count: decoded.flushed_microbatch_count,
                    flush_capture_identity: decoded.flush_capture_identity,
                    dropout_block_counter: decoded.dropout_block_counter,
                    accumulated_token_count: decoded.accumulated_token_count,
                    window_loss_report: false,
                    reset_transition_count: 0,
                    reset_capture_identity: None,
                },
                AdamWCheckpointTensors {
                    parameters: decoded.parameters,
                    first_moments: decoded.first_moments,
                    second_moments: decoded.second_moments,
                    gradient_accumulators: decoded.gradient_accumulators,
                    accumulated_loss_numerator: None,
                },
            )
            .unwrap(),
        )
        .unwrap();
        for (legacy, format, accumulation_index, discarded_microbatches) in [
            (v2, ADAMW_CHECKPOINT_FORMAT_V2, 1, 0),
            (v3, ADAMW_CHECKPOINT_FORMAT_V3, 0, 1),
        ] {
            let (_, metadata) = load_safetensors(legacy.as_bytes()).unwrap();
            assert_eq!(metadata["format"], format);
            let info = legacy.info();
            assert_eq!(info.replay_step(), 1);
            assert_eq!(info.optimizer_step(), 0);
            assert_eq!(info.gradient_accumulation_steps(), 2);
            assert_eq!(info.accumulation_index(), accumulation_index);
            assert_eq!(info.discarded_microbatches(), discarded_microbatches);
            assert_eq!(info.reset_transition_count(), 0);
            assert_eq!(info.reset_capture_identity(), None);
            assert_eq!(info.flushed_window_count(), 0);
            assert_eq!(info.flushed_microbatch_count(), 0);
            assert_eq!(info.flush_capture_identity(), None);
            assert_eq!(info.dropout_block_counter(), None);
            assert_eq!(info.accumulated_token_count(), None);
            assert!(
                CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
                    module_config().with_gradient_accumulation(2).unwrap(),
                    key,
                    &fresh,
                    &legacy,
                    build_tied_dropout,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn compiled_dropout_rejects_counter_exhaustion_before_replay() {
        let module = TiedFrozenModule::new([0.1, -0.2]);
        let mut plan = CompiledAdamWPlan::compile_module_with_dropout(
            module_config(),
            CompiledDropoutConfig::new(CompiledDropoutKey([17, 19])),
            &module,
            build_double_dropout,
        )
        .unwrap();
        assert_eq!(plan.dropout_blocks_per_replay(), Some(2));
        let replay_step = u64::MAX / 2;
        let mut values = plan.inner.state_values.clone();
        values.insert(
            RecurrentStateKey::adamw_global(AdamWGlobalState::Step),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(replay_step)])
                .unwrap(),
        );
        values.insert(
            RecurrentStateKey::dropout_counter(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(replay_step * 2)])
                .unwrap(),
        );
        plan.inner = plan
            .inner
            .clone()
            .restore_frontier(replay_step, values)
            .unwrap();
        plan.progress = AdamWProgress {
            replay_step,
            optimizer_step: replay_step,
            accumulation_index: 0,
            discarded_microbatches: 0,
            flushed_window_count: 0,
            flushed_microbatch_count: 0,
            reset_transition_count: 0,
        };
        let mut runtime = plan.prepare_cpu().unwrap();
        let before = runtime.checkpoint().unwrap();
        assert!(
            runtime
                .step(
                    BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap(),)]),
                    TensorData::scalar(0.01),
                )
                .is_err()
        );
        assert_eq!(runtime.checkpoint().unwrap(), before);
    }

    fn module_config() -> CompiledAdamWConfig {
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_input("x", [2], DType::F32)
            .unwrap()
    }

    #[test]
    fn compiled_adamw_host_token_input_is_atomic_sorted_and_strict() {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_host_token_input("tokens_b", [2, 3])
            .unwrap()
            .with_host_token_input("tokens_a", [1, 2])
            .unwrap();
        assert_eq!(
            config
                .host_token_inputs()
                .map(|(name, shape)| (name, shape.dims()))
                .collect::<Vec<_>>(),
            [("tokens_a", &[1, 2][..]), ("tokens_b", &[2, 3][..])]
        );
        assert_eq!(
            config
                .inputs()
                .map(|(name, shape, dtype)| (name, shape.dims(), dtype))
                .collect::<Vec<_>>(),
            [
                ("tokens_a", &[1, 2][..], DType::I32),
                ("tokens_b", &[2, 3][..], DType::I32),
            ]
        );

        let base = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0).unwrap();
        assert!(
            base.clone()
                .with_input("tokens", [1, 2], DType::I32)
                .unwrap()
                .with_host_token_input("tokens", [1, 2])
                .is_err()
        );
        assert!(
            base.clone()
                .with_host_token_input("tokens", [1, 2])
                .unwrap()
                .with_input("tokens", [1, 2], DType::I32)
                .is_err()
        );
        for shape in [
            Shape::new([0, 2]),
            Shape::new([2, 0]),
            Shape::new([2]),
            Shape::new([1, 2, 1]),
            Shape::new([usize::MAX, 2]),
        ] {
            assert!(base.clone().with_host_token_input("tokens", shape).is_err());
        }
    }

    fn build_tied_frozen(
        module: &TiedFrozenModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let shared = module.shared.bind(graph)?;
        assert_eq!(module.shared.bind(graph)?, shared);
        assert_eq!(module.shared.node(graph)?, shared);
        let frozen = module.frozen.bind(graph)?;
        assert!(!graph.requires_grad(frozen)?);
        let scaled = graph.mul(inputs["x"], shared)?;
        let tied = graph.add(scaled, shared)?;
        let output = graph.add(tied, frozen)?;
        let squared = graph.square(output)?;
        let loss = graph.reduce(squared, crate::ReduceKind::Mean, None, false)?;
        Ok((loss, BTreeMap::from([("output".into(), output)])))
    }

    fn tied_token_mean_config() -> CompiledAdamWConfig {
        module_config()
            .with_input("mask", [2], DType::F32)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_token_weighted_gradient_accumulation("mask")
            .unwrap()
            .with_max_gradient_norm(0.25)
            .unwrap()
            .with_clip_report()
    }

    fn tied_token_mean_dropout() -> CompiledDropoutConfig {
        CompiledDropoutConfig::new(CompiledDropoutKey([71, 73]))
    }

    fn build_tied_frozen_token_mean(
        module: &TiedFrozenModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<CompiledAdamWGraph> {
        let shared = module.shared.bind(graph)?;
        assert_eq!(module.shared.bind(graph)?, shared);
        let frozen = module.frozen.bind(graph)?;
        assert!(!graph.requires_grad(frozen)?);
        let scaled = graph.mul(inputs["x"], shared)?;
        let tied = graph.add(scaled, shared)?;
        let output = graph.add(tied, frozen)?;
        let dropped = dropout.dropout(graph, output, 0.25)?;
        let losses = graph.square(dropped)?;
        Ok(CompiledAdamWGraph::token_mean(
            losses,
            BTreeMap::from([("output".into(), dropped)]),
        ))
    }

    fn tied_token_mean_batch(x: [f32; 2], mask: [f32; 2]) -> BTreeMap<String, TensorData> {
        BTreeMap::from([
            ("mask".into(), TensorData::new([2], mask.to_vec()).unwrap()),
            ("x".into(), TensorData::new([2], x.to_vec()).unwrap()),
        ])
    }

    fn initial_parameters() -> Vec<TrainingParameterInit> {
        vec![
            TrainingParameterInit::new(
                "w1",
                TensorData::new(
                    [2, 4],
                    vec![0.20, -0.10, 0.05, 0.30, -0.25, 0.15, 0.40, -0.20],
                )
                .unwrap(),
            )
            .unwrap(),
            TrainingParameterInit::new(
                "w2",
                TensorData::new(
                    [4, 2],
                    vec![0.10, -0.20, 0.30, 0.05, -0.15, 0.25, 0.20, -0.10],
                )
                .unwrap(),
            )
            .unwrap(),
        ]
    }

    fn config() -> CompiledMomentumSgdConfig {
        CompiledMomentumSgdConfig::new(0.9)
            .unwrap()
            .with_input("x", [4, 2], DType::F32)
            .unwrap()
            .with_input("target", [4], DType::I64)
            .unwrap()
    }

    fn build_tinybob(
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        parameters: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let hidden = graph.matmul(inputs["x"], parameters["w1"])?;
        let hidden = graph.relu(hidden)?;
        let logits = graph.matmul(hidden, parameters["w2"])?;
        let loss = cross_entropy(graph, logits, inputs["target"], LossOptions::default())?;
        Ok((loss, BTreeMap::from([("logits".into(), logits)])))
    }

    fn compiled() -> CpuCompiledMomentumSgd {
        CpuCompiledMomentumSgd::compile(config(), initial_parameters(), build_tinybob).unwrap()
    }

    fn adamw_config() -> CompiledAdamWConfig {
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)
            .unwrap()
            .with_input_batch::<TinyBobBatch>()
            .unwrap()
    }

    fn accumulated_adamw_config(steps: u64) -> CompiledAdamWConfig {
        adamw_config().with_gradient_accumulation(steps).unwrap()
    }

    fn compiled_adamw() -> CpuCompiledAdamW {
        CpuCompiledAdamW::compile(adamw_config(), initial_parameters(), build_tinybob).unwrap()
    }

    fn non_finite_config(accumulation_steps: u64) -> CompiledAdamWConfig {
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(accumulation_steps)
            .unwrap()
            .with_input("x", [], DType::F32)
            .unwrap()
    }

    fn non_finite_plan(accumulation_steps: u64) -> CompiledAdamWPlan {
        let parameter = TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap();
        CompiledAdamWPlan::compile(
            non_finite_config(accumulation_steps),
            [parameter],
            |graph, inputs, parameters| {
                let radicand = graph.add(parameters["weight"], inputs["x"])?;
                let loss = graph.sqrt(radicand)?;
                Ok((loss, BTreeMap::new()))
            },
        )
        .unwrap()
    }

    fn non_finite_flush_plan() -> CompiledAdamWPlan {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 2.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap()
            .with_clip_report()
            .with_input("x", [], DType::F32)
            .unwrap();
        let parameter = TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap();
        CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
            let loss = graph.mul(parameters["weight"], inputs["x"])?;
            Ok((loss, BTreeMap::new()))
        })
        .unwrap()
    }

    fn scalar_batch(value: f32) -> BTreeMap<String, TensorData> {
        BTreeMap::from([("x".into(), TensorData::scalar(value))])
    }

    fn rejecting_cpu_target() -> ConfiguredCpuSessionTarget {
        CpuSessionTarget.with_non_finite_policy(CpuNonFinitePolicy::RejectTransition)
    }

    #[test]
    fn cpu_non_finite_validation_admits_empty_signed_zero_and_subnormal() {
        let empty = TensorData::new([0], Vec::<f32>::new()).unwrap();
        let finite = TensorData::new([2], vec![-0.0, f32::from_bits(1)]).unwrap();
        assert!(validate_finite_tensors([&empty, &finite], "fixture").is_ok());
        assert!(validate_finite_tensors([&TensorData::scalar(f32::INFINITY)], "fixture",).is_err());
    }

    #[test]
    fn clip_report_extraction_preserves_exact_non_finite_f32_bits() {
        let norm_bits = 0x7f80_0123;
        let scale_bits = 0x7fc0_4567;
        let mut values = [
            TensorData::scalar(f32::from_bits(norm_bits)),
            TensorData::scalar(f32::from_bits(scale_bits)),
        ]
        .into_iter();
        let report = take_compiled_clip_report(&mut values, true).unwrap();
        assert_eq!(report.pre_clip_global_norm().to_bits(), norm_bits);
        assert_eq!(report.applied_scale().to_bits(), scale_bits);
        assert!(!report.is_finite());
        assert_eq!(report.did_clip(), None);
        assert!(values.next().is_none());
    }

    #[test]
    fn window_loss_report_extraction_preserves_exact_f32_bits() {
        let loss_bits = 0x7fc0_4567;
        let mut values = [
            TensorData::scalar(f32::from_bits(loss_bits)),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(11)]).unwrap(),
        ]
        .into_iter();
        let value = take_compiled_window_loss_value(&mut values, true).unwrap();
        let report = CompiledAdamWWindowLossReport::new(value, 3);
        assert_eq!(report.mean_loss().to_bits(), loss_bits);
        assert_eq!(report.loss_weight(), 11);
        assert_eq!(report.microbatch_count(), 3);
        assert!(!report.is_finite());
        assert!(values.next().is_none());
    }

    #[test]
    fn global_norm_commits_through_stacked_typed_f32_sum() {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_clip_report();
        let mut graph = Graph::new();
        let left = graph.input_dtype("left", [2], DType::F32);
        let clipped = clip_gradients_by_global_norm(
            &config,
            &mut graph,
            &BTreeMap::from([("left".into(), left)]),
        )
        .unwrap();
        let committed_gradient = clipped.gradients["left"];
        let gradient_owner = crate::rangeify::computed_view(&graph, committed_gradient)
            .unwrap()
            .source;
        assert!(matches!(
            graph.op(gradient_owner).unwrap(),
            Op::Concat { axis: 0, .. }
        ));
        let norm = clipped.report.unwrap().pre_clip_global_norm;
        let Op::Unary {
            op: crate::UnaryOp::Sqrt,
            input: committed,
        } = graph.op(norm).unwrap()
        else {
            panic!("global norm must end in sqrt");
        };
        let Op::Reduce {
            input: stacked,
            kind: crate::ReduceKind::Sum,
            accumulator: DType::F32,
            ..
        } = graph.op(*committed).unwrap()
        else {
            panic!("global norm must consume a typed F32 sum");
        };
        assert_eq!(graph.dtype(*committed).unwrap(), DType::F32);
        assert!(matches!(
            graph.op(*stacked).unwrap(),
            Op::Concat { axis: 0, .. }
        ));
        let Op::Concat { inputs, .. } = graph.op(*stacked).unwrap() else {
            unreachable!();
        };
        let parameter_sum = crate::rangeify::computed_view(&graph, inputs[0])
            .unwrap()
            .source;
        let Op::Reduce { input: squared, .. } = graph.op(parameter_sum).unwrap() else {
            panic!("stacked norm input must be one parameter reduction");
        };
        assert!(matches!(
            graph.op(*squared).unwrap(),
            Op::Binary {
                op: crate::BinaryOp::Mul,
                lhs,
                rhs,
            } if lhs == &committed_gradient && rhs == &committed_gradient
        ));
    }

    #[test]
    fn guarded_cpu_preparation_rejects_non_finite_initial_and_restored_frontiers() {
        let initial = CompiledAdamWPlan::compile(
            non_finite_config(1),
            [TrainingParameterInit::new("weight", TensorData::scalar(f32::INFINITY)).unwrap()],
            |graph, _, parameters| {
                let loss = graph.square(parameters["weight"])?;
                Ok((loss, BTreeMap::new()))
            },
        )
        .unwrap();
        let error = initial
            .prepare(&rejecting_cpu_target())
            .err()
            .expect("guarded preparation must reject a non-finite initial frontier");
        assert!(
            error
                .to_string()
                .contains("non-finite prepared recurrent state")
        );

        let plan = non_finite_plan(1);
        let mut propagating = plan.prepare(&CpuSessionTarget).unwrap();
        propagating
            .step(scalar_batch(0.0), TensorData::scalar(0.01))
            .unwrap();
        let restored = plan
            .restore_checkpoint(&propagating.checkpoint().unwrap())
            .unwrap();
        let error = restored
            .prepare(&rejecting_cpu_target())
            .err()
            .expect("guarded preparation must reject a non-finite restored frontier");
        assert!(
            error
                .to_string()
                .contains("non-finite prepared recurrent state")
        );
    }

    #[test]
    fn cpu_non_finite_policy_preserves_default_identity_and_rejects_before_commit() {
        let plan = non_finite_plan(1);
        let legacy = plan.prepare(&CpuSessionTarget).unwrap();
        let explicit = plan
            .prepare(&CpuSessionTarget.with_non_finite_policy(CpuNonFinitePolicy::Propagate))
            .unwrap();
        let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
        assert_eq!(legacy.capture_identity(), plan.capture_identity());
        assert_eq!(explicit.capture_identity(), plan.capture_identity());
        assert_eq!(legacy.checkpoint().unwrap(), explicit.checkpoint().unwrap());
        assert_eq!(legacy.checkpoint().unwrap(), guarded.checkpoint().unwrap());
        assert_eq!(legacy.non_finite_policy(), CpuNonFinitePolicy::Propagate);
        assert_eq!(
            guarded.non_finite_policy(),
            CpuNonFinitePolicy::RejectTransition
        );

        let before = guarded.checkpoint().unwrap();
        let error = guarded
            .step(scalar_batch(0.0), TensorData::scalar(0.01))
            .unwrap_err();
        assert!(error.to_string().contains("non-finite recurrent successor"));
        assert_eq!(guarded.checkpoint().unwrap(), before);
        assert_eq!(guarded.step_count(), 0);

        for invalid_rate in [f32::NAN, f32::INFINITY] {
            let error = guarded
                .step(scalar_batch(1.0), TensorData::scalar(invalid_rate))
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("non-finite external learning rate")
            );
            assert_eq!(guarded.checkpoint().unwrap(), before);
        }
        let mut reference = plan.prepare(&rejecting_cpu_target()).unwrap();
        let expected = reference
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        let actual = guarded
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(
            guarded.checkpoint().unwrap(),
            reference.checkpoint().unwrap()
        );
    }

    #[test]
    fn cpu_non_finite_policy_checks_loss_but_not_named_outputs() {
        let parameter = TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap();
        let plan = CompiledAdamWPlan::compile(
            non_finite_config(1),
            [parameter],
            |graph, inputs, parameters| {
                let loss = graph.square(parameters["weight"])?;
                let one = scalar_f32(graph, 1.0)?;
                let diagnostic = graph.div(one, inputs["x"])?;
                Ok((loss, BTreeMap::from([("diagnostic".into(), diagnostic)])))
            },
        )
        .unwrap();
        let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
        let result = guarded
            .step(scalar_batch(0.0), TensorData::scalar(0.01))
            .unwrap();
        assert!(result.output("diagnostic").unwrap().values()[0].is_infinite());

        let mut guarded = non_finite_plan(1).prepare(&rejecting_cpu_target()).unwrap();
        let before = guarded.checkpoint().unwrap();
        let error = guarded
            .step(scalar_batch(f32::NAN), TensorData::scalar(0.01))
            .unwrap_err();
        assert!(error.to_string().contains("non-finite loss"));
        assert_eq!(guarded.checkpoint().unwrap(), before);
    }

    #[test]
    fn non_finite_clip_report_rejects_atomically_and_retries_on_both_cpu_paths() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap()
            .with_clip_report()
            .with_input("x", [], DType::F32)
            .unwrap();
        let plan = CompiledAdamWPlan::compile(
            config,
            [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
            |graph, inputs, parameters| {
                Ok((
                    graph.mul(parameters["weight"], inputs["x"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap();
        let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
        let before = interpreted.checkpoint().unwrap();
        let error = interpreted
            .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
            .expect_err("non-finite interpreted clip report must reject");
        assert!(error.to_string().contains("non-finite clip report"));
        assert_eq!(interpreted.checkpoint().unwrap(), before);
        assert_eq!(interpreted.step_count(), 0);

        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor)
            .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
        let mut native = target.prepare(&plan).unwrap();
        let error = match native.step(scalar_batch(f32::MAX), TensorData::scalar(0.01)) {
            Ok(_) => panic!("non-finite native clip report must reject"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-finite clip report"));
        assert_eq!(native.checkpoint().unwrap(), before);
        assert_eq!(native.successful_steps, 0);

        let interpreted_step = interpreted
            .step(scalar_batch(3.0), TensorData::scalar(0.01))
            .unwrap();
        let native_step = native
            .step(scalar_batch(3.0), TensorData::scalar(0.01))
            .unwrap();
        let report = interpreted_step.clip_report().unwrap();
        assert_eq!(report.pre_clip_global_norm(), 3.0);
        assert_eq!(report.applied_scale(), 1.0 / 3.0);
        assert!(report.is_finite());
        assert_eq!(report.did_clip(), Some(true));
        assert_eq!(native_step.clip_report(), Some(report));
        assert_eq!(native.successful_steps, 1);
        assert_eq!(
            native.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );
    }

    #[test]
    fn unreported_global_norm_overflow_matches_both_cpu_paths() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap()
            .with_input("x", [], DType::F32)
            .unwrap();
        let plan = CompiledAdamWPlan::compile(
            config,
            [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
            |graph, inputs, parameters| {
                Ok((
                    graph.mul(parameters["weight"], inputs["x"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap();
        let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor)
            .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
        let mut native = target.prepare(&plan).unwrap();

        // MAX is finite: its F32 squared norm overflows, scale becomes zero,
        // and the finite gradient clips to zero. Both final frontiers are valid.
        let interpreted_step = interpreted
            .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
            .unwrap();
        let native_step = native
            .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
            .unwrap();
        assert!(interpreted_step.clip_report().is_none());
        assert!(native_step.clip_report().is_none());
        assert_eq!(interpreted_step.optimizer_step(), 1);
        assert_eq!(native_step.optimizer_step(), 1);
        assert_eq!(native.successful_steps, 1);
        assert_eq!(
            native.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );
    }

    #[test]
    fn accumulation_only_clip_report_overflow_is_discarded_before_safe_commit() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap()
            .with_clip_report()
            .with_input("x", [], DType::F32)
            .unwrap();
        let plan = CompiledAdamWPlan::compile(
            config,
            [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
            |graph, inputs, parameters| {
                Ok((
                    graph.mul(parameters["weight"], inputs["x"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap();
        let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor)
            .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
        let mut native = target.prepare(&plan).unwrap();

        let interpreted_first = interpreted
            .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
            .unwrap();
        let native_first = native
            .step(scalar_batch(f32::MAX), TensorData::scalar(0.01))
            .unwrap();
        assert!(interpreted_first.clip_report().is_none());
        assert!(native_first.clip_report().is_none());
        assert_eq!(interpreted_first.accumulation_index(), 1);
        assert_eq!(native_first.accumulation_index(), 1);
        assert_eq!(
            native.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );

        let interpreted_commit = interpreted
            .step(scalar_batch(-f32::MAX), TensorData::scalar(0.01))
            .unwrap();
        let native_commit = native
            .step(scalar_batch(-f32::MAX), TensorData::scalar(0.01))
            .unwrap();
        let report = interpreted_commit.clip_report().unwrap();
        assert_eq!(report.pre_clip_global_norm(), 0.0);
        assert_eq!(report.applied_scale(), 1.0);
        assert!(report.is_finite());
        assert_eq!(report.did_clip(), Some(false));
        assert_eq!(native_commit.clip_report(), Some(report));
        assert_eq!(interpreted_commit.optimizer_step(), 1);
        assert_eq!(native_commit.optimizer_step(), 1);
        assert_eq!(native.successful_steps, 2);
        assert_eq!(
            native.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );
    }

    #[test]
    fn cpu_non_finite_rejection_preserves_partial_dropout_window_and_flush() {
        let config = module_config().with_gradient_accumulation(2).unwrap();
        let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([41, 43]));
        let module = TiedFrozenModule::new([0.1, -0.2]);
        let plan = CompiledAdamWPlan::compile_module_with_dropout(
            config,
            dropout,
            &module,
            build_tied_dropout_with_input_guard,
        )
        .unwrap();
        let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
        let mut reference = plan.prepare(&rejecting_cpu_target()).unwrap();
        let finite =
            || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
        guarded.step(finite(), TensorData::scalar(0.01)).unwrap();
        reference.step(finite(), TensorData::scalar(0.01)).unwrap();
        let partial = guarded.checkpoint().unwrap();
        assert_eq!(guarded.dropout_block_counter().unwrap(), Some(1));
        let error = guarded
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![0.0, 2.0]).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap_err();
        assert!(error.to_string().contains("non-finite loss"));
        assert_eq!(guarded.checkpoint().unwrap(), partial);
        assert_eq!(guarded.dropout_block_counter().unwrap(), Some(1));
        let expected = reference.step(finite(), TensorData::scalar(0.01)).unwrap();
        let actual = guarded.step(finite(), TensorData::scalar(0.01)).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(
            guarded.checkpoint().unwrap(),
            reference.checkpoint().unwrap()
        );

        let flush_plan = CompiledAdamWPlan::compile(
            accumulated_adamw_config(2),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        let mut guarded = flush_plan.prepare(&rejecting_cpu_target()).unwrap();
        let mut reference = flush_plan.prepare(&rejecting_cpu_target()).unwrap();
        guarded.step(batch(), lr()).unwrap();
        reference.step(batch(), lr()).unwrap();
        let partial = guarded.checkpoint().unwrap();
        assert!(
            guarded
                .flush_partial_window(TensorData::scalar(f32::NAN))
                .is_err()
        );
        assert_eq!(guarded.checkpoint().unwrap(), partial);
        assert_eq!(
            guarded.flush_partial_window(lr()).unwrap().optimizer_step(),
            1
        );
        reference.flush_partial_window(lr()).unwrap();
        assert_eq!(
            guarded.checkpoint().unwrap(),
            reference.checkpoint().unwrap()
        );
    }

    #[test]
    fn non_finite_partial_flush_successors_reject_atomically_on_both_cpu_paths() {
        let plan = non_finite_flush_plan();
        let mut guarded = plan.prepare(&rejecting_cpu_target()).unwrap();
        let mut reference = plan.prepare(&rejecting_cpu_target()).unwrap();
        guarded
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        reference
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        let before = guarded.checkpoint().unwrap();
        let error = guarded
            .flush_partial_window(TensorData::scalar(f32::MAX))
            .unwrap_err();
        assert!(error.to_string().contains("non-finite recurrent successor"));
        assert_eq!(guarded.checkpoint().unwrap(), before);
        assert_eq!(guarded.optimizer_step().unwrap(), 0);
        assert_eq!(guarded.accumulation_index().unwrap(), 1);
        let actual = guarded
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap();
        let expected = reference
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.clip_report(), expected.clip_report());
        assert!(actual.clip_report().is_some());
        assert_eq!(
            guarded.checkpoint().unwrap(),
            reference.checkpoint().unwrap()
        );

        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor)
            .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
        let mut native = target.prepare(&plan).unwrap();
        let mut native_reference = target.prepare(&plan).unwrap();
        native
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        native_reference
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        let before = native.checkpoint().unwrap();
        let error = native
            .flush_partial_window(TensorData::scalar(f32::MAX))
            .err()
            .expect("non-finite native partial flush must reject");
        assert!(error.to_string().contains("non-finite recurrent successor"));
        assert_eq!(native.checkpoint().unwrap(), before);
        assert_eq!(native.inner.optimizer_step().unwrap(), 0);
        assert_eq!(native.inner.accumulation_index().unwrap(), 1);
        assert_eq!(native.successful_flushes, 0);
        let actual = native
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap();
        let expected = native_reference
            .flush_partial_window(TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(
            actual.flushed_microbatches(),
            expected.flushed_microbatches()
        );
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.clip_report(), expected.clip_report());
        assert!(actual.clip_report().is_some());
        assert_eq!(actual.report().unwrap().successful_invocation(), 1);
        assert_eq!(native.successful_flushes, 1);
        assert_eq!(
            native.checkpoint().unwrap(),
            native_reference.checkpoint().unwrap()
        );
    }

    #[test]
    fn native_cpu_non_finite_rejection_is_atomic_and_retryable() {
        let plan = non_finite_plan(1);
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor)
            .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
        let mut native = target.prepare(&plan).unwrap();
        assert_eq!(
            native.non_finite_policy(),
            CpuNonFinitePolicy::RejectTransition
        );
        let prepared_workspace = native.main_replay.workspace_stats();
        assert_eq!(prepared_workspace.borrowed_external_input_bytes, 0);
        let before = native.checkpoint().unwrap();
        assert!(
            native
                .step(scalar_batch(0.0), TensorData::scalar(0.01))
                .is_err()
        );
        assert_eq!(native.checkpoint().unwrap(), before);
        assert_eq!(native.successful_steps, 0);
        let rejected_workspace = native.main_replay.workspace_stats();
        assert_eq!(rejected_workspace.input_import_count, 0);
        assert_eq!(rejected_workspace.borrowed_external_input_bytes, 8);
        assert!(
            native
                .step(scalar_batch(1.0), TensorData::scalar(f32::INFINITY))
                .is_err()
        );
        assert_eq!(native.checkpoint().unwrap(), before);
        assert_eq!(native.successful_steps, 0);
        assert_eq!(native.main_replay.workspace_stats(), rejected_workspace);

        let mut interpreted = plan.prepare(&rejecting_cpu_target()).unwrap();
        let expected = interpreted
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        let actual = native
            .step(scalar_batch(1.0), TensorData::scalar(0.01))
            .unwrap();
        assert_cross_engine_tensor_close("guarded retry loss", actual.loss(), expected.loss());
        assert_eq!(actual.report().successful_invocation(), 1);
        assert_eq!(actual.report().traffic().external_input_import_count(), 0);
        assert_eq!(actual.report().traffic().external_input_import_bytes(), 0);
        assert_eq!(native.successful_steps, 1);
        assert_native_adamw_state_close(&native, &interpreted);
    }

    fn assert_cross_engine_tensor_close(label: &str, actual: &TensorData, expected: &TensorData) {
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        assert_eq!(actual.dtype(), expected.dtype(), "{label} dtype");
        assert_eq!(actual.dtype(), DType::F32, "{label} comparison dtype");
        for index in 0..actual.len() {
            let actual = actual.scalar_at(index).as_f64();
            let expected = expected.scalar_at(index).as_f64();
            let error = (actual - expected).abs();
            assert!(
                error <= 1e-5,
                "{label}[{index}] mismatch: actual={actual}, expected={expected}, error={error}"
            );
        }
    }

    fn assert_cross_engine_tensor_maps_close(
        label: &str,
        actual: &BTreeMap<String, TensorData>,
        expected: &BTreeMap<String, TensorData>,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} key count");
        for (name, expected) in expected {
            let actual = actual
                .get(name)
                .unwrap_or_else(|| panic!("{label} missing {name}"));
            assert_cross_engine_tensor_close(&format!("{label} {name}"), actual, expected);
        }
    }

    fn assert_native_adamw_state_close(
        native: &NativeCpuCompiledAdamW<'_>,
        interpreted: &CpuCompiledAdamW,
    ) {
        assert_cross_engine_tensor_maps_close(
            "parameters",
            &native.parameter_snapshots().unwrap(),
            &interpreted.parameter_snapshots().unwrap(),
        );
        assert_cross_engine_tensor_maps_close(
            "first moments",
            &native.first_moment_snapshots().unwrap(),
            &interpreted.first_moment_snapshots().unwrap(),
        );
        assert_cross_engine_tensor_maps_close(
            "second moments",
            &native.second_moment_snapshots().unwrap(),
            &interpreted.second_moment_snapshots().unwrap(),
        );
        assert_cross_engine_tensor_maps_close(
            "gradient accumulators",
            &native.gradient_accumulator_snapshots().unwrap(),
            &interpreted.gradient_accumulator_snapshots().unwrap(),
        );
        let native_checkpoint = native.checkpoint().unwrap();
        let interpreted_checkpoint = interpreted.checkpoint().unwrap();
        assert_eq!(native_checkpoint.info(), interpreted_checkpoint.info());
    }

    fn native_recurrent_test_counts(native: &NativeCpuCompiledAdamW<'_>) -> (usize, usize) {
        native.inner.inner.runtime.recurrent_test_counts()
    }

    #[test]
    fn native_cpu_adamw_prepares_strictly_reuses_cache_and_commits_atomically() {
        let plan = CompiledAdamWPlan::compile(adamw_config(), initial_parameters(), build_tinybob)
            .unwrap();
        let inspection = plan.inspection().unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
        let mut native = target.prepare(&plan).unwrap();
        let mut scoreboard = crate::NativeTrainingScoreboard::new(
            inspection,
            native.preparation_report(),
            Duration::ZERO,
            Duration::ZERO,
        )
        .unwrap();
        assert_eq!(executor.native_item_plan_count(), 1);
        let workspace = native.main_replay.workspace_stats();
        assert!(workspace.allocation_count > 0);
        assert_eq!(workspace.input_import_count, 0);
        assert_eq!(workspace.intermediate_materialization_count, 0);
        assert_eq!(workspace.borrowed_recurrent_input_bytes, 0);
        assert_eq!(workspace.borrowed_recurrent_output_bytes, 0);
        let preparation = native.preparation_report();
        assert_eq!(
            preparation.main().capture_identity(),
            plan.capture_identity()
        );
        assert!(preparation.main().native_item_count() > 0);
        assert_eq!(
            preparation.main().cache_hit_count() + preparation.main().cache_miss_count(),
            preparation.main().native_item_count()
        );
        assert!(preparation.main().cache_miss_count() > 0);
        assert!(preparation.partial_flush().is_none());
        assert!(preparation.zero_grad().is_none());
        assert!(preparation.evaluation().is_none());
        assert!(preparation.recurrent_state_count() > 0);
        assert!(preparation.recurrent_state_bytes() > 0);
        let recurrent_state_bytes = preparation.recurrent_state_bytes();
        let prepared_native_identity = preparation.main().native_identity();

        let cached = target.prepare(&plan).unwrap();
        assert_eq!(executor.native_item_plan_count(), 2);
        assert_eq!(cached.preparation_report().main().cache_miss_count(), 0);
        assert_eq!(
            cached.preparation_report().main().native_identity(),
            prepared_native_identity
        );

        let mut interpreted = plan.prepare_cpu().unwrap();
        let expected = interpreted.step(batch(), lr()).unwrap();
        let before_replay = native_recurrent_test_counts(&native);
        let actual = native.step(batch(), lr()).unwrap();
        let after_replay = native_recurrent_test_counts(&native);
        assert_eq!(after_replay.0, before_replay.0);
        assert_eq!(after_replay.1, before_replay.1 + 1);
        assert_cross_engine_tensor_close("first loss", actual.loss(), expected.loss());
        assert_cross_engine_tensor_maps_close(
            "first outputs",
            actual.outputs(),
            expected.outputs(),
        );
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(expected.loss_weight(), 1);
        assert_eq!(actual.loss_weight(), 1);
        assert_eq!(actual.report().successful_invocation(), 1);
        assert!(actual.report().first_successful_invocation());
        assert_eq!(actual.report().native_identity(), prepared_native_identity);
        let first_traffic = *actual.report().traffic();
        assert_eq!(first_traffic.external_input_import_count(), 1);
        assert_eq!(first_traffic.external_input_import_bytes(), 32);
        assert_eq!(
            first_traffic.borrowed_recurrent_input_bytes(),
            u64::try_from(recurrent_state_bytes).unwrap()
        );
        assert_eq!(
            first_traffic.borrowed_recurrent_output_bytes(),
            u64::try_from(recurrent_state_bytes).unwrap()
        );
        let mut malformed_report = actual.report().clone();
        malformed_report.traffic.borrowed_recurrent_output_bytes -= 1;
        assert!(scoreboard.record(&malformed_report).is_err());
        scoreboard.record(actual.report()).unwrap();
        assert_native_adamw_state_close(&native, &interpreted);
        let first_workspace = native.main_replay.workspace_stats();
        assert_eq!(first_workspace.allocation_count, workspace.allocation_count);
        assert_eq!(first_workspace.input_import_count, 1);
        assert_eq!(first_workspace.borrowed_external_input_bytes, 36);
        assert_eq!(first_workspace.intermediate_materialization_count, 0);
        assert_eq!(
            first_workspace.borrowed_recurrent_input_bytes,
            recurrent_state_bytes
        );
        assert_eq!(
            first_workspace.borrowed_recurrent_output_bytes,
            recurrent_state_bytes
        );

        let before_failure = native.checkpoint().unwrap();
        let before_failure_counts = native_recurrent_test_counts(&native);
        assert!(native.step_inner(batch(), lr(), Some(0)).is_err());
        let after_failure_counts = native_recurrent_test_counts(&native);
        assert_eq!(after_failure_counts.0, before_failure_counts.0);
        assert_eq!(after_failure_counts.1, before_failure_counts.1);
        assert_eq!(native.checkpoint().unwrap(), before_failure);
        let failed_workspace = native.main_replay.workspace_stats();
        assert_eq!(
            failed_workspace.allocation_count,
            workspace.allocation_count
        );
        assert_eq!(failed_workspace.intermediate_materialization_count, 0);
        assert_eq!(failed_workspace.input_import_count, 2);
        assert_eq!(failed_workspace.borrowed_external_input_bytes, 72);
        assert_eq!(
            failed_workspace.borrowed_recurrent_input_bytes,
            first_workspace.borrowed_recurrent_input_bytes + recurrent_state_bytes
        );
        assert_eq!(
            failed_workspace.borrowed_recurrent_output_bytes,
            first_workspace.borrowed_recurrent_output_bytes + recurrent_state_bytes
        );
        let expected = interpreted.step(batch(), lr()).unwrap();
        let before_retry = native_recurrent_test_counts(&native);
        let actual = native.step(batch(), lr()).unwrap();
        let after_retry = native_recurrent_test_counts(&native);
        assert_eq!(after_retry.0, before_retry.0);
        assert_eq!(after_retry.1, before_retry.1 + 1);
        assert_cross_engine_tensor_close("retry loss", actual.loss(), expected.loss());
        assert_cross_engine_tensor_maps_close(
            "retry outputs",
            actual.outputs(),
            expected.outputs(),
        );
        assert_eq!(expected.loss_weight(), 1);
        assert_eq!(actual.loss_weight(), 1);
        assert_eq!(actual.report().successful_invocation(), 2);
        assert_eq!(actual.report().traffic(), &first_traffic);
        scoreboard.record(actual.report()).unwrap();
        assert_eq!(
            scoreboard.report().unwrap().main_replay_traffic().unwrap(),
            &first_traffic
        );
        assert_native_adamw_state_close(&native, &interpreted);
        assert_eq!(executor.native_item_plan_count(), 2);
        let retried_workspace = native.main_replay.workspace_stats();
        assert_eq!(
            retried_workspace.allocation_count,
            workspace.allocation_count
        );
        assert_eq!(retried_workspace.input_import_count, 3);
        assert_eq!(retried_workspace.borrowed_external_input_bytes, 108);
        assert_eq!(retried_workspace.intermediate_materialization_count, 0);
        assert_eq!(
            retried_workspace.borrowed_recurrent_input_bytes,
            failed_workspace.borrowed_recurrent_input_bytes + recurrent_state_bytes
        );
        assert_eq!(
            retried_workspace.borrowed_recurrent_output_bytes,
            failed_workspace.borrowed_recurrent_output_bytes + recurrent_state_bytes
        );
    }

    #[test]
    fn native_cpu_adamw_retains_transposed_parameter_storage() {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_input("x", [2, 3], DType::F32)
            .unwrap()
            .with_input("y", [2, 3], DType::F32)
            .unwrap();
        let parameter = TrainingParameterInit::new(
            "weight",
            TensorData::new(
                [4, 3],
                vec![
                    0.2, -0.1, 0.3, 0.4, 0.05, -0.2, -0.3, 0.25, 0.1, 0.15, -0.4, 0.35,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
            let weight = graph.permute(parameters["weight"], [1, 0])?;
            let output = graph.matmul(inputs["x"], weight)?;
            let second = graph.matmul(inputs["y"], weight)?;
            let combined = graph.add(output, second)?;
            let loss = graph.sum_all(combined)?;
            Ok((loss, BTreeMap::from([("output".into(), output)])))
        })
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut native = target.prepare(&plan).unwrap();
        let mut interpreted = plan.prepare_cpu().unwrap();
        let prepared = native.main_replay.workspace_stats();
        assert!(prepared.retained_transpose_matmul_input_count >= 2);
        assert_eq!(prepared.affine_matmul_materialization_bytes, 0);
        assert_eq!(native.preparation_report().main().fallback_count(), 0);

        let batch = BTreeMap::from([
            (
                "x".into(),
                TensorData::new([2, 3], vec![1.0, -0.5, 0.25, -1.0, 0.75, 0.5]).unwrap(),
            ),
            (
                "y".into(),
                TensorData::new([2, 3], vec![0.5, 0.25, -1.0, 0.75, -0.5, 1.0]).unwrap(),
            ),
        ]);
        let before = native.checkpoint().unwrap();
        assert!(
            native
                .step_inner(batch.clone(), TensorData::scalar(0.01), Some(0))
                .is_err()
        );
        assert_eq!(native.checkpoint().unwrap(), before);
        assert_eq!(
            native.main_replay.workspace_stats().allocation_count,
            prepared.allocation_count
        );
        assert_eq!(
            native
                .main_replay
                .workspace_stats()
                .affine_matmul_materialization_bytes,
            0
        );

        let expected = interpreted
            .step(batch.clone(), TensorData::scalar(0.01))
            .unwrap();
        let actual = native
            .step(batch.clone(), TensorData::scalar(0.01))
            .unwrap();
        assert_cross_engine_tensor_close("retained transpose loss", actual.loss(), expected.loss());
        assert_cross_engine_tensor_maps_close(
            "retained transpose output",
            actual.outputs(),
            expected.outputs(),
        );
        let expected = interpreted
            .step(batch.clone(), TensorData::scalar(0.01))
            .unwrap();
        let actual = native.step(batch, TensorData::scalar(0.01)).unwrap();
        assert_cross_engine_tensor_close(
            "retained transpose update loss",
            actual.loss(),
            expected.loss(),
        );
        assert_native_adamw_state_close(&native, &interpreted);
        let replayed = native.main_replay.workspace_stats();
        assert_eq!(replayed.allocation_count, prepared.allocation_count);
        assert_eq!(replayed.affine_matmul_materialization_bytes, 0);
    }

    #[test]
    fn native_cpu_adamw_materializes_duplicate_transposes_of_recurrent_parameter() {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_input("x", [2, 2], DType::F32)
            .unwrap();
        let parameter = TrainingParameterInit::new(
            "weight",
            TensorData::new([2, 2], vec![0.2, -0.1, 0.3, 0.4]).unwrap(),
        )
        .unwrap();
        let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
            let lhs = graph.permute(parameters["weight"], [1, 0])?;
            let rhs = graph.permute(parameters["weight"], [1, 0])?;
            let product = graph.matmul(lhs, rhs)?;
            let weighted = graph.mul(product, inputs["x"])?;
            let loss = graph.sum_all(weighted)?;
            Ok((loss, BTreeMap::from([("output".into(), product)])))
        })
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut native = target.prepare(&plan).unwrap();
        let mut interpreted = plan.prepare_cpu().unwrap();
        assert_eq!(native.preparation_report().main().fallback_count(), 0);

        let batch = BTreeMap::from([(
            "x".into(),
            TensorData::new([2, 2], vec![1.0, -0.5, 0.25, 2.0]).unwrap(),
        )]);
        let expected = interpreted
            .step(batch.clone(), TensorData::scalar(0.01))
            .unwrap();
        let actual = native.step(batch, TensorData::scalar(0.01)).unwrap();
        assert_cross_engine_tensor_close(
            "duplicate recurrent transpose loss",
            actual.loss(),
            expected.loss(),
        );
        assert_cross_engine_tensor_maps_close(
            "duplicate recurrent transpose output",
            actual.outputs(),
            expected.outputs(),
        );
        assert_native_adamw_state_close(&native, &interpreted);
        assert!(actual.report().traffic().borrowed_recurrent_input_bytes() > 0);
        assert!(actual.report().traffic().borrowed_recurrent_output_bytes() > 0);
    }

    #[test]
    fn native_cpu_adamw_materializes_transpose_colliding_with_dense_recurrent_parameter() {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_input("x", [2, 2], DType::F32)
            .unwrap();
        let parameter = TrainingParameterInit::new(
            "weight",
            TensorData::new([2, 2], vec![0.2, -0.1, 0.3, 0.4]).unwrap(),
        )
        .unwrap();
        let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
            let transposed = graph.permute(parameters["weight"], [1, 0])?;
            let product = graph.matmul(parameters["weight"], transposed)?;
            let weighted = graph.mul(product, inputs["x"])?;
            let loss = graph.sum_all(weighted)?;
            Ok((loss, BTreeMap::from([("output".into(), product)])))
        })
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut native = target.prepare(&plan).unwrap();
        let mut interpreted = plan.prepare_cpu().unwrap();
        assert_eq!(native.preparation_report().main().fallback_count(), 0);

        let batch = BTreeMap::from([(
            "x".into(),
            TensorData::new([2, 2], vec![1.0, -0.5, 0.25, 2.0]).unwrap(),
        )]);
        let expected = interpreted
            .step(batch.clone(), TensorData::scalar(0.01))
            .unwrap();
        let actual = native.step(batch, TensorData::scalar(0.01)).unwrap();
        assert_cross_engine_tensor_close(
            "dense recurrent transpose loss",
            actual.loss(),
            expected.loss(),
        );
        assert_cross_engine_tensor_maps_close(
            "dense recurrent transpose output",
            actual.outputs(),
            expected.outputs(),
        );
        assert_native_adamw_state_close(&native, &interpreted);
        assert!(actual.report().traffic().borrowed_recurrent_input_bytes() > 0);
        assert!(actual.report().traffic().borrowed_recurrent_output_bytes() > 0);
    }

    #[test]
    fn native_cpu_adamw_partial_flush_and_zero_grad_match_interpreter() {
        let plan = CompiledAdamWPlan::compile(
            accumulated_adamw_config(3),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        let reset_transition = plan.zero_grad.as_ref().unwrap();
        assert_eq!(
            reset_transition.state_buffers.len(),
            initial_parameters().len() + 1
        );
        assert!(
            reset_transition
                .state_buffers
                .keys()
                .all(RecurrentStateKey::is_accumulation_reset_state)
        );
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut native = target.prepare(&plan).unwrap();
        assert_eq!(executor.native_item_plan_count(), 3);
        let flush_workspace = native
            .partial_flush_replay
            .as_ref()
            .unwrap()
            .workspace_stats();
        let reset_workspace = native.zero_grad_replay.as_ref().unwrap().workspace_stats();
        assert!(flush_workspace.allocation_count > 0);
        assert!(reset_workspace.allocation_count > 0);
        let mut interpreted = plan.prepare_cpu().unwrap();
        assert!(native.preparation_report().partial_flush().is_some());
        assert!(native.preparation_report().zero_grad().is_some());

        let actual = native.step(batch(), lr()).unwrap();
        let expected = interpreted.step(batch(), lr()).unwrap();
        assert_cross_engine_tensor_close("cancelled-window loss", actual.loss(), expected.loss());
        assert_cross_engine_tensor_maps_close(
            "cancelled-window outputs",
            actual.outputs(),
            expected.outputs(),
        );
        let before_failed_reset = native.checkpoint().unwrap();
        let before_failed_reset_counts = native_recurrent_test_counts(&native);
        assert_eq!(native.successful_zero_grads, 0);
        assert!(native.zero_grad_with_injected_failure(0).is_err());
        let after_failed_reset_counts = native_recurrent_test_counts(&native);
        assert_eq!(after_failed_reset_counts.0, before_failed_reset_counts.0);
        assert_eq!(after_failed_reset_counts.1, before_failed_reset_counts.1);
        assert_eq!(native.checkpoint().unwrap(), before_failed_reset);
        assert_eq!(native.successful_zero_grads, 0);
        let before_reset = native_recurrent_test_counts(&native);
        assert_eq!(
            native.zero_grad().unwrap(),
            interpreted.zero_grad().unwrap()
        );
        let after_reset = native_recurrent_test_counts(&native);
        assert_eq!(after_reset.0, before_reset.0);
        assert_eq!(after_reset.1, before_reset.1 + 1);
        assert_eq!(native.successful_zero_grads, 1);
        assert_native_adamw_state_close(&native, &interpreted);
        let used_reset_workspace = native.zero_grad_replay.as_ref().unwrap().workspace_stats();
        assert_eq!(
            used_reset_workspace.allocation_count,
            reset_workspace.allocation_count
        );
        assert_eq!(used_reset_workspace.input_import_count, 0);
        assert!(used_reset_workspace.borrowed_recurrent_input_bytes > 0);
        assert_eq!(
            used_reset_workspace.borrowed_recurrent_input_bytes,
            used_reset_workspace.borrowed_recurrent_output_bytes
        );
        assert_eq!(used_reset_workspace.intermediate_materialization_count, 0);

        let before_empty_reset = native.checkpoint().unwrap();
        let before_empty_reset_counts = native_recurrent_test_counts(&native);
        let before_empty_reset_workspace =
            native.zero_grad_replay.as_ref().unwrap().workspace_stats();
        assert!(!native.zero_grad().unwrap().did_discard());
        assert_eq!(
            native_recurrent_test_counts(&native),
            before_empty_reset_counts
        );
        assert_eq!(native.successful_zero_grads, 1);
        assert_eq!(native.checkpoint().unwrap(), before_empty_reset);
        assert_eq!(
            native.zero_grad_replay.as_ref().unwrap().workspace_stats(),
            before_empty_reset_workspace
        );

        let actual = native.step(batch(), lr()).unwrap();
        let expected = interpreted.step(batch(), lr()).unwrap();
        assert_cross_engine_tensor_close("partial-window loss", actual.loss(), expected.loss());
        assert_cross_engine_tensor_maps_close(
            "partial-window outputs",
            actual.outputs(),
            expected.outputs(),
        );
        let before_failed_flush = native.checkpoint().unwrap();
        let before_failed_flush_counts = native_recurrent_test_counts(&native);
        assert_eq!(native.successful_flushes, 0);
        assert!(
            native
                .flush_partial_window_with_injected_failure(lr(), 0)
                .is_err()
        );
        let after_failed_flush_counts = native_recurrent_test_counts(&native);
        assert_eq!(after_failed_flush_counts.0, before_failed_flush_counts.0);
        assert_eq!(after_failed_flush_counts.1, before_failed_flush_counts.1);
        assert_eq!(native.checkpoint().unwrap(), before_failed_flush);
        assert_eq!(native.successful_flushes, 0);
        let before_flush = native_recurrent_test_counts(&native);
        let actual = native.flush_partial_window(lr()).unwrap();
        let after_flush = native_recurrent_test_counts(&native);
        assert_eq!(after_flush.0, before_flush.0);
        assert_eq!(after_flush.1, before_flush.1 + 1);
        let expected = interpreted.flush_partial_window(lr()).unwrap();
        assert_eq!(
            actual.flushed_microbatches(),
            expected.flushed_microbatches()
        );
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.report().unwrap().successful_invocation(), 1);
        let flush_traffic = actual.report().unwrap().traffic();
        assert_eq!(flush_traffic.external_input_import_count(), 0);
        assert_eq!(flush_traffic.external_input_import_bytes(), 0);
        assert!(flush_traffic.borrowed_recurrent_input_bytes() > 0);
        assert_eq!(
            flush_traffic.borrowed_recurrent_input_bytes(),
            flush_traffic.borrowed_recurrent_output_bytes()
        );
        assert_native_adamw_state_close(&native, &interpreted);

        let before_empty_flush_workspace = native
            .partial_flush_replay
            .as_ref()
            .unwrap()
            .workspace_stats();
        let before_empty_flush_counts = native_recurrent_test_counts(&native);
        let empty = native.flush_partial_window(lr()).unwrap();
        assert!(!empty.did_update());
        assert!(empty.report().is_none());
        assert_eq!(
            native_recurrent_test_counts(&native),
            before_empty_flush_counts
        );
        assert_eq!(executor.native_item_plan_count(), 3);
        assert_eq!(
            native
                .partial_flush_replay
                .as_ref()
                .unwrap()
                .workspace_stats(),
            before_empty_flush_workspace
        );
        let used_flush_workspace = native
            .partial_flush_replay
            .as_ref()
            .unwrap()
            .workspace_stats();
        assert_eq!(
            used_flush_workspace.allocation_count,
            flush_workspace.allocation_count
        );
        assert_eq!(used_flush_workspace.input_import_count, 0);
        assert_eq!(used_flush_workspace.borrowed_external_input_bytes, 8);
        assert!(used_flush_workspace.borrowed_recurrent_input_bytes > 0);
        assert_eq!(
            used_flush_workspace.borrowed_recurrent_input_bytes,
            used_flush_workspace.borrowed_recurrent_output_bytes
        );
        assert_eq!(used_flush_workspace.intermediate_materialization_count, 0);
    }

    #[test]
    fn native_cpu_compiled_multi_step_lr_matches_interpreter() {
        let config = accumulated_adamw_config(3)
            .with_captured_multi_step_lr(CompiledMultiStepLr::new(0.05, 0.5, [1]).unwrap());
        let plan = CompiledAdamWPlan::compile(config, initial_parameters(), build_tinybob).unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut native = target.prepare(&plan).unwrap();
        let mut interpreted = plan.prepare_cpu().unwrap();

        let before = native.checkpoint().unwrap();
        assert!(native.step(batch(), lr()).is_err());
        assert_eq!(native.checkpoint().unwrap(), before);
        for _ in 0..2 {
            let actual = native.step_scheduled(batch()).unwrap();
            let expected = interpreted.step_scheduled(batch()).unwrap();
            assert_cross_engine_tensor_close("scheduled loss", actual.loss(), expected.loss());
            assert_cross_engine_tensor_maps_close(
                "scheduled outputs",
                actual.outputs(),
                expected.outputs(),
            );
            assert_eq!(actual.report().traffic().external_input_import_count(), 1);
            assert_eq!(actual.report().traffic().external_input_import_bytes(), 32);
        }
        assert_eq!(
            native
                .main_replay
                .workspace_stats()
                .borrowed_external_input_bytes,
            64
        );
        let actual = native.flush_partial_window_scheduled().unwrap();
        let expected = interpreted.flush_partial_window_scheduled().unwrap();
        assert_eq!(
            actual.flushed_microbatches(),
            expected.flushed_microbatches()
        );
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_native_adamw_state_close(&native, &interpreted);
    }

    #[test]
    fn native_cpu_adamw_evaluation_input_failure_is_atomic_and_retryable() {
        let plan = CompiledModuleAdamWPlan::compile(
            module_config(),
            TiedFrozenModule::new([0.1, -0.2]),
            build_tied_frozen,
        )
        .unwrap()
        .with_evaluation(build_tied_frozen)
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut session = target.prepare(plan).unwrap();
        assert_eq!(executor.native_item_plan_count(), 2);
        let checkpoint = session.checkpoint().unwrap();
        let workspace = session
            .runtime
            .evaluation_replay
            .as_ref()
            .unwrap()
            .plan
            .workspace_stats();
        assert!(workspace.allocation_count > 0);
        assert_eq!(workspace.input_import_count, 0);

        assert!(session.evaluate(BTreeMap::new()).is_err());
        assert_eq!(session.checkpoint().unwrap(), checkpoint);
        assert_eq!(session.runtime.successful_evaluations, 0);
        assert_eq!(
            session
                .runtime
                .evaluation_replay
                .as_ref()
                .unwrap()
                .plan
                .workspace_stats(),
            workspace
        );

        let evaluation = session
            .evaluate(BTreeMap::from([(
                "x".into(),
                TensorData::new([2], vec![1.0, 2.0]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(evaluation.report().successful_invocation(), 1);
        assert!(evaluation.report().first_successful_invocation());
        assert_eq!(
            evaluation.report().traffic().external_input_import_count(),
            0
        );
        assert_eq!(
            evaluation.report().traffic().external_input_import_bytes(),
            0
        );
        assert_eq!(
            evaluation
                .report()
                .traffic()
                .borrowed_recurrent_input_bytes(),
            0
        );
        assert_eq!(
            evaluation
                .report()
                .traffic()
                .borrowed_recurrent_output_bytes(),
            0
        );
        assert_eq!(session.checkpoint().unwrap(), checkpoint);
        assert_eq!(executor.native_item_plan_count(), 2);
        let evaluated_workspace = session
            .runtime
            .evaluation_replay
            .as_ref()
            .unwrap()
            .plan
            .workspace_stats();
        assert_eq!(
            evaluated_workspace.allocation_count,
            workspace.allocation_count
        );
        assert_eq!(evaluated_workspace.input_import_count, 0);
        // Evaluation borrows both the 8-byte batch and the current 8-byte
        // parameter snapshot as ordinary external inputs. Leasing the active
        // parameter bank directly is intentionally deferred to the next PR.
        assert_eq!(evaluated_workspace.borrowed_external_input_bytes, 16);
        assert_eq!(evaluated_workspace.intermediate_materialization_count, 0);
    }

    #[test]
    fn native_cpu_adamw_rejects_unsupported_pure_items_during_preparation() {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_input("x", [4], DType::F32)
            .unwrap();
        let parameter = TrainingParameterInit::new("weight", TensorData::scalar(2.0)).unwrap();
        let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
            let loss = graph.square(parameters["weight"])?;
            let unsupported = graph.binary(crate::BinaryOp::Atan2, inputs["x"], inputs["x"])?;
            Ok((loss, BTreeMap::from([("unsupported".into(), unsupported)])))
        })
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        assert!(target.prepare(&plan).is_err());
        assert_eq!(executor.compile_cache_len(false), 0);
        assert_eq!(plan.step_count(), 0);
    }

    fn run_core_training_step<R: CompiledTrainingRuntime>(
        runtime: &mut R,
    ) -> (TensorData, BTreeMap<String, TensorData>) {
        let identity = runtime.capture_identity();
        let before = runtime.parameter_snapshots().unwrap();
        let step = runtime.step_batch(TinyBobBatch(batch()), 0.05).unwrap();
        assert_eq!(step.step(), 1);
        assert_eq!(step.capture_identity(), identity);
        assert_eq!(step.output("logits"), step.outputs().get("logits"));
        assert_eq!(runtime.step_count(), 1);
        assert_eq!(runtime.capture_identity(), identity);
        assert_ne!(runtime.parameter_snapshots().unwrap(), before);
        (step.loss().clone(), step.outputs().clone())
    }

    struct TinyBobBatch(BTreeMap<String, TensorData>);

    impl TinyBobBatch {
        const SCHEMA: [CompiledInputSpec; 2] = [
            CompiledInputSpec::new("target", &[4], DType::I64),
            CompiledInputSpec::new("x", &[4, 2], DType::F32),
        ];
    }

    impl CompiledInputBatch for TinyBobBatch {
        fn schema() -> &'static [CompiledInputSpec] {
            &Self::SCHEMA
        }

        fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
            Ok(self.0)
        }
    }

    struct RejectedBatch;

    impl CompiledInputBatch for RejectedBatch {
        fn schema() -> &'static [CompiledInputSpec] {
            &[]
        }

        fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
            Err(training("rejected test batch"))
        }
    }

    #[test]
    fn compiled_input_batch_conversion_precedes_recurrent_mutation() {
        let mut runtime = compiled_adamw();
        let checkpoint = runtime.checkpoint().unwrap();
        assert!(runtime.step_batch(RejectedBatch, 0.05).is_err());
        assert_eq!(runtime.step_count(), 0);
        assert_eq!(runtime.checkpoint().unwrap(), checkpoint);
    }

    #[test]
    fn compiled_training_runtime_is_optimizer_neutral() {
        let mut momentum = compiled();
        let mut adamw = compiled_adamw();

        let (momentum_loss, momentum_outputs) = run_core_training_step(&mut momentum);
        let (adamw_loss, adamw_outputs) = run_core_training_step(&mut adamw);
        assert_eq!(momentum_loss.shape(), adamw_loss.shape());
        assert_eq!(
            momentum_outputs.keys().collect::<Vec<_>>(),
            adamw_outputs.keys().collect::<Vec<_>>()
        );
        let checkpoint = adamw.checkpoint().unwrap();
        assert_eq!(
            decode_adamw_checkpoint(checkpoint.as_bytes())
                .unwrap()
                .capture_identity,
            adamw.capture_identity()
        );
    }

    #[test]
    fn optimizer_neutral_plan_renders_momentum_through_shared_metal_core() {
        let plan = CompiledTrainingPlan::compile(
            MomentumProgram {
                config: CompiledMomentumSgdConfig::new(0.9).unwrap(),
            },
            [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
            |graph, _, parameters| Ok((graph.square(parameters["weight"])?, BTreeMap::new())),
        )
        .unwrap();
        let capture_identity = plan.capture_identity().unwrap();
        let metal = plan
            .metal_plan(
                MetalRenderer::new(
                    8,
                    crate::runtime::metal::MetalCapabilities {
                        max_buffer_length: 1 << 30,
                        unified_memory: true,
                        family: "Apple9".into(),
                    },
                )
                .unwrap(),
                &BTreeMap::new(),
                None,
            )
            .unwrap();

        assert_eq!(metal.program_identity, capture_identity);
        assert_eq!(metal.inner.summary().fallback_count, 0);
        assert!(metal.inner.rendered_items().next().is_some());
        assert_eq!(
            metal
                .state_input_keys
                .values()
                .filter_map(RecurrentStateKey::momentum_parameter_name)
                .collect::<Vec<_>>(),
            ["weight"]
        );
    }

    #[test]
    fn adamw_plan_prepares_independent_cpu_runtimes() {
        let plan = CompiledAdamWPlan::compile(adamw_config(), initial_parameters(), build_tinybob)
            .unwrap();
        let identity = plan.capture_identity();
        assert_eq!(plan.step_count(), 0);

        let mut first = plan.prepare_cpu().unwrap();
        let second = plan.prepare_cpu().unwrap();
        assert_eq!(first.capture_identity(), identity);
        assert_eq!(second.capture_identity(), identity);
        assert_eq!(
            first.parameter_snapshots().unwrap(),
            second.parameter_snapshots().unwrap()
        );

        first.step(batch(), lr()).unwrap();
        assert_eq!(first.step_count(), 1);
        assert_eq!(second.step_count(), 0);
        assert_ne!(
            first.parameter_snapshots().unwrap(),
            second.parameter_snapshots().unwrap()
        );
    }

    fn build_two_parameter_linear_loss(
        graph: &mut Graph,
        _inputs: &BTreeMap<String, NodeId>,
        parameters: &BTreeMap<String, NodeId>,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let three = graph.full_with_dtype(Shape::from([]), Scalar::F(3.0), DType::F32)?;
        let four = graph.full_with_dtype(Shape::from([]), Scalar::F(4.0), DType::F32)?;
        let a = graph.mul(parameters["a"], three)?;
        let b = graph.mul(parameters["b"], four)?;
        Ok((graph.add(a, b)?, BTreeMap::new()))
    }

    fn batch() -> BTreeMap<String, TensorData> {
        BTreeMap::from([
            (
                "target".into(),
                TensorData::from_scalars(
                    [4],
                    DType::I64,
                    [Scalar::I(0), Scalar::I(1), Scalar::I(1), Scalar::I(0)],
                )
                .unwrap(),
            ),
            (
                "x".into(),
                TensorData::new([4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5]).unwrap(),
            ),
        ])
    }

    fn lr() -> TensorData {
        TensorData::scalar(0.05)
    }

    fn fresh_oracle_step(
        parameters: &BTreeMap<String, TensorData>,
        momentum: &BTreeMap<String, TensorData>,
    ) -> (
        TensorData,
        TensorData,
        BTreeMap<String, TensorData>,
        BTreeMap<String, TensorData>,
    ) {
        let values = batch();
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [4, 2], DType::F32);
        let target = graph.input_dtype("target", [4], DType::I64);
        let learning_rate = graph.input_dtype("lr", Shape::from([]), DType::F32);
        let mut parameter_nodes = BTreeMap::new();
        let mut momentum_nodes = BTreeMap::new();
        let mut bindings = HashMap::from([
            ("x".into(), values["x"].clone()),
            ("target".into(), values["target"].clone()),
            ("lr".into(), lr()),
        ]);
        for name in ["w1", "w2"] {
            let parameter_name = format!("parameter_{name}");
            let momentum_name = format!("momentum_{name}");
            let parameter = graph.input_dtype(
                parameter_name.clone(),
                parameters[name].shape().clone(),
                DType::F32,
            );
            let velocity = graph.input_dtype(
                momentum_name.clone(),
                momentum[name].shape().clone(),
                DType::F32,
            );
            parameter_nodes.insert(name.to_string(), parameter);
            momentum_nodes.insert(name.to_string(), velocity);
            bindings.insert(parameter_name, parameters[name].clone());
            bindings.insert(momentum_name, momentum[name].clone());
        }
        let hidden = graph.matmul(x, parameter_nodes["w1"]).unwrap();
        let hidden = graph.relu(hidden).unwrap();
        let logits = graph.matmul(hidden, parameter_nodes["w2"]).unwrap();
        let loss = cross_entropy(&mut graph, logits, target, LossOptions::default()).unwrap();
        let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
        let gradients = graph.gradient_default(loss, &targets).unwrap();
        let retained = graph
            .full_with_dtype(Shape::from([]), Scalar::F(0.9), DType::F32)
            .unwrap();
        let mut next_parameters = BTreeMap::new();
        let mut next_momentum = BTreeMap::new();
        let mut update_nodes = Vec::new();
        for ((name, parameter), gradient) in parameter_nodes.iter().zip(gradients) {
            let velocity = momentum_nodes[name];
            let velocity = graph
                .mul(retained, velocity)
                .and_then(|value| graph.add(value, gradient))
                .unwrap();
            let update = graph.mul(learning_rate, velocity).unwrap();
            let parameter = graph.sub(*parameter, update).unwrap();
            update_nodes.push((name.clone(), velocity, parameter));
        }
        let cpu = CpuBackend;
        for (name, velocity, parameter) in update_nodes {
            next_momentum.insert(
                name.clone(),
                cpu.execute(&graph, velocity, &bindings).unwrap(),
            );
            next_parameters.insert(name, cpu.execute(&graph, parameter, &bindings).unwrap());
        }
        (
            cpu.execute(&graph, loss, &bindings).unwrap(),
            cpu.execute(&graph, logits, &bindings).unwrap(),
            next_parameters,
            next_momentum,
        )
    }

    #[test]
    fn tinybob_three_step_compiled_replay_matches_fresh_cpu_training() {
        let mut compiled = compiled();
        let identity = compiled.capture_identity();
        let mut parameters = compiled.parameter_snapshots().unwrap();
        let mut momentum = compiled.momentum_snapshots().unwrap();
        for step in 1..=3 {
            let (loss, logits, next_parameters, next_momentum) =
                fresh_oracle_step(&parameters, &momentum);
            let result = compiled.step(batch(), lr()).unwrap();
            assert_eq!(result.loss().storage(), loss.storage());
            assert_eq!(result.output("logits").unwrap().storage(), logits.storage());
            assert_eq!(result.step(), step);
            assert_eq!(result.capture_identity(), identity);
            assert_eq!(compiled.parameter_snapshots().unwrap(), next_parameters);
            assert_eq!(compiled.momentum_snapshots().unwrap(), next_momentum);
            assert_eq!(
                compiled.parameter_versions().unwrap(),
                BTreeMap::from([("w1".into(), step), ("w2".into(), step)])
            );
            assert_eq!(
                compiled.momentum_versions().unwrap(),
                BTreeMap::from([("w1".into(), step), ("w2".into(), step)])
            );
            parameters = next_parameters;
            momentum = next_momentum;
        }
    }

    #[test]
    fn compile_identity_is_stable_and_initial_values_are_detached() {
        let original = initial_parameters();
        let before = original
            .iter()
            .map(|parameter| (parameter.name().to_string(), parameter.value().clone()))
            .collect::<BTreeMap<_, _>>();
        let first = CpuCompiledMomentumSgd::compile(config(), original, build_tinybob).unwrap();
        let second = compiled();
        assert_eq!(first.capture_identity(), second.capture_identity());
        assert_eq!(first.parameter_snapshots().unwrap(), before);
        let mut detached = first.parameter_snapshots().unwrap();
        detached
            .get_mut("w1")
            .unwrap()
            .assign(&TensorData::zeros([2, 4]).unwrap())
            .unwrap();
        assert_ne!(detached, first.parameter_snapshots().unwrap());
    }

    #[test]
    fn step_inputs_exclude_every_persistent_state_binding() {
        let compiled = compiled();
        let external = compiled
            .inner
            .inputs
            .keys()
            .map(String::as_str)
            .chain([LEARNING_RATE_INPUT])
            .collect::<BTreeSet<_>>();
        assert_eq!(
            external,
            BTreeSet::from(["target", "x", LEARNING_RATE_INPUT])
        );
        let persistent = compiled
            .inner
            .capture
            .state_bindings
            .iter()
            .map(|binding| {
                compiled
                    .inner
                    .capture
                    .schedule
                    .inputs
                    .iter()
                    .find(|input| input.node == binding.input_node)
                    .unwrap()
                    .name
                    .as_str()
            })
            .collect::<BTreeSet<_>>();
        assert!(!persistent.is_empty());
        assert!(
            persistent
                .iter()
                .all(|name| name.starts_with(INTERNAL_PREFIX))
        );
        assert!(external.is_disjoint(&persistent));
        let mut consumer_views = BTreeMap::<NodeId, BTreeSet<bool>>::new();
        for binding in &compiled.inner.capture.state_bindings {
            consumer_views
                .entry(binding.input_node)
                .or_default()
                .insert(binding.desc.view.is_some());
        }
        assert!(
            consumer_views
                .values()
                .any(|views| views == &BTreeSet::from([false, true]))
        );
        let pure_items = compiled
            .inner
            .capture
            .schedule
            .items
            .iter()
            .take_while(|item| !item.is_effect())
            .collect::<Vec<_>>();
        assert!(!pure_items.is_empty());
        assert!(pure_items.iter().all(|item| item.boundary.is_none()));
    }

    #[test]
    fn malformed_and_duplicate_inputs_fail_before_state_publication() {
        let duplicate = TrainingParameterInit::new(
            "w1",
            TensorData::zeros_with_dtype([2, 4], DType::F32).unwrap(),
        )
        .unwrap();
        assert!(
            CpuCompiledMomentumSgd::compile(
                config(),
                initial_parameters().into_iter().chain([duplicate]),
                build_tinybob,
            )
            .is_err()
        );
        assert!(
            CompiledMomentumSgdConfig::new(0.9)
                .unwrap()
                .with_input(INTERNAL_PREFIX, [1], DType::F32)
                .is_err()
        );
        assert!(
            CompiledMomentumSgdConfig::new(0.9)
                .unwrap()
                .with_input("x", [1], DType::F32)
                .unwrap()
                .with_input("x", [1], DType::F32)
                .is_err()
        );
        assert!(
            TrainingParameterInit::new(
                "bad",
                TensorData::zeros_with_dtype([1], DType::I32).unwrap(),
            )
            .is_err()
        );

        let mut compiled = compiled();
        let before = compiled.parameter_snapshots().unwrap();
        let mut missing = batch();
        missing.remove("target");
        assert!(compiled.step(missing, lr()).is_err());
        assert!(
            compiled
                .step(batch(), TensorData::new([1], vec![0.05]).unwrap())
                .is_err()
        );
        assert_eq!(compiled.step_count(), 0);
        assert_eq!(compiled.parameter_snapshots().unwrap(), before);
    }

    #[test]
    fn cross_namespace_names_reject_before_the_private_graph_builder_runs() {
        let invoked = std::cell::Cell::new(false);
        let conflicting = CompiledMomentumSgdConfig::new(0.9)
            .unwrap()
            .with_input("w1", [4, 2], DType::F32)
            .unwrap();
        let result =
            CpuCompiledMomentumSgd::compile(conflicting, initial_parameters(), |_, _, _| {
                invoked.set(true);
                Err(training("builder should not run"))
            });
        assert!(result.is_err());
        assert!(!invoked.get());
    }

    #[test]
    fn injected_and_stale_replay_failures_preserve_runtime_cursor_and_step() {
        let mut compiled = compiled();
        let initial_parameters = compiled.parameter_snapshots().unwrap();
        let initial_momentum = compiled.momentum_snapshots().unwrap();
        let initial_cursor = compiled.inner.cursor.clone();
        assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
        assert_eq!(compiled.step_count(), 0);
        assert_eq!(compiled.inner.cursor, initial_cursor);
        assert_eq!(compiled.parameter_snapshots().unwrap(), initial_parameters);
        assert_eq!(compiled.momentum_snapshots().unwrap(), initial_momentum);

        compiled.step(batch(), lr()).unwrap();
        let advanced = compiled.inner.cursor.clone();
        let advanced_parameters = compiled.parameter_snapshots().unwrap();
        compiled.inner.cursor = initial_cursor;
        assert!(compiled.step(batch(), lr()).is_err());
        assert_eq!(compiled.step_count(), 1);
        compiled.inner.cursor = advanced;
        assert_eq!(compiled.parameter_snapshots().unwrap(), advanced_parameters);
    }

    #[test]
    fn adamw_replays_one_capture_with_graph_owned_state() {
        let mut compiled = compiled_adamw();
        let identity = compiled.capture_identity();
        let zeros = initial_parameters()
            .into_iter()
            .map(|parameter| {
                (
                    parameter.name().to_string(),
                    TensorData::zeros_with_dtype(parameter.value().shape().clone(), DType::F32)
                        .unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(compiled.optimizer_step().unwrap(), 0);
        assert_eq!(compiled.first_moment_snapshots().unwrap(), zeros);
        assert_eq!(compiled.second_moment_snapshots().unwrap(), zeros);

        let first = compiled.step(batch(), lr()).unwrap();
        assert_eq!(first.step(), 1);
        assert_eq!(first.capture_identity(), identity);
        assert_eq!(compiled.optimizer_step().unwrap(), 1);
        assert_eq!(
            compiled
                .parameter_versions()
                .unwrap()
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            compiled
                .first_moment_versions()
                .unwrap()
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            compiled
                .second_moment_versions()
                .unwrap()
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_ne!(compiled.first_moment_snapshots().unwrap(), zeros);
        assert_ne!(compiled.second_moment_snapshots().unwrap(), zeros);

        let second = compiled.step(batch(), lr()).unwrap();
        assert_eq!(second.step(), 2);
        assert_eq!(second.capture_identity(), identity);
        assert_eq!(compiled.optimizer_step().unwrap(), 2);
    }

    #[test]
    fn adamw_clips_the_complete_parameter_gradient_set_by_one_global_norm() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap();
        assert_eq!(config.max_gradient_norm(), Some(1.0));
        assert!(!config.clip_report_enabled());
        let parameters = || {
            ["a", "b"]
                .map(|name| TrainingParameterInit::new(name, TensorData::scalar(0.0)).unwrap())
        };
        let unclipped = CpuCompiledAdamW::compile(
            CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0).unwrap(),
            parameters(),
            build_two_parameter_linear_loss,
        )
        .unwrap();
        let mut clipped = CpuCompiledAdamW::compile(
            config.clone(),
            parameters(),
            build_two_parameter_linear_loss,
        )
        .unwrap();
        assert_eq!(clipped.max_gradient_norm(), Some(1.0));
        assert_ne!(clipped.capture_identity(), unclipped.capture_identity());

        let unreported = clipped
            .step(BTreeMap::new(), TensorData::scalar(0.1))
            .unwrap();
        assert!(unreported.clip_report().is_none());
        let moments = clipped.first_moment_snapshots().unwrap();
        let a = moments["a"].scalar_at(0).as_f64();
        let b = moments["b"].scalar_at(0).as_f64();
        assert!((a - 0.6).abs() < 1e-6, "clipped a gradient was {a}");
        assert!((b - 0.8).abs() < 1e-6, "clipped b gradient was {b}");

        let reported_plan = CompiledAdamWPlan::compile(
            config.with_clip_report(),
            parameters(),
            build_two_parameter_linear_loss,
        )
        .unwrap();
        assert!(reported_plan.clip_report_enabled());
        assert_ne!(reported_plan.capture_identity(), clipped.capture_identity());
        let renderer = MetalRenderer::new(
            8,
            crate::runtime::metal::MetalCapabilities {
                max_buffer_length: 1 << 30,
                unified_memory: true,
                family: "Apple9".into(),
            },
        )
        .unwrap();
        let error = match reported_plan.metal_plan(renderer) {
            Ok(_) => panic!("clip-report plan rendered for Metal"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("clip reporting is currently CPU-only")
        );

        let mut interpreted = reported_plan.prepare_cpu().unwrap();
        let executor = CapturedReplayExecutor::default();
        let mut native = reported_plan
            .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
            .unwrap();
        let interpreted_step = interpreted
            .step(BTreeMap::new(), TensorData::scalar(0.1))
            .unwrap();
        let native_step = native
            .step(BTreeMap::new(), TensorData::scalar(0.1))
            .unwrap();
        let interpreted_report = interpreted_step.clip_report().unwrap();
        let native_report = native_step.clip_report().unwrap();
        assert_eq!(interpreted_report.pre_clip_global_norm(), 5.0);
        assert_eq!(interpreted_report.applied_scale(), 0.2);
        assert_eq!(interpreted_report.did_clip(), Some(true));
        assert_eq!(native_report, interpreted_report);
        let reported_checkpoint = interpreted.checkpoint().unwrap();
        assert_eq!(native.checkpoint().unwrap(), reported_checkpoint);
        let legacy_checkpoint = clipped.checkpoint().unwrap();
        assert!(
            reported_plan
                .restore_checkpoint(&legacy_checkpoint)
                .is_err()
        );
        let (legacy_tensors, mut legacy_metadata) =
            load_safetensors(legacy_checkpoint.as_bytes()).unwrap();
        let (reported_tensors, mut reported_metadata) =
            load_safetensors(reported_checkpoint.as_bytes()).unwrap();
        assert_eq!(reported_tensors, legacy_tensors);
        assert_ne!(
            reported_metadata.remove("capture_identity"),
            legacy_metadata.remove("capture_identity")
        );
        assert_eq!(reported_metadata, legacy_metadata);
    }

    #[test]
    fn adamw_loss_scaling_unscales_before_optimizer_policies() {
        let parameters = || {
            ["a", "b"]
                .map(|name| TrainingParameterInit::new(name, TensorData::scalar(0.0)).unwrap())
        };
        let base = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0).unwrap();
        let unit = CpuCompiledAdamW::compile(
            base.clone().with_loss_scale(1.0).unwrap(),
            parameters(),
            build_two_parameter_linear_loss,
        )
        .unwrap();
        let unscaled =
            CpuCompiledAdamW::compile(base.clone(), parameters(), build_two_parameter_linear_loss)
                .unwrap();
        assert_eq!(unit.loss_scale(), 1.0);
        assert_eq!(unit.capture_identity(), unscaled.capture_identity());

        let scaled_config = base
            .with_loss_scale(128.0)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap();
        assert_eq!(scaled_config.loss_scale(), 128.0);
        let mut scaled =
            CpuCompiledAdamW::compile(scaled_config, parameters(), build_two_parameter_linear_loss)
                .unwrap();
        let mut clipped = CpuCompiledAdamW::compile(
            CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
                .unwrap()
                .with_max_gradient_norm(1.0)
                .unwrap(),
            parameters(),
            build_two_parameter_linear_loss,
        )
        .unwrap();
        assert_eq!(scaled.loss_scale(), 128.0);
        assert_ne!(scaled.capture_identity(), clipped.capture_identity());

        let scaled_result = scaled
            .step(BTreeMap::new(), TensorData::scalar(0.1))
            .unwrap();
        let clipped_result = clipped
            .step(BTreeMap::new(), TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(scaled_result.loss(), clipped_result.loss());
        assert_eq!(
            scaled.parameter_snapshots().unwrap(),
            clipped.parameter_snapshots().unwrap()
        );
        assert_eq!(
            scaled.first_moment_snapshots().unwrap(),
            clipped.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            scaled.second_moment_snapshots().unwrap(),
            clipped.second_moment_snapshots().unwrap()
        );
    }

    #[test]
    fn adamw_weight_decay_exclusions_preserve_the_complete_optimizer_frontier() {
        let parameters = || {
            [
                TrainingParameterInit::new("a", TensorData::scalar(2.0)).unwrap(),
                TrainingParameterInit::new("b", TensorData::scalar(3.0)).unwrap(),
            ]
        };
        let config = |weight_decay, exclusions: &[&str]| {
            let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, weight_decay)
                .unwrap()
                .with_gradient_accumulation(2)
                .unwrap()
                .with_max_gradient_norm(1.0)
                .unwrap();
            config
                .with_weight_decay_exclusions(exclusions.iter().copied())
                .unwrap()
        };
        let compile = |config| {
            CpuCompiledAdamW::compile(config, parameters(), build_two_parameter_linear_loss)
                .unwrap()
        };
        let mut excluded = compile(config(0.1, &["b"]));
        let mut full_decay = compile(config(0.1, &[]));
        let mut no_decay = compile(config(0.0, &[]));

        for runtime in [&mut excluded, &mut full_decay, &mut no_decay] {
            let first = runtime
                .step(BTreeMap::new(), TensorData::scalar(0.1))
                .unwrap();
            assert!(!first.did_update());
            assert_eq!(
                runtime.gradient_accumulator_snapshots().unwrap(),
                BTreeMap::from([
                    ("a".into(), TensorData::scalar(3.0)),
                    ("b".into(), TensorData::scalar(4.0)),
                ])
            );
            let second = runtime
                .step(BTreeMap::new(), TensorData::scalar(0.1))
                .unwrap();
            assert!(second.did_update());
        }

        let excluded_parameters = excluded.parameter_snapshots().unwrap();
        let full_decay_parameters = full_decay.parameter_snapshots().unwrap();
        let no_decay_parameters = no_decay.parameter_snapshots().unwrap();
        assert_eq!(excluded_parameters["a"], full_decay_parameters["a"]);
        assert_ne!(excluded_parameters["a"], no_decay_parameters["a"]);
        assert_eq!(excluded_parameters["b"], no_decay_parameters["b"]);
        assert_ne!(excluded_parameters["b"], full_decay_parameters["b"]);
        assert_eq!(
            excluded.first_moment_snapshots().unwrap(),
            full_decay.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            excluded.first_moment_snapshots().unwrap(),
            no_decay.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            excluded.second_moment_snapshots().unwrap(),
            full_decay.second_moment_snapshots().unwrap()
        );
        assert_eq!(
            excluded.second_moment_snapshots().unwrap(),
            no_decay.second_moment_snapshots().unwrap()
        );
        assert!(
            excluded
                .gradient_accumulator_snapshots()
                .unwrap()
                .values()
                .all(|value| value == &TensorData::scalar(0.0))
        );

        let checkpoint = excluded.checkpoint().unwrap();
        let resumed = CpuCompiledAdamW::compile_from_checkpoint(
            config(0.1, &["b"]),
            &checkpoint,
            build_two_parameter_linear_loss,
        )
        .unwrap();
        assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(
                config(0.1, &[]),
                &checkpoint,
                build_two_parameter_linear_loss,
            )
            .is_err()
        );
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(
                config(0.1, &["a"]),
                &checkpoint,
                build_two_parameter_linear_loss,
            )
            .is_err()
        );
    }

    #[test]
    fn adamw_weight_decay_exclusion_names_validate_before_graph_construction() {
        struct NamesModule {
            trainable: Parameter,
            frozen: Parameter,
            buffer: Parameter,
        }

        impl Module for NamesModule {
            fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
                assert!(prefix.is_empty());
                visitor("weight".into(), &self.trainable, StateKind::Parameter);
                visitor("weight_alias".into(), &self.trainable, StateKind::Parameter);
                visitor("frozen".into(), &self.frozen, StateKind::Parameter);
                visitor("running".into(), &self.buffer, StateKind::Buffer);
            }
        }

        struct RepeatedCanonicalName(Parameter);

        impl Module for RepeatedCanonicalName {
            fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
                assert!(prefix.is_empty());
                visitor("weight".into(), &self.0, StateKind::Parameter);
                visitor("weight".into(), &self.0, StateKind::Parameter);
            }
        }

        let base = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.1).unwrap();
        assert_eq!(base.weight_decay_exclusions().len(), 0);
        assert_eq!(
            base.clone()
                .with_weight_decay_exclusions(["z", "a"])
                .unwrap()
                .weight_decay_exclusions()
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert!(
            base.clone()
                .with_weight_decay_exclusions(["weight", "weight"])
                .is_err()
        );
        assert!(
            base.clone()
                .with_weight_decay_exclusions(["weight"])
                .unwrap()
                .with_weight_decay_exclusions(["weight"])
                .is_err()
        );

        let parameters =
            || [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()];
        let empty = CpuCompiledAdamW::compile(
            base.clone()
                .with_weight_decay_exclusions(std::iter::empty::<&str>())
                .unwrap(),
            parameters(),
            |graph, _, parameters| {
                Ok((
                    graph.mul(parameters["weight"], parameters["weight"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap();
        let ordinary =
            CpuCompiledAdamW::compile(base.clone(), parameters(), |graph, _, parameters| {
                assert_eq!(graph.dtype(parameters["weight"])?, DType::F32);
                Ok((
                    graph.mul(parameters["weight"], parameters["weight"])?,
                    BTreeMap::new(),
                ))
            })
            .unwrap();
        assert_eq!(empty.capture_identity(), ordinary.capture_identity());
        assert_eq!(empty.checkpoint().unwrap(), ordinary.checkpoint().unwrap());

        let repeated = RepeatedCanonicalName(Parameter::new(TensorData::scalar(1.0), true));
        let invoked = std::cell::Cell::new(false);
        let result = CpuCompiledAdamW::compile_module(base.clone(), &repeated, |_, _, _| {
            invoked.set(true);
            Err(training("builder should not run"))
        });
        assert!(result.is_err());
        assert!(!invoked.get());

        let invoked = std::cell::Cell::new(false);
        let result = CpuCompiledAdamW::compile(
            base.clone()
                .with_weight_decay_exclusions(["missing"])
                .unwrap(),
            parameters(),
            |_, _, _| {
                invoked.set(true);
                Err(training("builder should not run"))
            },
        );
        assert!(result.is_err());
        assert!(!invoked.get());

        let module = NamesModule {
            trainable: Parameter::new(TensorData::scalar(1.0), true),
            frozen: Parameter::new(TensorData::scalar(2.0), false),
            buffer: Parameter::new(TensorData::scalar(3.0), false),
        };
        for invalid in ["weight_alias", "frozen", "running", "missing"] {
            let invoked = std::cell::Cell::new(false);
            let result = CpuCompiledAdamW::compile_module(
                base.clone()
                    .with_weight_decay_exclusions([invalid])
                    .unwrap(),
                &module,
                |_, _, _| {
                    invoked.set(true);
                    Err(training("builder should not run"))
                },
            );
            assert!(
                result.is_err(),
                "invalid exclusion {invalid:?} was accepted"
            );
            assert!(!invoked.get());
        }
    }

    #[test]
    fn token_weighted_accumulation_validates_static_policy_before_compilation() {
        let base = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0).unwrap();
        assert!(
            base.clone()
                .with_token_weighted_gradient_accumulation("mask")
                .is_err()
        );
        assert!(
            base.clone()
                .with_input("mask", [2], DType::F32)
                .unwrap()
                .with_token_weighted_gradient_accumulation("mask")
                .is_err()
        );
        assert!(
            base.clone()
                .with_gradient_accumulation(2)
                .unwrap()
                .with_input("mask", [2], DType::I32)
                .unwrap()
                .with_token_weighted_gradient_accumulation("mask")
                .is_err()
        );
        assert!(
            base.clone()
                .with_gradient_accumulation(2)
                .unwrap()
                .with_input("mask", [0], DType::F32)
                .unwrap()
                .with_token_weighted_gradient_accumulation("mask")
                .is_err()
        );
        assert!(
            base.with_gradient_accumulation(2)
                .unwrap()
                .with_input("mask", [8_388_609], DType::F32)
                .unwrap()
                .with_token_weighted_gradient_accumulation("mask")
                .is_err()
        );
    }

    fn token_weighted_config(steps: u64) -> CompiledAdamWConfig {
        CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(steps)
            .unwrap()
            .with_input("features", [3], DType::F32)
            .unwrap()
            .with_input("mask", [3], DType::F32)
            .unwrap()
            .with_token_weighted_gradient_accumulation("mask")
            .unwrap()
    }

    struct TokenMeanModule {
        weight: Parameter,
    }

    impl TokenMeanModule {
        fn new() -> Self {
            Self {
                weight: Parameter::new(TensorData::scalar(2.0), true),
            }
        }
    }

    impl Module for TokenMeanModule {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            assert!(prefix.is_empty());
            visitor("weight".into(), &self.weight, StateKind::Parameter);
        }
    }

    fn token_mean_dropout() -> CompiledDropoutConfig {
        CompiledDropoutConfig::new(CompiledDropoutKey([47, 53]))
    }

    fn build_token_losses(
        module: &TokenMeanModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<(NodeId, BTreeMap<String, NodeId>)> {
        let observation = dropout.dropout(graph, inputs["features"], 0.5)?;
        let weight = module.weight.bind(graph)?;
        let losses = graph.mul(weight, inputs["features"])?;
        Ok((
            losses,
            BTreeMap::from([("dropout_observation".into(), observation)]),
        ))
    }

    fn build_token_graph(
        module: &TokenMeanModule,
        graph: &mut Graph,
        inputs: &BTreeMap<String, NodeId>,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<CompiledAdamWGraph> {
        let (losses, outputs) = build_token_losses(module, graph, inputs, dropout)?;
        Ok(CompiledAdamWGraph::token_mean(losses, outputs))
    }

    fn compile_token_weighted_plan(steps: u64) -> CompiledAdamWPlan {
        CompiledAdamWPlan::compile_token_mean_module_with_dropout(
            token_weighted_config(steps),
            token_mean_dropout(),
            &TokenMeanModule::new(),
            build_token_losses,
        )
        .unwrap()
    }

    fn compile_direct_token_mean_without_dropout(
        config: CompiledAdamWConfig,
        module: &TokenMeanModule,
    ) -> CompiledAdamWPlan {
        let (mask_input, mask_shape) = token_mean_loss_descriptor(&config).unwrap();
        let parameter =
            TrainingParameterInit::new("weight", module.weight.value().unwrap()).unwrap();
        CompiledAdamWPlan::compile_parameters_with_lowered_loss(
            config,
            [parameter],
            |graph, inputs, parameters| {
                let losses = graph.mul(parameters["weight"], inputs["features"])?;
                let loss =
                    lower_token_mean_loss(graph, losses, inputs[mask_input.as_str()], &mask_shape)?;
                Ok((loss, BTreeMap::new()))
            },
        )
        .unwrap()
    }

    #[test]
    fn unified_scalar_objective_matches_legacy_capture_replay_and_checkpoint() {
        let module = TiedFrozenModule::new([0.1, -0.2]);
        let legacy =
            CompiledAdamWPlan::compile_module(module_config(), &module, build_tied_frozen).unwrap();
        let unified = CompiledAdamWPlan::compile_module_graph(
            module_config(),
            &module,
            |module, graph, inputs| {
                let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                let built = CompiledAdamWGraph::scalar(loss, outputs);
                assert_eq!(built.objective().node(), loss);
                assert_eq!(built.outputs().len(), 1);
                assert!(built.outputs().contains_key("output"));
                Ok(built)
            },
        )
        .unwrap();
        assert_eq!(unified.capture_identity(), legacy.capture_identity());
        assert_eq!(unified.inspection().unwrap(), legacy.inspection().unwrap());

        let inputs =
            || BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]);
        let mut legacy_runtime = legacy.prepare_cpu().unwrap();
        let mut unified_runtime = unified.prepare_cpu().unwrap();
        let legacy_step = legacy_runtime
            .step(inputs(), TensorData::scalar(0.01))
            .unwrap();
        let unified_step = unified_runtime
            .step(inputs(), TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(unified_step.loss(), legacy_step.loss());
        assert_eq!(unified_step.outputs(), legacy_step.outputs());
        assert_eq!(unified_step.optimizer_step(), legacy_step.optimizer_step());
        assert_eq!(unified_step.loss_weight(), 1);
        let checkpoint = unified_runtime.checkpoint().unwrap();
        assert_eq!(checkpoint, legacy_runtime.checkpoint().unwrap());
        assert_eq!(
            unified
                .restore_checkpoint(&checkpoint)
                .unwrap()
                .prepare_cpu()
                .unwrap()
                .checkpoint()
                .unwrap(),
            checkpoint
        );

        let owned = CompiledModuleAdamWPlan::compile_graph(
            module_config(),
            TiedFrozenModule::new([0.1, -0.2]),
            |module, graph, inputs| {
                let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                Ok(CompiledAdamWGraph::new(
                    CompiledAdamWObjective::scalar(loss),
                    outputs,
                ))
            },
        )
        .unwrap();
        assert_eq!(owned.capture_identity(), legacy.capture_identity());
        assert_eq!(owned.inspection().unwrap(), legacy.inspection().unwrap());

        let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([61, 67]));
        let legacy_dropout = CompiledAdamWPlan::compile_module_with_dropout(
            module_config(),
            dropout,
            &module,
            build_tied_dropout,
        )
        .unwrap();
        let unified_dropout = CompiledAdamWPlan::compile_module_graph_with_dropout(
            module_config(),
            dropout,
            &module,
            |module, graph, inputs, dropout| {
                let (loss, outputs) = build_tied_dropout(module, graph, inputs, dropout)?;
                Ok(CompiledAdamWGraph::scalar(loss, outputs))
            },
        )
        .unwrap();
        assert_eq!(
            unified_dropout.capture_identity(),
            legacy_dropout.capture_identity()
        );
        assert_eq!(
            unified_dropout.inspection().unwrap(),
            legacy_dropout.inspection().unwrap()
        );
    }

    #[test]
    fn unified_token_mean_objective_matches_legacy_and_rejects_policy_mismatch_atomically() {
        let module = TokenMeanModule::new();
        let direct = compile_direct_token_mean_without_dropout(token_weighted_config(2), &module);
        let unified_without_dropout = CompiledAdamWPlan::compile_module_graph(
            token_weighted_config(2),
            &module,
            |module, graph, inputs| {
                let weight = module.weight.bind(graph)?;
                let losses = graph.mul(weight, inputs["features"])?;
                Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
            },
        )
        .unwrap();
        assert_eq!(
            unified_without_dropout.capture_identity(),
            direct.capture_identity()
        );
        assert_eq!(
            unified_without_dropout.inspection().unwrap(),
            direct.inspection().unwrap()
        );
        let first_batch = || token_weighted_batch([1.0, 100.0, 1.0], [1.0, 0.0, 1.0]);
        let mut direct_runtime = direct.prepare_cpu().unwrap();
        let mut unified_without_dropout_runtime = unified_without_dropout.prepare_cpu().unwrap();
        let direct_step = direct_runtime
            .step(first_batch(), TensorData::scalar(0.1))
            .unwrap();
        let unified_without_dropout_step = unified_without_dropout_runtime
            .step(first_batch(), TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(unified_without_dropout_step.loss(), direct_step.loss());
        assert_eq!(
            unified_without_dropout_step.outputs(),
            direct_step.outputs()
        );
        assert_eq!(unified_without_dropout_step.loss_weight(), 2);
        let without_dropout_checkpoint = unified_without_dropout_runtime.checkpoint().unwrap();
        assert_eq!(
            without_dropout_checkpoint,
            direct_runtime.checkpoint().unwrap()
        );
        assert_eq!(
            unified_without_dropout
                .restore_checkpoint(&without_dropout_checkpoint)
                .unwrap()
                .prepare_cpu()
                .unwrap()
                .checkpoint()
                .unwrap(),
            without_dropout_checkpoint
        );

        let legacy = CompiledAdamWPlan::compile_token_mean_module_with_dropout(
            token_weighted_config(2),
            token_mean_dropout(),
            &module,
            build_token_losses,
        )
        .unwrap();
        let unified = CompiledAdamWPlan::compile_module_graph_with_dropout(
            token_weighted_config(2),
            token_mean_dropout(),
            &module,
            build_token_graph,
        )
        .unwrap();
        assert_eq!(unified.capture_identity(), legacy.capture_identity());
        assert_eq!(unified.inspection().unwrap(), legacy.inspection().unwrap());

        let mut legacy_runtime = legacy.prepare_cpu().unwrap();
        let mut unified_runtime = unified.prepare_cpu().unwrap();
        let legacy_step = legacy_runtime
            .step(first_batch(), TensorData::scalar(0.1))
            .unwrap();
        let unified_step = unified_runtime
            .step(first_batch(), TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(unified_step.loss(), legacy_step.loss());
        assert_eq!(unified_step.outputs(), legacy_step.outputs());
        assert_eq!(unified_step.loss_weight(), 2);
        let checkpoint = unified_runtime.checkpoint().unwrap();
        assert_eq!(checkpoint, legacy_runtime.checkpoint().unwrap());
        assert_eq!(
            unified
                .restore_checkpoint(&checkpoint)
                .unwrap()
                .prepare_cpu()
                .unwrap()
                .checkpoint()
                .unwrap(),
            checkpoint
        );

        let owned = CompiledModuleAdamWPlan::compile_graph_with_dropout(
            token_weighted_config(2),
            token_mean_dropout(),
            TokenMeanModule::new(),
            build_token_graph,
        )
        .unwrap();
        assert_eq!(owned.capture_identity(), legacy.capture_identity());
        assert_eq!(owned.inspection().unwrap(), legacy.inspection().unwrap());

        let scalar_before = module.weight.snapshot().unwrap();
        let scalar_mismatch = CompiledAdamWPlan::compile_module_graph(
            token_weighted_config(2),
            &module,
            |module, graph, inputs| {
                let weight = module.weight.bind(graph)?;
                let losses = graph.mul(weight, inputs["features"])?;
                Ok(CompiledAdamWGraph::scalar(
                    graph.sum_all(losses)?,
                    BTreeMap::new(),
                ))
            },
        );
        let scalar_error = match scalar_mismatch {
            Ok(_) => panic!("scalar objective compiled with token weighting"),
            Err(error) => error,
        };
        assert!(
            scalar_error
                .to_string()
                .contains("requires the token-mean-loss compile surface")
        );
        assert_parameter_snapshot_eq(&module.weight.snapshot().unwrap(), &scalar_before);

        let ordinary_config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_input("features", [3], DType::F32)
            .unwrap();
        let token_before = module.weight.snapshot().unwrap();
        let token_mismatch = CompiledAdamWPlan::compile_module_graph(
            ordinary_config.clone(),
            &module,
            |module, graph, inputs| {
                let weight = module.weight.bind(graph)?;
                Ok(CompiledAdamWGraph::token_mean(
                    graph.mul(weight, inputs["features"])?,
                    BTreeMap::new(),
                ))
            },
        );
        let token_error = match token_mismatch {
            Ok(_) => panic!("token-mean objective compiled without token weighting"),
            Err(error) => error,
        };
        assert!(
            token_error
                .to_string()
                .contains("token-mean-loss compilation requires token weighting")
        );
        assert_parameter_snapshot_eq(&module.weight.snapshot().unwrap(), &token_before);

        let owned_module = TokenMeanModule::new();
        let owned_before = owned_module.weight.snapshot().unwrap();
        let owned_error = match CompiledModuleAdamWPlan::compile_graph(
            ordinary_config,
            owned_module,
            |module, graph, inputs| {
                let weight = module.weight.bind(graph)?;
                Ok(CompiledAdamWGraph::token_mean(
                    graph.mul(weight, inputs["features"])?,
                    BTreeMap::new(),
                ))
            },
        ) {
            Ok(_) => panic!("token-mean objective compiled without its mask policy"),
            Err(error) => error,
        };
        assert!(
            owned_error
                .source_error()
                .to_string()
                .contains("token-mean-loss compilation requires token weighting")
        );
        let returned = owned_error.into_module();
        assert_parameter_snapshot_eq(&returned.weight.snapshot().unwrap(), &owned_before);
    }

    #[test]
    fn token_mean_loss_surface_owns_normalization_and_rejects_scalar_seams() {
        let module = TokenMeanModule::new();
        let config = token_weighted_config(2);
        let invoked = Cell::new(false);
        let raw = CompiledAdamWPlan::compile(
            config.clone(),
            [TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap()],
            |_, _, _| {
                invoked.set(true);
                Err(training("scalar builder should not run"))
            },
        );
        assert!(raw.is_err());
        assert!(!invoked.get());

        let module_scalar =
            CompiledAdamWPlan::compile_module(config.clone(), &module, |_, _, _| {
                invoked.set(true);
                Err(training("scalar builder should not run"))
            });
        assert!(module_scalar.is_err());
        assert!(!invoked.get());

        let dropout_scalar = CompiledAdamWPlan::compile_module_with_dropout(
            config.clone(),
            token_mean_dropout(),
            &module,
            |_, _, _, _| {
                invoked.set(true);
                Err(training("scalar builder should not run"))
            },
        );
        assert!(dropout_scalar.is_err());
        assert!(!invoked.get());

        let without_policy = CompiledAdamWPlan::compile_token_mean_module_with_dropout(
            CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
                .unwrap()
                .with_gradient_accumulation(2)
                .unwrap()
                .with_input("features", [3], DType::F32)
                .unwrap()
                .with_input("mask", [3], DType::F32)
                .unwrap(),
            token_mean_dropout(),
            &module,
            |_, _, _, _| {
                invoked.set(true);
                Err(training("token-loss builder should not run"))
            },
        );
        assert!(without_policy.is_err());
        assert!(!invoked.get());

        for wrong_dtype in [false, true] {
            let invalid = CompiledAdamWPlan::compile_token_mean_module_with_dropout(
                config.clone(),
                token_mean_dropout(),
                &module,
                |module, graph, inputs, dropout| {
                    let _ = dropout.dropout(graph, inputs["features"], 0.5)?;
                    let weight = module.weight.bind(graph)?;
                    let losses = graph.mul(weight, inputs["features"])?;
                    let losses = if wrong_dtype {
                        graph.cast(losses, DType::I32)?
                    } else {
                        graph.sum_all(losses)?
                    };
                    Ok((losses, BTreeMap::new()))
                },
            );
            assert!(invalid.is_err());
        }

        let plan = compile_token_weighted_plan(2);
        let mut runtime = plan.prepare_cpu().unwrap();
        let first = runtime
            .step(
                token_weighted_batch([1.0, 100.0, 1.0], [1.0, 0.0, 1.0]),
                TensorData::scalar(0.1),
            )
            .unwrap();
        assert_eq!(first.loss().scalar_at(0).as_f64(), 2.0);
        assert_eq!(first.loss_weight(), 2);
    }

    fn token_weighted_batch(features: [f32; 3], mask: [f32; 3]) -> BTreeMap<String, TensorData> {
        BTreeMap::from([
            (
                "features".into(),
                TensorData::new([3], features.to_vec()).unwrap(),
            ),
            ("mask".into(), TensorData::new([3], mask.to_vec()).unwrap()),
        ])
    }

    #[test]
    fn token_weighted_accumulation_is_atomic_native_consistent_and_checkpointed() {
        let plan = compile_token_weighted_plan(2);
        assert_eq!(
            plan.token_weighted_gradient_accumulation_mask(),
            Some("mask")
        );
        let renderer = MetalRenderer::new(
            8,
            crate::runtime::metal::MetalCapabilities {
                max_buffer_length: 1 << 30,
                unified_memory: true,
                family: "Apple9".into(),
            },
        )
        .unwrap();
        let error = match plan.metal_plan(renderer) {
            Ok(_) => panic!("token-weighted accumulation rendered for Metal"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("token-weighted accumulation is currently CPU-only")
        );
        let mut interpreted = plan.prepare_cpu().unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut native = target.prepare(&plan).unwrap();
        let initial = interpreted.checkpoint().unwrap();
        for invalid_mask in [[f32::NAN, 1.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]] {
            let batch = token_weighted_batch([1.0, 1.0, 100.0], invalid_mask);
            assert!(
                interpreted
                    .step(batch.clone(), TensorData::scalar(0.1))
                    .is_err()
            );
            assert!(native.step(batch, TensorData::scalar(0.1)).is_err());
            assert_eq!(interpreted.checkpoint().unwrap(), initial);
            assert_eq!(native.checkpoint().unwrap(), initial);
        }

        let first_batch = token_weighted_batch([1.0, 100.0, 1.0], [1.0, -0.0, 1.0]);
        let interpreted_first = interpreted
            .step(first_batch.clone(), TensorData::scalar(0.1))
            .unwrap();
        assert!(!interpreted_first.did_update());
        assert_eq!(interpreted_first.loss_weight(), 2);
        let native_first = native.step(first_batch, TensorData::scalar(0.1)).unwrap();
        assert_eq!(native_first.loss_weight(), interpreted_first.loss_weight());
        assert_eq!(native_first.report().successful_invocation(), 1);
        assert_eq!(native_first.report().fallback_count(), 0);
        let checkpoint = interpreted.checkpoint().unwrap();
        let native_checkpoint = native.checkpoint().unwrap();
        assert_native_adamw_state_close(&native, &interpreted);
        assert_eq!(checkpoint.info().accumulated_token_count(), Some(2));
        let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V6);
        let (mut tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        tensors.insert(
            "accumulated_token_count".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(0)]).unwrap(),
        );
        assert!(
            CompiledAdamWCheckpoint::from_bytes(save_safetensors(&tensors, &metadata).unwrap())
                .is_err()
        );
        let (mut tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        tensors.insert(
            "accumulated_token_count".into(),
            TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(4)]).unwrap(),
        );
        let impossible =
            CompiledAdamWCheckpoint::from_bytes(save_safetensors(&tensors, &metadata).unwrap())
                .unwrap();
        assert!(plan.restore_checkpoint(&impossible).is_err());

        let mut restored = plan
            .restore_checkpoint(&checkpoint)
            .unwrap()
            .prepare_cpu()
            .unwrap();
        let native_restored_plan = plan.restore_checkpoint(&native_checkpoint).unwrap();
        let mut native_restored = target.prepare(&native_restored_plan).unwrap();
        let second_batch = token_weighted_batch([3.0, 100.0, 100.0], [1.0, 0.0, 0.0]);
        let interpreted_second = interpreted
            .step(second_batch.clone(), TensorData::scalar(0.1))
            .unwrap();
        let restored_second = restored
            .step(second_batch.clone(), TensorData::scalar(0.1))
            .unwrap();
        let native_second = native
            .step(second_batch.clone(), TensorData::scalar(0.1))
            .unwrap();
        let native_restored_second = native_restored
            .step(second_batch, TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(interpreted_second.loss_weight(), 1);
        assert_eq!(restored_second.loss_weight(), 1);
        assert_eq!(native_second.loss_weight(), 1);
        assert_eq!(native_restored_second.loss_weight(), 1);
        assert_eq!(native_second.report().fallback_count(), 0);
        assert_eq!(
            restored.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );
        assert_eq!(
            native_restored.checkpoint().unwrap(),
            native.checkpoint().unwrap()
        );
        assert_native_adamw_state_close(&native, &interpreted);
        let first_moment = interpreted.first_moment_snapshots().unwrap()["weight"]
            .scalar_at(0)
            .as_f64();
        assert!((first_moment - 5.0 / 3.0).abs() < 1e-6);
        assert_eq!(
            interpreted
                .checkpoint()
                .unwrap()
                .info()
                .accumulated_token_count(),
            Some(0)
        );
    }

    #[test]
    fn token_weighted_partial_flush_and_zero_grad_reset_the_count() {
        let flush_plan = compile_token_weighted_plan(3);
        let mut flushed = flush_plan.prepare_cpu().unwrap();
        let executor = CapturedReplayExecutor::default();
        let target = NativeCpuSessionTarget::new(&executor);
        let mut native_flushed = target.prepare(&flush_plan).unwrap();
        let mut complete = compile_token_weighted_plan(2).prepare_cpu().unwrap();
        let batches = [
            token_weighted_batch([1.0, 1.0, 100.0], [1.0, 1.0, 0.0]),
            token_weighted_batch([3.0, 100.0, 100.0], [1.0, 0.0, 0.0]),
        ];
        for batch in &batches {
            flushed
                .step(batch.clone(), TensorData::scalar(0.1))
                .unwrap();
            let native = native_flushed
                .step(batch.clone(), TensorData::scalar(0.1))
                .unwrap();
            assert_eq!(native.report().fallback_count(), 0);
            complete
                .step(batch.clone(), TensorData::scalar(0.1))
                .unwrap();
        }
        assert_eq!(
            flushed
                .checkpoint()
                .unwrap()
                .info()
                .accumulated_token_count(),
            Some(3)
        );
        let native_checkpoint = native_flushed.checkpoint().unwrap();
        let native_restored_plan = flush_plan.restore_checkpoint(&native_checkpoint).unwrap();
        let mut native_restored = target.prepare(&native_restored_plan).unwrap();
        flushed
            .flush_partial_window(TensorData::scalar(0.1))
            .unwrap();
        let native_flush = native_flushed
            .flush_partial_window(TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(
            native_flush
                .report()
                .expect("a nonempty partial window runs")
                .fallback_count(),
            0
        );
        native_restored
            .flush_partial_window(TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(
            flushed.parameter_snapshots().unwrap(),
            complete.parameter_snapshots().unwrap()
        );
        assert_eq!(
            flushed.first_moment_snapshots().unwrap(),
            complete.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            flushed
                .checkpoint()
                .unwrap()
                .info()
                .accumulated_token_count(),
            Some(0)
        );
        assert_eq!(
            native_restored.checkpoint().unwrap(),
            native_flushed.checkpoint().unwrap()
        );
        assert_native_adamw_state_close(&native_flushed, &flushed);

        flushed
            .step(batches[0].clone(), TensorData::scalar(0.1))
            .unwrap();
        native_flushed
            .step(batches[0].clone(), TensorData::scalar(0.1))
            .unwrap();
        assert_eq!(
            flushed
                .checkpoint()
                .unwrap()
                .info()
                .accumulated_token_count(),
            Some(2)
        );
        flushed.zero_grad().unwrap();
        native_flushed.zero_grad().unwrap();
        assert_eq!(flushed.accumulation_index().unwrap(), 0);
        let reset_checkpoint = flushed.checkpoint().unwrap();
        assert_eq!(reset_checkpoint.info().accumulated_token_count(), Some(0));
        assert_eq!(reset_checkpoint.info().reset_transition_count(), 1);
        assert_eq!(
            native_flushed
                .checkpoint()
                .unwrap()
                .info()
                .accumulated_token_count(),
            Some(0)
        );
        assert_native_adamw_state_close(&native_flushed, &flushed);
        let (_, metadata) = load_safetensors(reset_checkpoint.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V7);
        assert_eq!(metadata["token_weighted_accumulation_present"], "true");
    }

    #[test]
    fn adamw_accumulation_clips_once_after_averaging_the_complete_window() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_loss_scale(128.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap()
            .with_input("scale", [], DType::F32)
            .unwrap();
        let parameter = TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap();
        let mut compiled =
            CpuCompiledAdamW::compile(config, [parameter], |graph, inputs, parameters| {
                Ok((
                    graph.mul(parameters["weight"], inputs["scale"])?,
                    BTreeMap::new(),
                ))
            })
            .unwrap();

        let input = |scale| BTreeMap::from([("scale".into(), TensorData::scalar(scale))]);
        let first = compiled
            .step(input(100.0), TensorData::scalar(0.1))
            .unwrap();
        assert!(!first.did_update());
        assert_eq!(
            compiled.gradient_accumulator_snapshots().unwrap()["weight"]
                .scalar_at(0)
                .as_f64(),
            100.0
        );

        let second = compiled
            .step(input(-99.0), TensorData::scalar(0.1))
            .unwrap();
        assert!(second.did_update());
        let first_moment = compiled.first_moment_snapshots().unwrap()["weight"]
            .scalar_at(0)
            .as_f64();
        assert!(
            (first_moment - 0.5).abs() < 1e-6,
            "window-average gradient was {first_moment}"
        );
        assert_eq!(
            compiled.gradient_accumulator_snapshots().unwrap()["weight"]
                .scalar_at(0)
                .as_f64(),
            0.0
        );
    }

    #[test]
    fn adamw_clip_report_marks_only_full_or_flushed_windows_on_both_cpu_paths() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_max_gradient_norm(1.0)
            .unwrap()
            .with_clip_report()
            .with_input("scale", [], DType::F32)
            .unwrap();
        let compile = || {
            CompiledAdamWPlan::compile(
                config.clone(),
                [TrainingParameterInit::new("weight", TensorData::scalar(0.0)).unwrap()],
                |graph, inputs, parameters| {
                    Ok((
                        graph.mul(parameters["weight"], inputs["scale"])?,
                        BTreeMap::new(),
                    ))
                },
            )
            .unwrap()
        };
        let input = |scale| BTreeMap::from([("scale".into(), TensorData::scalar(scale))]);
        let plan = compile();
        let mut interpreted = plan.prepare_cpu().unwrap();
        let executor = CapturedReplayExecutor::default();
        let mut native = plan
            .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
            .unwrap();

        assert!(
            interpreted
                .step(input(3.0), TensorData::scalar(0.1))
                .unwrap()
                .clip_report()
                .is_none()
        );
        assert!(
            native
                .step(input(3.0), TensorData::scalar(0.1))
                .unwrap()
                .clip_report()
                .is_none()
        );
        let interpreted_step = interpreted
            .step(input(5.0), TensorData::scalar(0.1))
            .unwrap();
        let native_step = native.step(input(5.0), TensorData::scalar(0.1)).unwrap();
        let expected = interpreted_step.clip_report().unwrap();
        assert_eq!(expected.pre_clip_global_norm(), 4.0);
        assert_eq!(expected.applied_scale(), 0.25);
        assert_eq!(native_step.clip_report(), Some(expected));
        assert_eq!(
            native.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );

        let mut interpreted_flush = compile().prepare_cpu().unwrap();
        let flush_plan = compile();
        let mut native_flush = flush_plan
            .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
            .unwrap();
        assert!(
            interpreted_flush
                .step(input(4.0), TensorData::scalar(0.1))
                .unwrap()
                .clip_report()
                .is_none()
        );
        assert!(
            native_flush
                .step(input(4.0), TensorData::scalar(0.1))
                .unwrap()
                .clip_report()
                .is_none()
        );
        let interpreted_flush_result = interpreted_flush
            .flush_partial_window(TensorData::scalar(0.1))
            .unwrap();
        let native_flush_result = native_flush
            .flush_partial_window(TensorData::scalar(0.1))
            .unwrap();
        let expected = interpreted_flush_result.clip_report().unwrap();
        assert_eq!(expected.pre_clip_global_norm(), 4.0);
        assert_eq!(expected.applied_scale(), 0.25);
        assert_eq!(native_flush_result.clip_report(), Some(expected));
        assert!(
            interpreted_flush
                .flush_partial_window(TensorData::scalar(0.1))
                .unwrap()
                .clip_report()
                .is_none()
        );
        assert!(
            native_flush
                .flush_partial_window(TensorData::scalar(0.1))
                .unwrap()
                .clip_report()
                .is_none()
        );
        assert_eq!(
            native_flush.checkpoint().unwrap(),
            interpreted_flush.checkpoint().unwrap()
        );
    }

    #[test]
    fn adamw_window_loss_reports_full_and_flushed_windows_on_both_cpu_paths() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_window_loss_report()
            .with_input("scale", [], DType::F32)
            .unwrap();
        let plan = CompiledAdamWPlan::compile(
            config,
            [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
            |graph, inputs, parameters| {
                Ok((
                    graph.mul(parameters["weight"], inputs["scale"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap();
        assert!(plan.window_loss_report_enabled());
        let input = |scale| BTreeMap::from([("scale".into(), TensorData::scalar(scale))]);
        let executor = CapturedReplayExecutor::default();
        let mut interpreted = plan.prepare_cpu().unwrap();
        let mut native = plan
            .prepare_native_cpu(&NativeCpuSessionTarget::new(&executor))
            .unwrap();

        assert!(
            interpreted
                .step(input(2.0), TensorData::scalar(0.0))
                .unwrap()
                .window_loss_report()
                .is_none()
        );
        assert!(
            native
                .step(input(2.0), TensorData::scalar(0.0))
                .unwrap()
                .window_loss_report()
                .is_none()
        );
        let interpreted_step = interpreted
            .step(input(4.0), TensorData::scalar(0.0))
            .unwrap();
        let native_step = native.step(input(4.0), TensorData::scalar(0.0)).unwrap();
        let report = interpreted_step.window_loss_report().unwrap();
        assert_eq!(report.mean_loss(), 3.0);
        assert_eq!(report.loss_weight(), 2);
        assert_eq!(report.microbatch_count(), 2);
        assert_eq!(native_step.window_loss_report(), Some(report));
        let checkpoint = interpreted.checkpoint().unwrap();
        assert_eq!(native.checkpoint().unwrap(), checkpoint);
        let (mut checkpoint_tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V8);
        assert_eq!(metadata["window_loss_report_enabled"], "true");
        assert!(
            checkpoint_tensors
                .remove("accumulated_loss_numerator")
                .is_some()
        );
        assert!(
            CompiledAdamWCheckpoint::from_bytes(
                save_safetensors(&checkpoint_tensors, &metadata).unwrap()
            )
            .is_err()
        );
        assert_eq!(checkpoint.info().accumulated_loss_numerator(), Some(0.0));
        let without_report = CompiledAdamWPlan::compile(
            CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
                .unwrap()
                .with_gradient_accumulation(2)
                .unwrap()
                .with_input("scale", [], DType::F32)
                .unwrap(),
            [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
            |graph, inputs, parameters| {
                Ok((
                    graph.mul(parameters["weight"], inputs["scale"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap();
        assert!(without_report.restore_checkpoint(&checkpoint).is_err());

        let mut flushed = plan.prepare_cpu().unwrap();
        let partial = flushed.step(input(6.0), TensorData::scalar(0.0)).unwrap();
        assert!(partial.window_loss_report().is_none());
        assert_eq!(
            flushed
                .checkpoint()
                .unwrap()
                .info()
                .accumulated_loss_numerator(),
            Some(6.0)
        );
        assert_eq!(flushed.zero_grad().unwrap().discarded_microbatches(), 1);
        assert_eq!(
            flushed
                .checkpoint()
                .unwrap()
                .info()
                .accumulated_loss_numerator(),
            Some(0.0)
        );
        flushed.step(input(8.0), TensorData::scalar(0.0)).unwrap();
        let flush = flushed
            .flush_partial_window(TensorData::scalar(0.0))
            .unwrap();
        let report = flush.window_loss_report().unwrap();
        assert_eq!(report.mean_loss(), 8.0);
        assert_eq!(report.loss_weight(), 1);
        assert_eq!(report.microbatch_count(), 1);
        assert!(
            flushed
                .flush_partial_window(TensorData::scalar(0.0))
                .unwrap()
                .window_loss_report()
                .is_none()
        );
    }

    #[test]
    fn non_finite_completed_window_loss_rejects_atomically_and_retries() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_window_loss_report()
            .with_input("offset", [], DType::F32)
            .unwrap();
        let plan = CompiledAdamWPlan::compile(
            config,
            [TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap()],
            |graph, inputs, parameters| {
                Ok((
                    graph.add(parameters["weight"], inputs["offset"])?,
                    BTreeMap::new(),
                ))
            },
        )
        .unwrap();
        let input = |offset| BTreeMap::from([("offset".into(), TensorData::scalar(offset))]);
        let target = rejecting_cpu_target();
        let executor = CapturedReplayExecutor::default();
        let native_target = NativeCpuSessionTarget::new(&executor)
            .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
        let mut interpreted = plan.prepare(&target).unwrap();
        let mut native = plan.prepare(&native_target).unwrap();

        assert!(
            interpreted
                .step(input(f32::MAX), TensorData::scalar(0.0))
                .unwrap()
                .window_loss_report()
                .is_none()
        );
        assert!(
            native
                .step(input(f32::MAX), TensorData::scalar(0.0))
                .unwrap()
                .window_loss_report()
                .is_none()
        );
        let interpreted_before = interpreted.checkpoint().unwrap();
        let native_before = native.checkpoint().unwrap();
        assert!(
            interpreted
                .step(input(f32::MAX), TensorData::scalar(0.0))
                .is_err()
        );
        assert!(
            native
                .step(input(f32::MAX), TensorData::scalar(0.0))
                .is_err()
        );
        assert_eq!(interpreted.checkpoint().unwrap(), interpreted_before);
        assert_eq!(native.checkpoint().unwrap(), native_before);

        let interpreted_retry = interpreted
            .step(input(-f32::MAX), TensorData::scalar(0.0))
            .unwrap();
        let native_retry = native
            .step(input(-f32::MAX), TensorData::scalar(0.0))
            .unwrap();
        let report = interpreted_retry.window_loss_report().unwrap();
        assert_eq!(report.mean_loss().to_bits(), 0.0_f32.to_bits());
        assert_eq!(report.loss_weight(), 2);
        assert_eq!(native_retry.window_loss_report(), Some(report));
        assert_eq!(
            native.checkpoint().unwrap(),
            interpreted.checkpoint().unwrap()
        );
    }

    #[test]
    fn adamw_accumulates_recurrent_gradients_and_commits_only_at_window_end() {
        let mut compiled = CpuCompiledAdamW::compile(
            accumulated_adamw_config(2),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        let identity = compiled.capture_identity();
        let initial_parameters = compiled.parameter_snapshots().unwrap();
        let initial_first = compiled.first_moment_snapshots().unwrap();
        let initial_second = compiled.second_moment_snapshots().unwrap();

        let partial = compiled.step(batch(), lr()).unwrap();
        assert_eq!(partial.step(), 1);
        assert_eq!(partial.optimizer_step(), 0);
        assert_eq!(partial.accumulation_index(), 1);
        assert!(!partial.did_update());
        assert_eq!(partial.capture_identity(), identity);
        assert_eq!(compiled.optimizer_step().unwrap(), 0);
        assert_eq!(compiled.accumulation_index().unwrap(), 1);
        assert_eq!(compiled.parameter_snapshots().unwrap(), initial_parameters);
        assert_eq!(compiled.first_moment_snapshots().unwrap(), initial_first);
        assert_eq!(compiled.second_moment_snapshots().unwrap(), initial_second);
        assert!(
            compiled
                .gradient_accumulator_snapshots()
                .unwrap()
                .values()
                .any(|value| value
                    != &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32,).unwrap())
        );

        let committed = compiled.step(batch(), lr()).unwrap();
        assert_eq!(committed.step(), 2);
        assert_eq!(committed.optimizer_step(), 1);
        assert_eq!(committed.accumulation_index(), 0);
        assert!(committed.did_update());
        assert_eq!(compiled.optimizer_step().unwrap(), 1);
        assert_eq!(compiled.accumulation_index().unwrap(), 0);
        assert_ne!(compiled.parameter_snapshots().unwrap(), initial_parameters);
        assert_ne!(compiled.first_moment_snapshots().unwrap(), initial_first);
        assert_ne!(compiled.second_moment_snapshots().unwrap(), initial_second);
        assert!(
            compiled
                .gradient_accumulator_snapshots()
                .unwrap()
                .values()
                .all(|value| value
                    == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32,).unwrap())
        );
    }

    #[test]
    fn adamw_partial_accumulation_checkpoint_resumes_exactly() {
        let config = accumulated_adamw_config(3);
        let mut uninterrupted =
            CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
        uninterrupted.step(batch(), lr()).unwrap();
        let checkpoint = uninterrupted.checkpoint().unwrap();
        let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V2);
        assert_eq!(metadata["replay_step"], "1");
        assert_eq!(metadata["optimizer_step"], "0");
        assert_eq!(metadata["gradient_accumulation_steps"], "3");
        assert_eq!(metadata["accumulation_index"], "1");

        let mut resumed =
            CpuCompiledAdamW::compile_from_checkpoint(config, &checkpoint, build_tinybob).unwrap();
        assert_eq!(resumed.step_count(), 1);
        assert_eq!(resumed.optimizer_step().unwrap(), 0);
        assert_eq!(resumed.accumulation_index().unwrap(), 1);
        assert_eq!(
            resumed.gradient_accumulator_snapshots().unwrap(),
            uninterrupted.gradient_accumulator_snapshots().unwrap()
        );
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            uninterrupted.parameter_snapshots().unwrap()
        );

        for expected_step in 2..=4 {
            let expected = uninterrupted.step(batch(), lr()).unwrap();
            let actual = resumed.step(batch(), lr()).unwrap();
            assert_eq!(actual.step(), expected_step);
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.optimizer_step(), expected.optimizer_step());
            assert_eq!(actual.accumulation_index(), expected.accumulation_index());
            assert_eq!(actual.did_update(), expected.did_update());
        }
        assert_eq!(
            resumed.checkpoint().unwrap(),
            uninterrupted.checkpoint().unwrap()
        );
    }

    #[test]
    fn adamw_plan_restore_rejects_authenticated_state_schema_mismatch_atomically() {
        let config = accumulated_adamw_config(3);
        let plan = CompiledAdamWPlan::compile(config, initial_parameters(), build_tinybob).unwrap();
        let initial = plan.prepare_cpu().unwrap().checkpoint().unwrap();
        let mut runtime = plan.prepare_cpu().unwrap();
        runtime.step(batch(), lr()).unwrap();
        let decoded = decode_adamw_checkpoint(runtime.checkpoint().unwrap().as_bytes()).unwrap();
        let progress = AdamWCheckpointProgress {
            capture_identity: decoded.capture_identity,
            replay_step: decoded.replay_step,
            optimizer_step: decoded.optimizer_step,
            accumulation_steps: decoded.accumulation_steps,
            accumulation_index: decoded.accumulation_index,
            discarded_microbatches: decoded.discarded_microbatches,
            flushed_window_count: decoded.flushed_window_count,
            flushed_microbatch_count: decoded.flushed_microbatch_count,
            flush_capture_identity: decoded.flush_capture_identity,
            dropout_block_counter: decoded.dropout_block_counter,
            accumulated_token_count: decoded.accumulated_token_count,
            window_loss_report: decoded.window_loss_report,
            reset_transition_count: decoded.reset_transition_count,
            reset_capture_identity: decoded.reset_capture_identity,
        };
        let mut tensors = AdamWCheckpointTensors {
            parameters: decoded.parameters,
            first_moments: decoded.first_moments,
            second_moments: decoded.second_moments,
            gradient_accumulators: decoded.gradient_accumulators,
            accumulated_loss_numerator: decoded.accumulated_loss_numerator,
        };
        for values in [
            &mut tensors.parameters,
            &mut tensors.first_moments,
            &mut tensors.second_moments,
            &mut tensors.gradient_accumulators,
        ] {
            values.insert(
                "w1".into(),
                TensorData::zeros_with_dtype([4, 2], DType::F32).unwrap(),
            );
        }
        let malformed = CompiledAdamWCheckpoint::from_bytes(
            encode_adamw_checkpoint(progress, tensors).unwrap(),
        )
        .unwrap();

        let error = match plan.restore_checkpoint(&malformed) {
            Ok(_) => panic!("an equal-byte state descriptor mismatch restored"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("state descriptor mismatch"));
        assert_eq!(plan.prepare_cpu().unwrap().checkpoint().unwrap(), initial);
    }

    #[test]
    fn adamw_partial_flush_matches_a_complete_short_window_and_resumes_exactly() {
        let config = accumulated_adamw_config(3);
        let mut flushed =
            CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
        let mut short = CpuCompiledAdamW::compile(
            accumulated_adamw_config(2),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        assert!(flushed.flush_capture_identity().is_some());
        let transition = flushed.partial_flush.as_ref().unwrap();
        assert_eq!(
            flushed.flush_capture_identity(),
            Some(
                transition
                    .capture
                    .initial_recurrent_cursor()
                    .unwrap()
                    .capture_identity()
            )
        );
        assert_ne!(
            flushed.flush_capture_identity(),
            Some(flushed.capture_identity())
        );
        for runtime in [&mut flushed, &mut short] {
            runtime.step(batch(), lr()).unwrap();
            runtime.step(batch(), lr()).unwrap();
        }
        let dropout_before = flushed.dropout_block_counter().unwrap();
        let result = flushed.flush_partial_window(lr()).unwrap();
        assert!(result.did_update());
        assert_eq!(result.flushed_microbatches(), 2);
        assert_eq!(result.optimizer_step(), 1);
        assert_eq!(flushed.step_count(), 2);
        assert_eq!(flushed.optimizer_step().unwrap(), 1);
        assert_eq!(flushed.accumulation_index().unwrap(), 0);
        assert_eq!(flushed.dropout_block_counter().unwrap(), dropout_before);
        assert_eq!(
            flushed.parameter_snapshots().unwrap(),
            short.parameter_snapshots().unwrap()
        );
        assert_eq!(
            flushed.first_moment_snapshots().unwrap(),
            short.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            flushed.second_moment_snapshots().unwrap(),
            short.second_moment_snapshots().unwrap()
        );
        assert!(
            flushed
                .gradient_accumulator_snapshots()
                .unwrap()
                .values()
                .all(|value| value
                    == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32).unwrap())
        );

        let checkpoint = flushed.checkpoint().unwrap();
        let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V5);
        assert_eq!(metadata["replay_step"], "2");
        assert_eq!(metadata["optimizer_step"], "1");
        assert_eq!(metadata["flushed_window_count"], "1");
        assert_eq!(metadata["flushed_microbatch_count"], "2");
        assert_eq!(metadata["dropout_state_present"], "false");
        assert_eq!(
            metadata["flush_capture_identity"],
            flushed.flush_capture_identity().unwrap().to_string()
        );
        let info = checkpoint.info();
        assert_eq!(info.capture_identity(), flushed.capture_identity());
        assert_eq!(info.replay_step(), 2);
        assert_eq!(info.optimizer_step(), 1);
        assert_eq!(info.gradient_accumulation_steps(), 3);
        assert_eq!(info.accumulation_index(), 0);
        assert_eq!(info.discarded_microbatches(), 0);
        assert_eq!(info.flushed_window_count(), 1);
        assert_eq!(info.flushed_microbatch_count(), 2);
        assert_eq!(
            info.flush_capture_identity(),
            flushed.flush_capture_identity()
        );
        assert_eq!(info.dropout_block_counter(), None);
        assert_eq!(info.accumulated_token_count(), None);
        assert!(
            flushed
                .parameter_versions()
                .unwrap()
                .values()
                .all(|version| *version == 3)
        );
        assert!(
            flushed
                .inner
                .plan()
                .unwrap()
                .state_versions
                .values()
                .all(|version| *version == 3),
            "parameters, moments, accumulators, and AdamW globals advance once"
        );
        let (state, mut mismatched_metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        let mut impossible_progress = mismatched_metadata.clone();
        impossible_progress.insert("replay_step".into(), "5".into());
        impossible_progress.insert("optimizer_step".into(), "2".into());
        impossible_progress.insert("flushed_window_count".into(), "2".into());
        impossible_progress.insert("flushed_microbatch_count".into(), "5".into());
        assert!(
            CompiledAdamWCheckpoint::from_bytes(
                save_safetensors(&state, &impossible_progress).unwrap()
            )
            .is_err(),
            "two partial N=3 windows can retain at most four microbatches"
        );
        mismatched_metadata.insert(
            "flush_capture_identity".into(),
            flushed
                .flush_capture_identity()
                .unwrap()
                .wrapping_add(1)
                .to_string(),
        );
        let mismatched = CompiledAdamWCheckpoint::from_bytes(
            save_safetensors(&state, &mismatched_metadata).unwrap(),
        )
        .unwrap();
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(config.clone(), &mismatched, build_tinybob,)
                .is_err()
        );
        let mut resumed =
            CpuCompiledAdamW::compile_from_checkpoint(config, &checkpoint, build_tinybob).unwrap();
        assert_eq!(
            resumed.parameter_versions().unwrap(),
            flushed.parameter_versions().unwrap()
        );
        assert_eq!(
            resumed.inner.plan().unwrap().state_versions,
            flushed.inner.plan().unwrap().state_versions
        );
        assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
        for _ in 0..3 {
            let expected = flushed.step(batch(), lr()).unwrap();
            let actual = resumed.step(batch(), lr()).unwrap();
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.optimizer_step(), expected.optimizer_step());
            assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        }
        assert_eq!(resumed.checkpoint().unwrap(), flushed.checkpoint().unwrap());
    }

    #[test]
    fn adamw_partial_flush_empty_validation_and_commit_failure_are_atomic() {
        let mut runtime = CpuCompiledAdamW::compile(
            accumulated_adamw_config(3),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        let initial = runtime.checkpoint().unwrap();
        let noop = runtime.flush_partial_window(lr()).unwrap();
        assert!(!noop.did_update());
        assert_eq!(noop.flushed_microbatches(), 0);
        assert_eq!(runtime.checkpoint().unwrap(), initial);
        assert!(
            runtime
                .flush_partial_window(TensorData::new([1], vec![0.1]).unwrap())
                .is_err()
        );
        assert_eq!(runtime.checkpoint().unwrap(), initial);

        runtime.step(batch(), lr()).unwrap();
        let partial = runtime.checkpoint().unwrap();
        assert!(runtime.flush_partial_window_inner(lr(), Some(0)).is_err());
        assert_eq!(runtime.checkpoint().unwrap(), partial);
        assert_eq!(runtime.accumulation_index().unwrap(), 1);
        assert_eq!(runtime.optimizer_step().unwrap(), 0);
        assert!(runtime.flush_partial_window(lr()).unwrap().did_update());
    }

    #[test]
    fn adamw_partial_flush_preserves_dropout_version_and_restores_split_frontier() {
        let config = module_config().with_gradient_accumulation(3).unwrap();
        let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([41, 43]));
        let module = TiedFrozenModule::new([0.1, -0.2]);
        let mut runtime = CompiledAdamWPlan::compile_module_with_dropout(
            config.clone(),
            dropout,
            &module,
            build_tied_dropout,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap();
        runtime
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![1.0, 2.0]).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        assert_eq!(runtime.dropout_block_counter().unwrap(), Some(1));
        assert!(
            runtime
                .flush_partial_window(TensorData::scalar(0.01))
                .unwrap()
                .did_update()
        );
        assert_eq!(runtime.dropout_block_counter().unwrap(), Some(1));
        let versions = runtime.inner.plan().unwrap().state_versions;
        assert_eq!(versions[&RecurrentStateKey::dropout_counter()], 1);
        assert!(
            versions
                .iter()
                .filter(|(key, _)| *key != &RecurrentStateKey::dropout_counter())
                .all(|(_, version)| *version == 2)
        );

        let checkpoint = runtime.checkpoint().unwrap();
        let fresh = TiedFrozenModule::new([0.1, -0.2]);
        let resumed = CompiledAdamWPlan::compile_module_with_dropout_from_checkpoint(
            config,
            dropout,
            &fresh,
            &checkpoint,
            build_tied_dropout,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap();
        assert_eq!(resumed.inner.plan().unwrap().state_versions, versions);
        assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
    }

    #[test]
    fn adamw_accumulation_failure_preserves_the_partial_frontier() {
        let mut compiled = CpuCompiledAdamW::compile(
            accumulated_adamw_config(2),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        compiled.step(batch(), lr()).unwrap();
        let cursor = compiled.inner.cursor.clone();
        let parameters = compiled.parameter_snapshots().unwrap();
        let first = compiled.first_moment_snapshots().unwrap();
        let second = compiled.second_moment_snapshots().unwrap();
        let accumulators = compiled.gradient_accumulator_snapshots().unwrap();

        assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
        assert_eq!(compiled.step_count(), 1);
        assert_eq!(compiled.optimizer_step().unwrap(), 0);
        assert_eq!(compiled.accumulation_index().unwrap(), 1);
        assert_eq!(compiled.inner.cursor, cursor);
        assert_eq!(compiled.parameter_snapshots().unwrap(), parameters);
        assert_eq!(compiled.first_moment_snapshots().unwrap(), first);
        assert_eq!(compiled.second_moment_snapshots().unwrap(), second);
        assert_eq!(
            compiled.gradient_accumulator_snapshots().unwrap(),
            accumulators
        );

        let cursor = compiled.inner.cursor.clone();
        let malformed = BTreeMap::from([(
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex),
            TensorData::scalar(0.0),
        )]);
        let step = compiled.step_count();
        assert!(
            compiled
                .inner
                .replace_state_values(step, malformed)
                .is_err()
        );
        assert_eq!(compiled.inner.cursor, cursor);
        assert_eq!(compiled.parameter_snapshots().unwrap(), parameters);
        assert_eq!(compiled.first_moment_snapshots().unwrap(), first);
        assert_eq!(compiled.second_moment_snapshots().unwrap(), second);
        assert_eq!(
            compiled.gradient_accumulator_snapshots().unwrap(),
            accumulators
        );
    }

    #[test]
    fn adamw_zero_grad_discards_only_the_partial_window_and_resumes_exactly() {
        let config = accumulated_adamw_config(3);
        let mut cancelled =
            CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
        let mut clean =
            CpuCompiledAdamW::compile(config.clone(), initial_parameters(), build_tinybob).unwrap();
        let initial_parameter_values = cancelled.parameter_snapshots().unwrap();
        let initial_first = cancelled.first_moment_snapshots().unwrap();
        let initial_second = cancelled.second_moment_snapshots().unwrap();

        cancelled.step(batch(), lr()).unwrap();
        let partial = cancelled.step(batch(), lr()).unwrap();
        assert_eq!(partial.step(), 2);
        assert_eq!(partial.optimizer_step(), 0);
        assert_eq!(partial.accumulation_index(), 2);
        let parameter_versions = cancelled.parameter_versions().unwrap();
        let first_versions = cancelled.first_moment_versions().unwrap();
        let second_versions = cancelled.second_moment_versions().unwrap();
        let state_versions_before_reset = cancelled.inner.plan().unwrap().state_versions;
        let before_failed_reset = cancelled.checkpoint().unwrap();
        assert!(cancelled.zero_grad_with_injected_failure(0).is_err());
        assert_eq!(cancelled.checkpoint().unwrap(), before_failed_reset);
        let reset = cancelled.zero_grad().unwrap();
        assert!(reset.did_discard());
        assert_eq!(reset.discarded_microbatches(), 2);
        assert_eq!(cancelled.step_count(), 2);
        assert_eq!(cancelled.optimizer_step().unwrap(), 0);
        assert_eq!(cancelled.accumulation_index().unwrap(), 0);
        assert_eq!(
            cancelled.parameter_snapshots().unwrap(),
            initial_parameter_values
        );
        assert_eq!(cancelled.first_moment_snapshots().unwrap(), initial_first);
        assert_eq!(cancelled.second_moment_snapshots().unwrap(), initial_second);
        assert_eq!(cancelled.parameter_versions().unwrap(), parameter_versions);
        assert_eq!(cancelled.first_moment_versions().unwrap(), first_versions);
        assert_eq!(cancelled.second_moment_versions().unwrap(), second_versions);
        let state_versions_after_reset = cancelled.inner.plan().unwrap().state_versions;
        for (key, before) in &state_versions_before_reset {
            let expected = if key.is_accumulation_reset_state() {
                before.checked_add(1).unwrap()
            } else {
                *before
            };
            assert_eq!(state_versions_after_reset[key], expected, "{key:?}");
        }
        assert!(
            cancelled
                .gradient_accumulator_snapshots()
                .unwrap()
                .values()
                .all(|value| value
                    == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32).unwrap())
        );

        let checkpoint = cancelled.checkpoint().unwrap();
        let (_, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V7);
        assert_eq!(metadata["replay_step"], "2");
        assert_eq!(metadata["optimizer_step"], "0");
        assert_eq!(metadata["accumulation_index"], "0");
        assert_eq!(metadata["discarded_microbatch_count"], "2");
        assert_eq!(metadata["reset_transition_count"], "1");
        assert_eq!(
            checkpoint.info().reset_capture_identity(),
            cancelled.zero_grad_capture_identity()
        );
        assert_eq!(checkpoint.info().reset_transition_count(), 1);
        let (state, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        let mut malformed_metadata = metadata.clone();
        malformed_metadata.insert("discarded_microbatch_count".into(), "3".into());
        let malformed = save_safetensors(&state, &malformed_metadata).unwrap();
        assert!(CompiledAdamWCheckpoint::from_bytes(malformed).is_err());
        let mut malformed_metadata = metadata.clone();
        malformed_metadata.insert("reset_transition_count".into(), "3".into());
        let malformed = save_safetensors(&state, &malformed_metadata).unwrap();
        assert!(CompiledAdamWCheckpoint::from_bytes(malformed).is_err());
        let mut malformed_metadata = metadata.clone();
        malformed_metadata.insert("reset_capture_identity".into(), "0".into());
        let malformed = CompiledAdamWCheckpoint::from_bytes(
            save_safetensors(&state, &malformed_metadata).unwrap(),
        )
        .unwrap();
        assert!(
            CompiledAdamWPlan::compile(config.clone(), initial_parameters(), build_tinybob,)
                .unwrap()
                .restore_checkpoint(&malformed)
                .is_err()
        );
        let mut overflow_metadata = metadata.clone();
        overflow_metadata.insert("replay_step".into(), u64::MAX.to_string());
        overflow_metadata.insert("discarded_microbatch_count".into(), u64::MAX.to_string());
        let overflow = CompiledAdamWCheckpoint::from_bytes(
            save_safetensors(&state, &overflow_metadata).unwrap(),
        )
        .unwrap();
        assert!(
            CompiledAdamWPlan::compile(config.clone(), initial_parameters(), build_tinybob,)
                .unwrap()
                .restore_checkpoint(&overflow)
                .is_err()
        );
        let mut zero_discard_metadata = metadata;
        zero_discard_metadata.insert("replay_step".into(), "0".into());
        zero_discard_metadata.insert("discarded_microbatch_count".into(), "0".into());
        let malformed = save_safetensors(&state, &zero_discard_metadata).unwrap();
        assert!(CompiledAdamWCheckpoint::from_bytes(malformed).is_err());
        assert!(
            validate_adamw_progress(
                AdamWProgress {
                    replay_step: 1,
                    optimizer_step: 0,
                    accumulation_index: 0,
                    discarded_microbatches: 1,
                    flushed_window_count: 0,
                    flushed_microbatch_count: 0,
                    reset_transition_count: 0,
                },
                1,
            )
            .is_err()
        );
        let before_noop = checkpoint.clone();
        let noop = cancelled.zero_grad().unwrap();
        assert!(!noop.did_discard());
        assert_eq!(noop.discarded_microbatches(), 0);
        assert_eq!(cancelled.checkpoint().unwrap(), before_noop);

        let mut resumed =
            CpuCompiledAdamW::compile_from_checkpoint(config, &checkpoint, build_tinybob).unwrap();
        assert_eq!(resumed.step_count(), 2);
        assert_eq!(resumed.accumulation_index().unwrap(), 0);
        assert_eq!(
            resumed.inner.plan().unwrap().state_versions,
            cancelled.inner.plan().unwrap().state_versions
        );
        assert_eq!(resumed.checkpoint().unwrap(), checkpoint);
        for _ in 0..3 {
            let expected = cancelled.step(batch(), lr()).unwrap();
            let actual = resumed.step(batch(), lr()).unwrap();
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.step(), expected.step());
            clean.step(batch(), lr()).unwrap();
        }
        assert_eq!(
            resumed.checkpoint().unwrap(),
            cancelled.checkpoint().unwrap()
        );
        assert_eq!(
            cancelled.parameter_snapshots().unwrap(),
            clean.parameter_snapshots().unwrap()
        );
        assert_eq!(
            cancelled.first_moment_snapshots().unwrap(),
            clean.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            cancelled.second_moment_snapshots().unwrap(),
            clean.second_moment_snapshots().unwrap()
        );
    }

    #[test]
    fn captured_zero_grad_clears_non_finite_accumulators() {
        let config = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0)
            .unwrap()
            .with_gradient_accumulation(2)
            .unwrap()
            .with_input("scale", [], DType::F32)
            .unwrap();
        let parameter = TrainingParameterInit::new("weight", TensorData::scalar(1.0)).unwrap();
        let plan = CompiledAdamWPlan::compile(config, [parameter], |graph, inputs, parameters| {
            Ok((
                graph.mul(parameters["weight"], inputs["scale"])?,
                BTreeMap::new(),
            ))
        })
        .unwrap();
        let inputs = BTreeMap::from([("scale".into(), TensorData::scalar(f32::INFINITY))]);

        let mut interpreted = plan.prepare_cpu().unwrap();
        interpreted.step(inputs.clone(), lr()).unwrap();
        assert!(
            interpreted.gradient_accumulator_snapshots().unwrap()["weight"]
                .scalar_at(0)
                .as_f64()
                .is_infinite()
        );
        interpreted.zero_grad().unwrap();
        assert_eq!(
            interpreted.gradient_accumulator_snapshots().unwrap()["weight"]
                .scalar_at(0)
                .as_f64(),
            0.0
        );

        let executor = CapturedReplayExecutor::default();
        let mut native = NativeCpuSessionTarget::new(&executor)
            .prepare(&plan)
            .unwrap();
        native.step(inputs, lr()).unwrap();
        native.zero_grad().unwrap();
        assert_eq!(
            native.gradient_accumulator_snapshots().unwrap()["weight"]
                .scalar_at(0)
                .as_f64(),
            0.0
        );
    }

    #[test]
    fn adamw_default_keeps_v1_checkpoint_and_accumulation_one_behavior() {
        let mut compiled = compiled_adamw();
        assert_eq!(compiled.gradient_accumulation_steps(), 1);
        assert_eq!(compiled.max_gradient_norm(), None);
        assert_eq!(compiled.loss_scale(), 1.0);
        assert!(
            compiled
                .gradient_accumulator_snapshots()
                .unwrap()
                .is_empty()
        );
        let result = compiled.step(batch(), lr()).unwrap();
        assert_eq!(result.optimizer_step(), 1);
        assert_eq!(result.accumulation_index(), 0);
        assert!(result.did_update());
        let (_, metadata) = load_safetensors(compiled.checkpoint().unwrap().as_bytes()).unwrap();
        assert_eq!(metadata["format"], ADAMW_CHECKPOINT_FORMAT_V1);
        assert_eq!(metadata["step"], "1");
        let before = compiled.checkpoint().unwrap();
        assert!(!compiled.zero_grad().unwrap().did_discard());
        assert_eq!(compiled.checkpoint().unwrap(), before);
    }

    #[test]
    fn adamw_failure_preserves_parameters_moments_step_and_cursor() {
        let mut compiled = compiled_adamw();
        let parameters = compiled.parameter_snapshots().unwrap();
        let first = compiled.first_moment_snapshots().unwrap();
        let second = compiled.second_moment_snapshots().unwrap();
        let cursor = compiled.inner.cursor.clone();

        assert!(compiled.step_inner(batch(), lr(), Some(0)).is_err());
        assert_eq!(compiled.step_count(), 0);
        assert_eq!(compiled.optimizer_step().unwrap(), 0);
        assert_eq!(compiled.inner.cursor, cursor);
        assert_eq!(compiled.parameter_snapshots().unwrap(), parameters);
        assert_eq!(compiled.first_moment_snapshots().unwrap(), first);
        assert_eq!(compiled.second_moment_snapshots().unwrap(), second);
    }

    #[test]
    fn owned_module_compile_failures_retain_module_and_preflight_versions() {
        let module = TiedFrozenModule::new([1.0, -1.0]);
        let shared_identity = module.shared.id();
        let error = match CompiledModuleAdamWPlan::compile(module_config(), module, |_, _, _| {
            Err(training("owned graph builder rejected the program"))
        }) {
            Ok(_) => panic!("failing graph builder compiled"),
            Err(error) => error,
        };
        assert!(
            error
                .source_error()
                .to_string()
                .contains("owned graph builder rejected")
        );
        let (module, source) = error.into_parts();
        assert!(source.to_string().contains("owned graph builder rejected"));
        assert_eq!(module.shared.id(), shared_identity);

        module.shared.set_version_for_test(u64::MAX).unwrap();
        let error =
            match CompiledModuleAdamWPlan::compile(module_config(), module, build_tied_frozen) {
                Ok(_) => panic!("unpublishable maximum-version module compiled"),
                Err(error) => error,
            };
        assert!(matches!(
            error.source_error(),
            Error::ParameterVersionOverflow { version: u64::MAX }
        ));
        let module = error.into_module();
        assert_eq!(module.shared.id(), shared_identity);
        assert_eq!(module.shared.version().unwrap(), u64::MAX);

        let dropout = CompiledDropoutConfig::new(CompiledDropoutKey([31, 37]));
        let source = TiedFrozenModule::new([1.0, -1.0]);
        let checkpoint = CompiledAdamWPlan::compile_module_with_dropout(
            module_config(),
            dropout,
            &source,
            build_tied_dropout,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap()
        .checkpoint()
        .unwrap();
        let candidate = TiedFrozenModule::new([1.0, -1.0]);
        let candidate_identity = candidate.shared.id();
        let error = match CompiledModuleAdamWPlan::compile_from_checkpoint(
            module_config(),
            candidate,
            &checkpoint,
            build_tied_frozen,
        ) {
            Ok(_) => panic!("mismatched checkpoint restored"),
            Err(error) => error,
        };
        assert_eq!(error.into_module().shared.id(), candidate_identity);
    }

    #[test]
    fn owned_module_adamw_session_seals_replay_and_finishes_atomically() {
        let module = TiedFrozenModule::new([1.0, -1.0]);
        let shared = module.shared.clone();
        let frozen = module.frozen.clone();
        let buffer = module.buffer.clone();
        let shared_before = shared.snapshot().unwrap();
        let frozen_before = frozen.snapshot().unwrap();
        let buffer_before = buffer.snapshot().unwrap();
        let plan =
            CompiledModuleAdamWPlan::compile(module_config(), module, build_tied_frozen).unwrap();
        let capture_identity = plan.capture_identity();
        let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
        let input = BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
        let step = session.step(input, TensorData::scalar(0.01)).unwrap();
        assert_eq!(step.capture_identity(), capture_identity);
        let published = session.parameter_snapshots().unwrap();
        let expected_checkpoint = session.checkpoint().unwrap();
        let (module, checkpoint) = session.finish_with_checkpoint().unwrap();

        assert_eq!(module.shared.value().unwrap(), published["shared"]);
        assert_eq!(module.shared.version().unwrap(), shared_before.version + 1);
        assert_eq!(module.frozen.value().unwrap(), frozen_before.data);
        assert_eq!(module.frozen.version().unwrap(), frozen_before.version);
        assert_eq!(module.buffer.value().unwrap(), buffer_before.data);
        assert_eq!(module.buffer.version().unwrap(), buffer_before.version);
        assert_eq!(module.shared.id(), shared.id());
        assert_eq!(module.frozen.id(), frozen.id());
        assert_eq!(module.buffer.id(), buffer.id());
        assert_eq!(
            CompiledAdamWCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
            checkpoint
        );
        assert_eq!(checkpoint, expected_checkpoint);

        let stale = TiedFrozenModule::new([1.0, -1.0]);
        let stale_frozen = stale.frozen.clone();
        let stale_frozen_before = stale_frozen.snapshot().unwrap();
        let plan =
            CompiledModuleAdamWPlan::compile(module_config(), stale, build_tied_frozen).unwrap();
        let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
        session
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        let runtime_checkpoint = session.checkpoint().unwrap();
        stale_frozen
            .replace(TensorData::new([2], vec![7.0, 8.0]).unwrap())
            .unwrap();
        let error = match session.finish_with_checkpoint() {
            Ok(_) => panic!("stale module state must reject publication"),
            Err(error) => error,
        };
        assert_eq!(error.session().step_count(), 1);
        assert_eq!(error.session().checkpoint().unwrap(), runtime_checkpoint);
        assert_eq!(error.session().parameter_snapshots().unwrap().len(), 1);
        assert_eq!(stale_frozen.value().unwrap().values(), &[7.0, 8.0]);
        stale_frozen
            .replace(stale_frozen_before.data.clone())
            .unwrap();
        stale_frozen
            .set_version_for_test(stale_frozen_before.version)
            .unwrap();
        let (stale, retried_checkpoint) = error.into_session().finish_with_checkpoint().unwrap();
        assert_eq!(retried_checkpoint, runtime_checkpoint);
        assert_parameter_snapshot_eq(&stale.frozen.snapshot().unwrap(), &stale_frozen_before);

        let stale_before_prepare = TiedFrozenModule::new([1.0, -1.0]);
        let leaked = stale_before_prepare.shared.clone();
        let plan = CompiledModuleAdamWPlan::compile(
            module_config(),
            stale_before_prepare,
            build_tied_frozen,
        )
        .unwrap();
        let capture_identity = plan.capture_identity();
        leaked
            .replace(TensorData::new([2], vec![4.0, 5.0]).unwrap())
            .unwrap();
        let error = match plan.prepare(&CpuSessionTarget::new()) {
            Ok(_) => panic!("stale owned plan prepared"),
            Err(error) => error,
        };
        assert_eq!(error.into_plan().capture_identity(), capture_identity);
    }

    #[test]
    fn complete_module_checkpoint_restores_constants_without_mutating_destination() {
        let config = module_config().with_gradient_accumulation(2).unwrap();
        let source = TiedFrozenModule::new([1.0, -1.0]);
        let source_frozen = source.frozen.clone();
        let source_buffer = source.buffer.clone();
        let source_plan = CompiledModuleAdamWPlan::compile_graph(
            config.clone(),
            source,
            |module, graph, inputs| {
                let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                Ok(CompiledAdamWGraph::scalar(loss, outputs))
            },
        )
        .unwrap();
        let capture_identity = source_plan.capture_identity();
        let mut source = source_plan.prepare(&CpuSessionTarget::new()).unwrap();
        let batch =
            || BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
        source.step(batch(), TensorData::scalar(0.01)).unwrap();
        let checkpoint = source.module_checkpoint().unwrap();
        assert_eq!(
            checkpoint.optimizer_checkpoint(),
            &source.checkpoint().unwrap()
        );
        assert_eq!(
            CompiledModuleAdamWCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
            checkpoint
        );
        let (envelope_tensors, _) = load_safetensors(checkpoint.as_bytes()).unwrap();
        assert_eq!(envelope_tensors.len(), 3);
        assert!(envelope_tensors.contains_key("optimizer_checkpoint"));

        let schema_destination = TiedFrozenModule::new([4.0, 5.0]);
        let schema_shared_before = schema_destination.shared.snapshot().unwrap();
        let schema_frozen_before = schema_destination.frozen.snapshot().unwrap();
        let schema_buffer_before = schema_destination.buffer.snapshot().unwrap();
        let (schema_tensors, mut schema_metadata) =
            load_safetensors(checkpoint.as_bytes()).unwrap();
        schema_metadata.insert("unexpected".into(), "field".into());
        assert!(
            CompiledModuleAdamWCheckpoint::from_bytes(
                save_safetensors(&schema_tensors, &schema_metadata).unwrap()
            )
            .is_err()
        );
        assert_parameter_snapshot_eq(
            &schema_destination.shared.snapshot().unwrap(),
            &schema_shared_before,
        );
        assert_parameter_snapshot_eq(
            &schema_destination.frozen.snapshot().unwrap(),
            &schema_frozen_before,
        );
        assert_parameter_snapshot_eq(
            &schema_destination.buffer.snapshot().unwrap(),
            &schema_buffer_before,
        );

        let topology_destination = TiedFrozenModule::new([5.0, 6.0]);
        let topology_shared_before = topology_destination.shared.snapshot().unwrap();
        let topology_frozen_before = topology_destination.frozen.snapshot().unwrap();
        let topology_buffer_before = topology_destination.buffer.snapshot().unwrap();
        let (topology_tensors, mut topology_metadata) =
            load_safetensors(checkpoint.as_bytes()).unwrap();
        topology_metadata.insert("visit.1.name".into(), "renamed_alias".into());
        let topology_checkpoint = CompiledModuleAdamWCheckpoint::from_bytes(
            save_safetensors(&topology_tensors, &topology_metadata).unwrap(),
        )
        .unwrap();
        let topology_error = CompiledModuleAdamWPlan::compile_graph_from_module_checkpoint(
            config.clone(),
            topology_destination,
            &topology_checkpoint,
            |module, graph, inputs| {
                let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                Ok(CompiledAdamWGraph::scalar(loss, outputs))
            },
        )
        .err()
        .expect("mismatched alias topology must reject");
        let topology_destination = topology_error.into_module();
        assert_parameter_snapshot_eq(
            &topology_destination.shared.snapshot().unwrap(),
            &topology_shared_before,
        );
        assert_parameter_snapshot_eq(
            &topology_destination.frozen.snapshot().unwrap(),
            &topology_frozen_before,
        );
        assert_parameter_snapshot_eq(
            &topology_destination.buffer.snapshot().unwrap(),
            &topology_buffer_before,
        );

        let destination = TiedFrozenModule::new([7.0, 8.0]);
        let shared = destination.shared.clone();
        let frozen = destination.frozen.clone();
        let buffer = destination.buffer.clone();
        buffer.replace(TensorData::scalar(-4.0)).unwrap();
        let shared_before = shared.snapshot().unwrap();
        let frozen_before = frozen.snapshot().unwrap();
        let buffer_before = buffer.snapshot().unwrap();
        let restored_plan = CompiledModuleAdamWPlan::compile_graph_from_module_checkpoint(
            config.clone(),
            destination,
            &checkpoint,
            |module, graph, inputs| {
                let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                Ok(CompiledAdamWGraph::scalar(loss, outputs))
            },
        )
        .unwrap();
        assert_eq!(restored_plan.capture_identity(), capture_identity);
        assert_parameter_snapshot_eq(&shared.snapshot().unwrap(), &shared_before);
        assert_parameter_snapshot_eq(&frozen.snapshot().unwrap(), &frozen_before);
        assert_parameter_snapshot_eq(&buffer.snapshot().unwrap(), &buffer_before);

        let mut resumed = restored_plan.prepare(&CpuSessionTarget::new()).unwrap();
        assert_eq!(
            resumed.checkpoint().unwrap(),
            checkpoint.optimizer_checkpoint().clone()
        );
        let uninterrupted_step = source.step(batch(), TensorData::scalar(0.01)).unwrap();
        let resumed_step = resumed.step(batch(), TensorData::scalar(0.01)).unwrap();
        assert_eq!(resumed_step.loss(), uninterrupted_step.loss());
        assert_eq!(resumed_step.outputs(), uninterrupted_step.outputs());
        assert_eq!(resumed.checkpoint().unwrap(), source.checkpoint().unwrap());
        let source = source.finish().unwrap();
        let destination = resumed.finish().unwrap();
        assert_eq!(
            destination.state_dict().unwrap(),
            source.state_dict().unwrap()
        );
        assert_eq!(destination.shared.id(), shared.id());
        assert!(destination.shared.is_trainable());
        assert!(!destination.frozen.is_trainable());
        assert!(!destination.buffer.is_trainable());
        assert_eq!(
            destination.frozen.value().unwrap(),
            source_frozen.value().unwrap()
        );
        assert_eq!(
            destination.buffer.value().unwrap(),
            source_buffer.value().unwrap()
        );
        assert_eq!(
            destination.shared.version().unwrap(),
            shared_before.version + 1
        );
        assert_eq!(
            destination.frozen.version().unwrap(),
            frozen_before.version + 1
        );
        assert_eq!(
            destination.buffer.version().unwrap(),
            buffer_before.version + 1
        );

        let malformed_destination = TiedFrozenModule::new([9.0, 10.0]);
        let malformed_shared = malformed_destination.shared.snapshot().unwrap();
        let malformed_frozen = malformed_destination.frozen.snapshot().unwrap();
        let malformed_buffer = malformed_destination.buffer.snapshot().unwrap();
        let (mut tensors, metadata) = load_safetensors(checkpoint.as_bytes()).unwrap();
        let immutable = tensors
            .iter_mut()
            .find(|(name, _)| name.starts_with("immutable."))
            .unwrap();
        *immutable.1 = TensorData::scalar(1.0);
        let malformed = CompiledModuleAdamWCheckpoint::from_bytes(
            save_safetensors(&tensors, &metadata).unwrap(),
        )
        .unwrap();
        let error = CompiledModuleAdamWPlan::compile_graph_from_module_checkpoint(
            config,
            malformed_destination,
            &malformed,
            |module, graph, inputs| {
                let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                Ok(CompiledAdamWGraph::scalar(loss, outputs))
            },
        )
        .err()
        .expect("malformed immutable descriptor must reject");
        let malformed_destination = error.into_module();
        assert_parameter_snapshot_eq(
            &malformed_destination.shared.snapshot().unwrap(),
            &malformed_shared,
        );
        assert_parameter_snapshot_eq(
            &malformed_destination.frozen.snapshot().unwrap(),
            &malformed_frozen,
        );
        assert_parameter_snapshot_eq(
            &malformed_destination.buffer.snapshot().unwrap(),
            &malformed_buffer,
        );
    }

    #[test]
    fn complete_checkpoint_finish_publishes_one_snapshot_and_resumes_exactly() {
        let config = tied_token_mean_config();
        let dropout = tied_token_mean_dropout();
        let source_module = TiedFrozenModule::new([1.0, -1.0]);
        let source_shared = source_module.shared.snapshot().unwrap();
        let source_tied_identity = source_module.shared.id();
        let source_frozen = source_module.frozen.snapshot().unwrap();
        let source_buffer = source_module.buffer.snapshot().unwrap();
        let source_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout(
            config.clone(),
            dropout,
            source_module,
            build_tied_frozen_token_mean,
        )
        .unwrap();
        let capture_identity = source_plan.capture_identity();
        let mut source = source_plan.prepare(&CpuSessionTarget::new()).unwrap();

        let first = source
            .step(
                tied_token_mean_batch([0.5, -0.25], [1.0, 0.0]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        assert_eq!(first.loss_weight(), 1);
        assert!(first.clip_report().is_none());
        let second = source
            .step(
                tied_token_mean_batch([-0.75, 0.25], [1.0, 1.0]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        assert_eq!(second.loss_weight(), 2);
        assert!(second.clip_report().is_some());
        source
            .step(
                tied_token_mean_batch([0.25, 0.75], [0.0, 1.0]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        let partial = source.module_checkpoint().unwrap();
        assert_eq!(partial.optimizer_checkpoint().info().optimizer_step(), 1);
        assert_eq!(
            partial.optimizer_checkpoint().info().accumulation_index(),
            1
        );
        assert_eq!(
            partial
                .optimizer_checkpoint()
                .info()
                .accumulated_token_count(),
            Some(1)
        );
        assert!(
            partial
                .optimizer_checkpoint()
                .info()
                .dropout_block_counter()
                .unwrap()
                > 0
        );

        let destination_module = TiedFrozenModule::new([7.0, 8.0]);
        destination_module
            .buffer
            .replace(TensorData::scalar(-4.0))
            .unwrap();
        let destination_shared = destination_module.shared.snapshot().unwrap();
        let destination_frozen = destination_module.frozen.snapshot().unwrap();
        let destination_buffer = destination_module.buffer.snapshot().unwrap();
        let destination_tied_identity = destination_module.shared.id();
        let restored_plan =
            CompiledModuleAdamWPlan::compile_graph_with_dropout_from_module_checkpoint(
                config,
                dropout,
                destination_module,
                &partial,
                build_tied_frozen_token_mean,
            )
            .unwrap();
        assert_eq!(restored_plan.capture_identity(), capture_identity);
        let mut resumed = restored_plan.prepare(&CpuSessionTarget::new()).unwrap();
        assert_eq!(resumed.module_checkpoint().unwrap(), partial);
        assert_parameter_snapshot_eq(
            &resumed.module.shared.snapshot().unwrap(),
            &destination_shared,
        );
        assert_parameter_snapshot_eq(
            &resumed.module.frozen.snapshot().unwrap(),
            &destination_frozen,
        );
        assert_parameter_snapshot_eq(
            &resumed.module.buffer.snapshot().unwrap(),
            &destination_buffer,
        );

        for (x, mask) in [([1.0, -0.5], [1.0, 1.0]), ([-0.5, 0.5], [1.0, 0.0])] {
            let expected = source
                .step(tied_token_mean_batch(x, mask), TensorData::scalar(0.01))
                .unwrap();
            let actual = resumed
                .step(tied_token_mean_batch(x, mask), TensorData::scalar(0.01))
                .unwrap();
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.loss_weight(), expected.loss_weight());
            assert_eq!(actual.clip_report(), expected.clip_report());
            assert_eq!(
                resumed.module_checkpoint().unwrap(),
                source.module_checkpoint().unwrap()
            );
        }

        let expected_optimizer = source.checkpoint().unwrap();
        let expected_complete = source.module_checkpoint().unwrap();
        assert_eq!(expected_optimizer.info().accumulation_index(), 1);
        assert_eq!(expected_optimizer.info().accumulated_token_count(), Some(1));
        let CompiledModuleAdamWSession {
            module,
            runtime,
            seal,
        } = source;
        let checkpoint_calls = Rc::new(Cell::new(0));
        let source = CompiledModuleAdamWSession {
            module,
            runtime: CheckpointCountingRuntime {
                inner: runtime,
                checkpoint_calls: Rc::clone(&checkpoint_calls),
            },
            seal,
        };
        let (source_module, completed) = source.finish_with_module_checkpoint().unwrap();
        assert_eq!(checkpoint_calls.get(), 1);
        assert_eq!(completed, expected_complete);
        assert_eq!(completed.optimizer_checkpoint(), &expected_optimizer);
        assert_eq!(
            CompiledModuleAdamWCheckpoint::from_bytes(completed.as_bytes().to_vec()).unwrap(),
            completed
        );
        assert_eq!(
            source_module.shared.value().unwrap(),
            decode_adamw_checkpoint(completed.optimizer_checkpoint().as_bytes())
                .unwrap()
                .parameters["shared"]
        );
        assert_eq!(source_module.shared.id(), source_tied_identity);
        assert_eq!(
            source_module.shared.version().unwrap(),
            source_shared.version + 1
        );
        assert_parameter_snapshot_eq(&source_module.frozen.snapshot().unwrap(), &source_frozen);
        assert_parameter_snapshot_eq(&source_module.buffer.snapshot().unwrap(), &source_buffer);
        let decoded = decode_module_adamw_checkpoint(completed.as_bytes()).unwrap();
        assert_eq!(decoded.states.len(), 3);
        assert_eq!(decoded.visits.len(), 4);
        assert_eq!(decoded.visits[0].canonical_name, "shared");
        assert_eq!(decoded.visits[1].canonical_name, "shared");

        let (destination_module, resumed_complete) =
            resumed.finish_with_module_checkpoint().unwrap();
        assert_eq!(resumed_complete, completed);
        assert_eq!(
            destination_module.state_dict().unwrap(),
            source_module.state_dict().unwrap()
        );
        assert_eq!(destination_module.shared.id(), destination_tied_identity);
        assert!(!destination_module.frozen.is_trainable());
        assert!(!destination_module.buffer.is_trainable());
    }

    #[test]
    fn checkpointed_finish_retains_session_after_late_publication_race() {
        let module = FinishRaceModule::new();
        let weight = module.weight.clone();
        let initial = weight.snapshot().unwrap();
        let plan =
            CompiledModuleAdamWPlan::compile(module_config(), module, build_finish_race).unwrap();
        let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
        session
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        let checkpoint = session.module_checkpoint().unwrap();
        session.module.arm_finish_race();

        let error = session.finish_with_module_checkpoint().unwrap_err();
        assert!(matches!(
            error.source_error(),
            Error::ParameterVersionConflict {
                expected: 0,
                actual: 1
            }
        ));
        assert_eq!(
            error.session().checkpoint().unwrap(),
            checkpoint.optimizer_checkpoint().clone()
        );
        assert_eq!(error.session().step_count(), 1);

        weight.set_version_for_test(initial.version).unwrap();
        let (module, retried_checkpoint) = error
            .into_session()
            .finish_with_module_checkpoint()
            .unwrap();
        assert_eq!(retried_checkpoint, checkpoint);
        assert_eq!(module.weight.version().unwrap(), initial.version + 1);
        assert_eq!(
            module.weight.value().unwrap(),
            decode_adamw_checkpoint(checkpoint.optimizer_checkpoint().as_bytes())
                .unwrap()
                .parameters["weight"]
        );
    }

    #[test]
    fn adamw_module_binding_owns_trainable_state_and_preserves_ties_and_freezing() {
        let module = TiedFrozenModule::new([1.0, -1.0]);
        let mut compiled =
            CpuCompiledAdamW::compile_module(module_config(), &module, build_tied_frozen).unwrap();
        assert_eq!(
            compiled
                .parameter_snapshots()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["shared"]
        );

        // Host state is only an initialization source. Once compiled, the
        // recurrent runtime is the sole owner of the trainable value.
        module
            .shared
            .replace(TensorData::new([2], vec![9.0, 9.0]).unwrap())
            .unwrap();
        let input = BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
        let first = compiled
            .step(input.clone(), TensorData::scalar(0.01))
            .unwrap();
        assert_eq!(first.step(), 1);
        assert_eq!(compiled.optimizer_step().unwrap(), 1);
        assert_eq!(compiled.parameter_versions().unwrap()["shared"], 1);

        let runtime_checkpoint = compiled.checkpoint().unwrap();
        let runtime_progress = (compiled.step_count(), compiled.optimizer_step().unwrap());
        let published = compiled.parameter_snapshots().unwrap();
        let host_version = module.shared.version().unwrap();
        let frozen = module.frozen.snapshot().unwrap();
        assert!(compiled.publish_parameters(&module).unwrap().is_clean());
        assert_eq!(module.shared.value().unwrap(), published["shared"]);
        assert_eq!(module.shared.version().unwrap(), host_version + 1);
        assert_eq!(module.frozen.snapshot().unwrap().data, frozen.data);
        assert_eq!(module.frozen.version().unwrap(), frozen.version);
        assert_eq!(compiled.checkpoint().unwrap(), runtime_checkpoint);
        assert_eq!(
            (compiled.step_count(), compiled.optimizer_step().unwrap()),
            runtime_progress
        );

        let checkpoint = compiled.checkpoint().unwrap();
        let resumed_module = TiedFrozenModule::new([1.0, -1.0]);
        let mut resumed = CpuCompiledAdamW::compile_module_from_checkpoint(
            module_config(),
            &resumed_module,
            &checkpoint,
            build_tied_frozen,
        )
        .unwrap();
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            compiled.parameter_snapshots().unwrap()
        );
        assert_eq!(
            resumed.first_moment_snapshots().unwrap(),
            compiled.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.second_moment_snapshots().unwrap(),
            compiled.second_moment_snapshots().unwrap()
        );
        assert_eq!(resumed.step_count(), 1);
        assert_eq!(
            resumed
                .step(input.clone(), TensorData::scalar(0.01))
                .unwrap()
                .step(),
            2
        );

        let accumulation_module = TiedFrozenModule::new([1.0, -1.0]);
        let mut accumulated = CpuCompiledAdamW::compile_module(
            module_config().with_gradient_accumulation(2).unwrap(),
            &accumulation_module,
            build_tied_frozen,
        )
        .unwrap();
        let capture_identity = accumulated.capture_identity();
        let parameter = accumulated.parameter_snapshots().unwrap();
        accumulated.step(input, TensorData::scalar(0.01)).unwrap();
        assert_eq!(accumulated.parameter_snapshots().unwrap(), parameter);
        assert_eq!(
            accumulated
                .gradient_accumulator_snapshots()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["shared"]
        );
        assert!(accumulated.zero_grad().unwrap().did_discard());
        assert_eq!(accumulated.capture_identity(), capture_identity);
        assert_eq!(accumulated.parameter_snapshots().unwrap(), parameter);
        assert!(
            accumulated
                .gradient_accumulator_snapshots()
                .unwrap()
                .values()
                .all(|value| value
                    == &TensorData::zeros_with_dtype(value.shape().clone(), DType::F32).unwrap())
        );

        let changed_frozen = TiedFrozenModule::new([2.0, -1.0]);
        assert!(
            CpuCompiledAdamW::compile_module_from_checkpoint(
                module_config(),
                &changed_frozen,
                &checkpoint,
                build_tied_frozen,
            )
            .is_err()
        );
    }

    #[test]
    fn adamw_compile_time_freezing_is_canonical_tied_and_raw_fail_closed() {
        let config = module_config().with_frozen_parameters(["base"]).unwrap();
        assert_eq!(config.frozen_parameters().collect::<Vec<_>>(), ["base"]);
        assert!(config.clone().with_frozen_parameters(["base"]).is_err());
        let raw_parameter =
            TrainingParameterInit::new("adapter", TensorData::zeros([2]).unwrap()).unwrap();
        assert!(
            CompiledAdamWPlan::compile(config.clone(), [raw_parameter], |_, _, _| panic!(
                "raw frozen-name policy reached graph construction"
            ),)
            .is_err()
        );

        for name in ["base_alias", "frozen", "missing"] {
            let module = FineTuneModule::new();
            let invalid = module_config().with_frozen_parameters([name]).unwrap();
            assert!(
                CompiledAdamWPlan::compile_module(invalid, &module, |_, _, _| {
                    panic!("invalid frozen-name policy reached graph construction")
                })
                .is_err()
            );
        }
        let buffer = TiedFrozenModule::new([1.0, -1.0]);
        assert!(
            CompiledAdamWPlan::compile_module(
                module_config().with_frozen_parameters(["buffer"]).unwrap(),
                &buffer,
                build_tied_frozen,
            )
            .is_err()
        );
        let all_frozen = TiedFrozenModule::new([1.0, -1.0]);
        assert!(
            CompiledAdamWPlan::compile_module(
                module_config().with_frozen_parameters(["shared"]).unwrap(),
                &all_frozen,
                build_tied_frozen,
            )
            .is_err()
        );
        let overlap = FineTuneModule::new();
        assert!(
            CompiledAdamWPlan::compile_module(
                config
                    .clone()
                    .with_weight_decay_exclusions(["base"])
                    .unwrap(),
                &overlap,
                build_fine_tune,
            )
            .is_err()
        );
    }

    #[test]
    fn parameter_override_tracks_source_and_effective_trainability_separately() {
        let parameter = Parameter::new(TensorData::new([2], vec![1.0, 2.0]).unwrap(), true);
        let mut graph = Graph::new();
        let frozen = graph.constant(parameter.value().unwrap());
        assert!(
            graph
                .with_parameter_overrides(
                    BTreeMap::from([(parameter.id(), (frozen, false, false))]),
                    |graph| parameter.bind(graph),
                )
                .is_err(),
            "effective freezing must not weaken source-trainability authentication"
        );
        let bound = graph
            .with_parameter_overrides(
                BTreeMap::from([(parameter.id(), (frozen, true, false))]),
                |graph| parameter.bind(graph),
            )
            .unwrap();
        assert_eq!(bound, frozen);
        assert!(!graph.requires_grad(bound).unwrap());
        assert!(parameter.is_trainable());
    }

    #[test]
    fn adamw_compile_time_freezing_reduces_state_and_resumes_accumulation_exactly() {
        let config = module_config()
            .with_gradient_accumulation(3)
            .unwrap()
            .with_frozen_parameters(["base"])
            .unwrap();
        let module = FineTuneModule::new();
        let base = module.base.snapshot().unwrap();
        let adapter = module.adapter.snapshot().unwrap();
        let plan =
            CompiledAdamWPlan::compile_module(config.clone(), &module, build_fine_tune).unwrap();
        let expected_parameter_keys = vec!["adapter".to_owned()];
        assert_eq!(
            plan.inner
                .state_values
                .keys()
                .filter_map(RecurrentStateKey::parameter_name)
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            expected_parameter_keys
        );
        assert!(module.base.is_trainable());
        assert!(module.adapter.is_trainable());
        assert_parameter_snapshot_eq(&module.base.snapshot().unwrap(), &base);
        assert_parameter_snapshot_eq(&module.adapter.snapshot().unwrap(), &adapter);

        let metal = plan
            .metal_plan(
                MetalRenderer::new(
                    8,
                    crate::runtime::metal::MetalCapabilities {
                        max_buffer_length: 1 << 30,
                        unified_memory: true,
                        family: "Apple9".into(),
                    },
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            metal
                .inner
                .state_input_keys
                .values()
                .filter_map(RecurrentStateKey::parameter_name)
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            expected_parameter_keys
        );

        let input =
            || BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]);
        let mut uninterrupted = plan.prepare_cpu().unwrap();
        uninterrupted
            .step(input(), TensorData::scalar(0.01))
            .unwrap();
        let checkpoint = uninterrupted.checkpoint().unwrap();
        let decoded = decode_adamw_checkpoint(checkpoint.as_bytes()).unwrap();
        assert_eq!(
            decoded.parameters.keys().cloned().collect::<Vec<_>>(),
            expected_parameter_keys
        );
        assert_eq!(
            decoded.first_moments.keys().cloned().collect::<Vec<_>>(),
            expected_parameter_keys
        );
        assert_eq!(
            decoded.second_moments.keys().cloned().collect::<Vec<_>>(),
            expected_parameter_keys
        );
        assert_eq!(
            decoded
                .gradient_accumulators
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            expected_parameter_keys
        );

        let fresh = FineTuneModule::new();
        let mut resumed = CompiledAdamWPlan::compile_module_from_checkpoint(
            config,
            &fresh,
            &checkpoint,
            build_fine_tune,
        )
        .unwrap()
        .prepare_cpu()
        .unwrap();
        assert!(uninterrupted.zero_grad().unwrap().did_discard());
        assert!(resumed.zero_grad().unwrap().did_discard());
        uninterrupted
            .step(input(), TensorData::scalar(0.01))
            .unwrap();
        resumed.step(input(), TensorData::scalar(0.01)).unwrap();
        assert!(
            uninterrupted
                .flush_partial_window(TensorData::scalar(0.01))
                .unwrap()
                .did_update()
        );
        assert!(
            resumed
                .flush_partial_window(TensorData::scalar(0.01))
                .unwrap()
                .did_update()
        );
        assert_eq!(
            uninterrupted.checkpoint().unwrap(),
            resumed.checkpoint().unwrap()
        );
        let publication = FineTuneModule::new();
        let publication_base = publication.base.snapshot().unwrap();
        let publication_adapter_version = publication.adapter.version().unwrap();
        assert!(
            uninterrupted
                .publish_parameters(&publication)
                .unwrap()
                .is_clean()
        );
        assert_parameter_snapshot_eq(&publication.base.snapshot().unwrap(), &publication_base);
        assert_eq!(
            publication.adapter.value().unwrap(),
            uninterrupted.parameter_snapshots().unwrap()["adapter"]
        );
        assert_eq!(
            publication.adapter.version().unwrap(),
            publication_adapter_version + 1
        );
        assert_parameter_snapshot_eq(&module.base.snapshot().unwrap(), &base);
        assert_parameter_snapshot_eq(&module.adapter.snapshot().unwrap(), &adapter);
    }

    #[test]
    fn adamw_compile_time_freezing_excludes_tied_gradients_from_global_clipping() {
        let clipped = module_config().with_max_gradient_norm(1.0).unwrap();
        let frozen_config = clipped.clone().with_frozen_parameters(["base"]).unwrap();
        let frozen_module = FineTuneModule::new();
        let frozen_plan =
            CompiledAdamWPlan::compile_module(frozen_config, &frozen_module, build_fine_tune_clip)
                .unwrap();
        let metal = frozen_plan
            .metal_plan(
                MetalRenderer::new(
                    8,
                    crate::runtime::metal::MetalCapabilities {
                        max_buffer_length: 1 << 30,
                        unified_memory: true,
                        family: "Apple9".into(),
                    },
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            metal
                .inner
                .state_input_keys
                .values()
                .filter_map(RecurrentStateKey::parameter_name)
                .collect::<Vec<_>>(),
            ["adapter"]
        );

        let input = || {
            BTreeMap::from([(
                "x".into(),
                TensorData::new([2], vec![1_000.0, -1_000.0]).unwrap(),
            )])
        };
        let mut frozen = frozen_plan.prepare_cpu().unwrap();
        frozen.step(input(), TensorData::scalar(0.01)).unwrap();
        let frozen_moment = frozen.first_moment_snapshots().unwrap()["adapter"].to_vec_f64();
        assert!(
            frozen_moment
                .iter()
                .all(|value| (*value - 0.05).abs() < 1e-6)
        );

        let unfrozen_module = FineTuneModule::new();
        let mut unfrozen =
            CompiledAdamWPlan::compile_module(clipped, &unfrozen_module, build_fine_tune_clip)
                .unwrap()
                .prepare_cpu()
                .unwrap();
        unfrozen.step(input(), TensorData::scalar(0.01)).unwrap();
        let unfrozen_moment = unfrozen.first_moment_snapshots().unwrap()["adapter"].to_vec_f64();
        assert!(unfrozen_moment.iter().all(|value| value.abs() < 1e-3));
        assert_ne!(frozen_moment, unfrozen_moment);
    }

    #[test]
    fn owned_adamw_freezing_publishes_only_the_unfrozen_frontier() {
        let module = FineTuneModule::new();
        let base = module.base.clone();
        let adapter = module.adapter.clone();
        let base_before = base.snapshot().unwrap();
        let adapter_before = adapter.snapshot().unwrap();
        let plan = CompiledModuleAdamWPlan::compile(
            module_config().with_frozen_parameters(["base"]).unwrap(),
            module,
            build_fine_tune,
        )
        .unwrap();
        let mut session = plan.prepare(&CpuSessionTarget::new()).unwrap();
        session
            .step(
                BTreeMap::from([("x".into(), TensorData::new([2], vec![0.5, -0.25]).unwrap())]),
                TensorData::scalar(0.01),
            )
            .unwrap();
        assert_eq!(
            session
                .parameter_snapshots()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["adapter"]
        );
        let module = session.finish().unwrap();
        assert_parameter_snapshot_eq(&module.base.snapshot().unwrap(), &base_before);
        assert_eq!(module.base.id(), base.id());
        assert!(module.base.is_trainable());
        assert_ne!(module.adapter.value().unwrap(), adapter_before.data);
        assert_eq!(
            module.adapter.version().unwrap(),
            adapter_before.version + 1
        );
        assert_eq!(module.adapter.id(), adapter.id());
        assert!(module.adapter.is_trainable());
    }

    #[test]
    fn adamw_config_rejects_invalid_hyperparameters_before_build() {
        for config in [
            CompiledAdamWConfig::new(-0.1, 0.999, 1e-8, 0.0),
            CompiledAdamWConfig::new(1.0, 0.999, 1e-8, 0.0),
            CompiledAdamWConfig::new(0.9, 1.0, 1e-8, 0.0),
            CompiledAdamWConfig::new(0.9, 0.999, 0.0, 0.0),
            CompiledAdamWConfig::new(0.9, 0.999, f32::NAN, 0.0),
            CompiledAdamWConfig::new(0.9, 0.999, 1e-8, -0.1),
        ] {
            assert!(config.is_err());
        }
        assert!(
            CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
                .unwrap()
                .with_gradient_accumulation(0)
                .is_err()
        );
        for max_norm in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
                    .unwrap()
                    .with_max_gradient_norm(max_norm)
                    .is_err()
            );
        }
        for loss_scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
                    .unwrap()
                    .with_loss_scale(loss_scale)
                    .is_err()
            );
        }
    }

    #[test]
    fn adamw_checkpoint_resume_matches_uninterrupted_replay_exactly() {
        let mut uninterrupted = compiled_adamw();
        let mut saved = compiled_adamw();
        for _ in 0..2 {
            uninterrupted.step(batch(), lr()).unwrap();
            saved.step(batch(), lr()).unwrap();
        }
        let checkpoint = saved.checkpoint().unwrap();
        assert_eq!(checkpoint, saved.checkpoint().unwrap());
        assert_eq!(
            CompiledAdamWCheckpoint::from_bytes(checkpoint.as_bytes().to_vec()).unwrap(),
            checkpoint
        );

        let mut resumed =
            CpuCompiledAdamW::compile_from_checkpoint(adamw_config(), &checkpoint, build_tinybob)
                .unwrap();
        assert_eq!(resumed.step_count(), 2);
        assert_eq!(resumed.optimizer_step().unwrap(), 2);
        assert_eq!(resumed.capture_identity(), saved.capture_identity());
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            saved.parameter_snapshots().unwrap()
        );
        assert_eq!(
            resumed.first_moment_snapshots().unwrap(),
            saved.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.second_moment_snapshots().unwrap(),
            saved.second_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.parameter_versions().unwrap(),
            saved.parameter_versions().unwrap()
        );
        assert_eq!(
            resumed.first_moment_versions().unwrap(),
            saved.first_moment_versions().unwrap()
        );
        assert_eq!(
            resumed.second_moment_versions().unwrap(),
            saved.second_moment_versions().unwrap()
        );

        let expected = uninterrupted.step(batch(), lr()).unwrap();
        let actual = resumed.step(batch(), lr()).unwrap();
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(
            resumed.parameter_snapshots().unwrap(),
            uninterrupted.parameter_snapshots().unwrap()
        );
        assert_eq!(
            resumed.first_moment_snapshots().unwrap(),
            uninterrupted.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.second_moment_snapshots().unwrap(),
            uninterrupted.second_moment_snapshots().unwrap()
        );
        assert_eq!(
            resumed.parameter_versions().unwrap(),
            uninterrupted.parameter_versions().unwrap()
        );
    }

    #[test]
    fn adamw_checkpoint_rejects_corruption_and_wrong_program_identity() {
        let checkpoint = compiled_adamw().checkpoint().unwrap();
        let mut corrupt = checkpoint.as_bytes().to_vec();
        corrupt.pop();
        assert!(CompiledAdamWCheckpoint::from_bytes(corrupt).is_err());

        let wrong = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.02)
            .unwrap()
            .with_input("x", [4, 2], DType::F32)
            .unwrap()
            .with_input("target", [4], DType::I64)
            .unwrap();
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(wrong, &checkpoint, build_tinybob).is_err()
        );
        let clipped = adamw_config().with_max_gradient_norm(1.0).unwrap();
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(clipped, &checkpoint, build_tinybob).is_err()
        );
        let scaled = adamw_config().with_loss_scale(128.0).unwrap();
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(scaled, &checkpoint, build_tinybob).is_err()
        );

        let mut accumulated = CpuCompiledAdamW::compile(
            accumulated_adamw_config(2),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        accumulated.step(batch(), lr()).unwrap();
        let checkpoint = accumulated.checkpoint().unwrap();
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(adamw_config(), &checkpoint, build_tinybob,)
                .is_err()
        );
    }

    #[test]
    fn compiled_multi_step_lr_validates_immutable_schedule() {
        let schedule = CompiledMultiStepLr::new(0.05, 0.25, [1, 3, 8]).unwrap();
        assert_eq!(schedule.base(), 0.05);
        assert_eq!(schedule.gamma(), 0.25);
        assert_eq!(schedule.milestones(), &[1, 3, 8]);
        assert!(CompiledMultiStepLr::new(f32::NAN, 0.5, []).is_err());
        assert!(CompiledMultiStepLr::new(-0.1, 0.5, []).is_err());
        assert!(CompiledMultiStepLr::new(0.1, f32::INFINITY, []).is_err());
        assert!(CompiledMultiStepLr::new(0.1, -0.5, []).is_err());
        assert!(CompiledMultiStepLr::new(0.1, 0.5, [0]).is_err());
        assert!(CompiledMultiStepLr::new(0.1, 0.5, [1, 1]).is_err());
        assert!(CompiledMultiStepLr::new(0.1, 0.5, [2, 1]).is_err());
        assert!(CompiledMultiStepLr::new(0.1, 0.5, [u64::MAX]).is_err());
        assert!(CompiledMultiStepLr::new(f32::MAX, 2.0, [1]).is_err());
        assert!(CompiledMultiStepLr::new(f32::MAX / 2.0, 1.5, [1, 2]).is_err());
    }

    #[test]
    fn compiled_multi_step_lr_matches_external_updates_flush_and_restore() {
        let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1, 2]).unwrap();
        let scheduled_config =
            accumulated_adamw_config(2).with_captured_multi_step_lr(schedule.clone());
        assert_eq!(scheduled_config.captured_multi_step_lr(), Some(&schedule));
        let mut scheduled = CpuCompiledAdamW::compile(
            scheduled_config.clone(),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        let mut external = CpuCompiledAdamW::compile(
            accumulated_adamw_config(2),
            initial_parameters(),
            build_tinybob,
        )
        .unwrap();
        assert_eq!(scheduled.captured_multi_step_lr(), Some(&schedule));
        assert_eq!(external.captured_multi_step_lr(), None);
        assert_ne!(scheduled.capture_identity(), external.capture_identity());
        assert!(
            !scheduled
                .inner
                .capture
                .schedule
                .inputs
                .iter()
                .any(|input| input.name == LEARNING_RATE_INPUT)
        );
        assert!(
            external
                .inner
                .capture
                .schedule
                .inputs
                .iter()
                .any(|input| input.name == LEARNING_RATE_INPUT)
        );
        assert!(
            !scheduled
                .partial_flush
                .as_ref()
                .unwrap()
                .capture
                .schedule
                .inputs
                .iter()
                .any(|input| input.name == LEARNING_RATE_INPUT)
        );
        assert!(
            external
                .partial_flush
                .as_ref()
                .unwrap()
                .capture
                .schedule
                .inputs
                .iter()
                .any(|input| input.name == LEARNING_RATE_INPUT)
        );

        let scheduled_before = scheduled.checkpoint().unwrap();
        assert!(scheduled.step(batch(), TensorData::scalar(0.05)).is_err());
        assert_eq!(scheduled.checkpoint().unwrap(), scheduled_before);
        let external_before = external.checkpoint().unwrap();
        assert!(external.step_scheduled(batch()).is_err());
        assert_eq!(external.checkpoint().unwrap(), external_before);

        for external_rate in [0.05, 0.05, 0.025] {
            let actual = scheduled.step_scheduled(batch()).unwrap();
            let expected = external
                .step(batch(), TensorData::scalar(external_rate))
                .unwrap();
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.did_update(), expected.did_update());
        }
        let scheduled_before_wrong_flush = scheduled.checkpoint().unwrap();
        assert!(
            scheduled
                .flush_partial_window(TensorData::scalar(0.025))
                .is_err()
        );
        assert_eq!(
            scheduled.checkpoint().unwrap(),
            scheduled_before_wrong_flush
        );
        let actual = scheduled.flush_partial_window_scheduled().unwrap();
        let expected = external
            .flush_partial_window(TensorData::scalar(0.025))
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(scheduled.checkpoint().unwrap().info().optimizer_step(), 2);
        assert_eq!(
            scheduled.parameter_snapshots().unwrap(),
            external.parameter_snapshots().unwrap()
        );
        assert_eq!(
            scheduled.first_moment_snapshots().unwrap(),
            external.first_moment_snapshots().unwrap()
        );
        assert_eq!(
            scheduled.second_moment_snapshots().unwrap(),
            external.second_moment_snapshots().unwrap()
        );

        let checkpoint = scheduled.checkpoint().unwrap();
        let mut resumed = CpuCompiledAdamW::compile_from_checkpoint(
            scheduled_config.clone(),
            &checkpoint,
            build_tinybob,
        )
        .unwrap();
        for _ in 0..2 {
            let expected = scheduled.step_scheduled(batch()).unwrap();
            let actual = resumed.step_scheduled(batch()).unwrap();
            assert_eq!(actual.loss(), expected.loss());
            assert_eq!(actual.outputs(), expected.outputs());
        }
        assert_eq!(
            resumed.checkpoint().unwrap(),
            scheduled.checkpoint().unwrap()
        );
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(
                accumulated_adamw_config(2),
                &checkpoint,
                build_tinybob,
            )
            .is_err()
        );
        let wrong_schedule = accumulated_adamw_config(2)
            .with_captured_multi_step_lr(CompiledMultiStepLr::new(0.05, 0.5, [1, 3]).unwrap());
        assert!(
            CpuCompiledAdamW::compile_from_checkpoint(wrong_schedule, &checkpoint, build_tinybob)
                .is_err()
        );

        let scheduled_plan =
            CompiledAdamWPlan::compile(scheduled_config, initial_parameters(), build_tinybob)
                .unwrap();
        assert_eq!(scheduled_plan.captured_multi_step_lr(), Some(&schedule));
        let renderer = MetalRenderer::new(
            8,
            crate::runtime::metal::MetalCapabilities {
                max_buffer_length: 1 << 30,
                unified_memory: true,
                family: "Apple9".into(),
            },
        )
        .unwrap();
        assert!(scheduled_plan.metal_plan(renderer).is_err());
    }
}
