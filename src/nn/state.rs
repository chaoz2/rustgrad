//! Deterministic module traversal and state loading.

use super::parameter::next_version;
use super::{
    Parameter, ParameterId, ParameterRestore, ParameterSnapshot,
    norm::{BatchNorm, PendingBatchNormStats},
    restore_parameters,
};
use crate::{DType, Error, Graph, NodeId, Result, TensorData, TrainingContext, load_safetensors};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::Read,
    path::Path,
};

pub enum StateKind {
    Parameter,
    Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastPolicy {
    Exact,
    Allow,
}

/// Resource limit for the strict local safetensors convenience route.
///
/// The parser still owns safetensors syntax and schema validation; this limit
/// only bounds how many local file or caller-provided bytes reach that parser.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StrictStateLoadLimits {
    pub max_safetensors_bytes: usize,
}

impl Default for StrictStateLoadLimits {
    fn default() -> Self {
        Self {
            max_safetensors_bytes: 1 << 30,
        }
    }
}

/// Explicit execution mode for stateful module forwards.
///
/// [`ModeModuleForward::forward_mode`] remains the authoritative explicit
/// route. [`ModeModuleForward::forward_ambient`] derives this value from the
/// scoped, thread-local [`TrainingContext`] without introducing process-global
/// mutable module state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Training,
    Eval,
}

/// Output from a mode-aware module forward.
///
/// The caller realizes `output` first.  In training mode it then realizes the
/// stat nodes exposed by `pending` and commits them explicitly.  Evaluation is
/// therefore read-only; this type never performs an implicit state update.
pub struct ModeForwardOutput<'a> {
    pub output: NodeId,
    pub pending: PendingModeEffects<'a>,
}

/// A one-input module forward whose mode and pending state changes remain
/// visible to its caller.  This deliberately does not extend [`ModuleForward`]
/// because that trait promises a state-free one-input/one-output composition.
pub trait ModeModuleForward: Module {
    /// Composes this module under an explicit caller-selected mode.
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>>;

    /// Composes this module under the current scoped [`TrainingContext`].
    ///
    /// The complete graph change is staged on a clone before publication.
    /// Stateful work remains visible in [`ModeForwardOutput::pending`] and is
    /// never committed implicitly. Explicit [`Self::forward_mode`] behavior is
    /// unchanged.
    fn forward_ambient<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
    ) -> Result<ModeForwardOutput<'a>> {
        let mode = if TrainingContext::is_training() {
            Mode::Training
        } else {
            Mode::Eval
        };
        let mut candidate = graph.clone();
        let output = self.forward_mode(&mut candidate, input, mode)?;
        *graph = candidate;
        Ok(output)
    }
}

/// State-free modules participate in an explicit mode path without gaining an
/// implicit mode or mutation contract.  `BatchNorm` supplies its own explicit
/// implementation because it may return pending running-stat effects.
impl<T: ModuleForward + ?Sized> ModeModuleForward for T {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        _mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        Ok(ModeForwardOutput {
            output: self.forward(graph, input)?,
            pending: PendingModeEffects::empty(),
        })
    }
}

/// Realized values for one pending mode effect, in the deterministic order
/// returned by [`PendingModeEffects::batchnorm_stat_nodes`].
pub struct RealizedBatchNormStats {
    pub mean: TensorData,
    pub variance: TensorData,
}

/// Explicit, typed pending state work.  The enum leaves room for later
/// stateful modules without making BatchNorm mutation implicit.
pub enum PendingModeEffect<'a> {
    BatchNorm {
        module: &'a BatchNorm,
        stats: PendingBatchNormStats,
    },
}

/// A deterministic transaction for pending stateful-module effects.
///
/// It initially supports BatchNorm statistics.  `commit_batchnorm` reserves
/// every token, validates every candidate, then uses the existing all-lock
/// parameter restore transaction to replace every running buffer together.
/// A failed preparation or restore releases every reservation, leaving all
/// buffers and tokens retryable.
#[derive(Default)]
pub struct PendingModeEffects<'a> {
    effects: Vec<PendingModeEffect<'a>>,
}

impl<'a> PendingModeEffects<'a> {
    pub fn empty() -> Self {
        Self::default()
    }

    pub(crate) fn batchnorm(module: &'a BatchNorm, stats: PendingBatchNormStats) -> Self {
        Self {
            effects: vec![PendingModeEffect::BatchNorm { module, stats }],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Adds effects built by a later module in the same explicit mode-aware
    /// forward.  Insertion order is the transaction's deterministic order.
    pub fn append(&mut self, mut later: Self) {
        self.effects.append(&mut later.effects);
    }

    /// Returns graph-local `(mean, variance)` nodes in commit order.
    pub fn batchnorm_stat_nodes(&self) -> Vec<(NodeId, NodeId)> {
        self.effects
            .iter()
            .map(|effect| match effect {
                PendingModeEffect::BatchNorm { stats, .. } => (stats.mean, stats.variance),
            })
            .collect()
    }

    /// Atomically commits all BatchNorm effects after the caller has already
    /// successfully realized its requested output/loss/gradient nodes.
    pub fn commit_batchnorm(&self, realized: Vec<RealizedBatchNormStats>) -> Result<()> {
        self.commit_batchnorm_with(realized, Vec::new())
    }

    /// Adds already-prepared trainable-parameter replacements to the same
    /// all-lock commit as the pending running-buffer effects.  This is kept
    /// crate-private so public callers retain the small explicit BatchNorm
    /// capability API while CPU training can make its optimizer update and
    /// running-stat update one visible state transition.
    pub(crate) fn commit_batchnorm_with(
        &self,
        realized: Vec<RealizedBatchNormStats>,
        mut additional: Vec<ParameterRestore>,
    ) -> Result<()> {
        if realized.len() != self.effects.len() {
            return Err(Error::BatchNormToken {
                reason: "pending statistics count mismatch",
            });
        }
        let mut reserved = Vec::with_capacity(self.effects.len());
        let result = (|| {
            let mut restores = Vec::<ParameterRestore>::new();
            let mut targets = BTreeSet::new();
            for (effect, values) in self.effects.iter().zip(realized) {
                let PendingModeEffect::BatchNorm { module, stats } = effect;
                stats.reserve()?;
                reserved.push(stats);
                let candidates = stats.prepare(module, values.mean, values.variance)?;
                for candidate in &candidates {
                    if !targets.insert(candidate.parameter.identity()) {
                        return Err(Error::BatchNormToken {
                            reason: "duplicate pending mode effect target",
                        });
                    }
                }
                restores.extend(candidates);
            }
            restores.append(&mut additional);
            restore_parameters(restores)
        })();
        if result.is_ok() {
            // Keeping the reservation set makes every successfully committed
            // token deterministically one-shot.
            return Ok(());
        }
        for stats in reserved {
            stats.release();
        }
        result
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub missing_keys: Vec<String>,
    pub unexpected_keys: Vec<String>,
    pub shape_mismatches: Vec<String>,
    pub dtype_mismatches: Vec<String>,
    pub loaded_keys: Vec<String>,
}
impl LoadReport {
    pub fn is_clean(&self) -> bool {
        self.missing_keys.is_empty()
            && self.unexpected_keys.is_empty()
            && self.shape_mismatches.is_empty()
            && self.dtype_mismatches.is_empty()
    }
}

/// A deterministic state map that converts directly to RustGrad safetensors maps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StateDict {
    tensors: BTreeMap<String, TensorData>,
}
impl StateDict {
    pub fn tensors(&self) -> &BTreeMap<String, TensorData> {
        &self.tensors
    }
    pub fn into_tensors(self) -> BTreeMap<String, TensorData> {
        self.tensors
    }
    pub fn insert(&mut self, name: impl Into<String>, value: TensorData) {
        self.tensors.insert(name.into(), value);
    }
}
impl From<BTreeMap<String, TensorData>> for StateDict {
    fn from(tensors: BTreeMap<String, TensorData>) -> Self {
        Self { tensors }
    }
}
impl From<StateDict> for BTreeMap<String, TensorData> {
    fn from(value: StateDict) -> Self {
        value.tensors
    }
}

/// Ordered live parameter handles collected from an explicit module traversal.
///
/// Entries retain declaration order. Repeated names replace their prior value
/// in place, while tied handles under different names remain distinct entries.
#[derive(Clone, Debug, Default)]
pub struct LiveStateDict {
    entries: Vec<(String, Parameter)>,
}
impl LiveStateDict {
    pub fn get(&self, name: &str) -> Option<&Parameter> {
        self.entries
            .iter()
            .find_map(|(entry_name, parameter)| (entry_name == name).then_some(parameter))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&str, &Parameter)> {
        self.entries
            .iter()
            .map(|(name, parameter)| (name.as_str(), parameter))
    }

    pub fn keys(&self) -> impl ExactSizeIterator<Item = &str> {
        self.entries.iter().map(|(name, _)| name.as_str())
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = &Parameter> {
        self.entries.iter().map(|(_, parameter)| parameter)
    }

    pub fn into_entries(self) -> Vec<(String, Parameter)> {
        self.entries
    }

    fn insert(&mut self, name: String, parameter: Parameter) {
        if let Some((_, existing)) = self
            .entries
            .iter_mut()
            .find(|(entry_name, _)| entry_name.as_str() == name.as_str())
        {
            *existing = parameter;
        } else {
            self.entries.push((name, parameter));
        }
    }
}

/// Rust-native explicit state traversal. Implementors call `visit` for fields,
/// nested modules, vectors, and options in their declared deterministic order.
pub trait Module {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind));

    /// Returns canonical trainable parameter bindings for an optimizer.
    ///
    /// Traversal names are sorted deterministically. Shared/tied identities are
    /// emitted once at their first traversal name, and lock poisoning is
    /// reported before an optimizer can allocate or mutate state.
    fn trainable_parameters(&self) -> Result<Vec<(String, Parameter)>> {
        let mut parameters = BTreeMap::new();
        let mut identities = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut error = None;
        self.visit("", &mut |name, parameter, kind| {
            if error.is_some() || !matches!(kind, StateKind::Parameter) {
                return;
            }
            if !names.insert(name.clone()) {
                error = Some(Error::Serialization {
                    reason: format!("module traversal contains duplicate trainable key {name:?}"),
                });
                return;
            }
            match parameter.snapshot() {
                Ok(snapshot) => {
                    if snapshot.trainable && identities.insert(snapshot.identity) {
                        parameters.insert(name, parameter.clone());
                    }
                }
                Err(err) => error = Some(err),
            }
        });
        match error {
            Some(error) => Err(error),
            None => Ok(parameters.into_iter().collect()),
        }
    }

    fn state_dict(&self) -> Result<StateDict> {
        let mut tensors = BTreeMap::new();
        let mut seen = BTreeSet::new();
        let mut error = None;
        self.visit("", &mut |name, parameter, _| {
            if seen.insert(parameter.identity()) {
                match parameter.snapshot() {
                    Ok(snapshot) => {
                        tensors.insert(name, snapshot.data);
                    }
                    Err(err) => error = Some(err),
                }
            }
        });
        match error {
            Some(err) => Err(err),
            None => Ok(StateDict { tensors }),
        }
    }
    fn input_bindings(&self, graph: &Graph) -> Result<HashMap<String, TensorData>> {
        Ok(graph.parameter_bindings_for(&module_parameter_identities(self)?))
    }
    fn load_state_dict(
        &self,
        state: &StateDict,
        strict: bool,
        cast: CastPolicy,
    ) -> Result<LoadReport> {
        let mut entries = BTreeMap::<String, (Parameter, ParameterSnapshot)>::new();
        let mut seen = BTreeSet::new();
        let mut error = None;
        self.visit("", &mut |name, parameter, _| {
            if !seen.insert(parameter.identity()) {
                return;
            }
            if entries.contains_key(&name) {
                error = Some(Error::Serialization {
                    reason: format!("module traversal contains duplicate state key {name:?}"),
                });
                return;
            }
            match parameter.snapshot() {
                Ok(snapshot) => {
                    entries.insert(name, (parameter.clone(), snapshot));
                }
                Err(err) => error = Some(err),
            }
        });
        if let Some(err) = error {
            return Err(err);
        }
        let mut report = LoadReport::default();
        let mut restores = Vec::new();
        let mut loaded_keys = Vec::new();
        for (name, (parameter, snapshot)) in &entries {
            let Some(value) = state.tensors.get(name) else {
                report.missing_keys.push(name.clone());
                continue;
            };
            // tinygrad's loader admits only this one shape relaxation: a
            // scalar and a rank-one singleton carry the same single storage
            // lane, so it reshapes the incoming value to the parameter shape
            // before replacement.  Keep it in this preflight phase and clone
            // raw storage so narrow payloads, NaNs, and signed zero survive.
            let value = if value.shape() != &snapshot.shape {
                if singleton_scalar_rank_one_pair(value, snapshot) {
                    TensorData::from_storage(snapshot.shape.clone(), value.storage().clone())?
                } else {
                    report.shape_mismatches.push(name.clone());
                    continue;
                }
            } else {
                value.clone()
            };
            let value = if value.dtype() != snapshot.dtype {
                if cast == CastPolicy::Allow {
                    value.cast(snapshot.dtype)
                } else {
                    report.dtype_mismatches.push(name.clone());
                    continue;
                }
            } else {
                value
            };
            restores.push(ParameterRestore {
                parameter: parameter.clone(),
                data: value,
                expected_version: snapshot.version,
                restored_version: next_version(snapshot.version)?,
            });
            loaded_keys.push(name.clone());
        }
        report.unexpected_keys = state
            .tensors
            .keys()
            .filter(|name| !entries.contains_key(*name))
            .cloned()
            .collect();
        if strict && !report.is_clean() {
            return Err(Error::Serialization {
                reason: format!(
                    "state_dict mismatch: missing={:?}, unexpected={:?}, shape={:?}, dtype={:?}",
                    report.missing_keys,
                    report.unexpected_keys,
                    report.shape_mismatches,
                    report.dtype_mismatches
                ),
            });
        }
        // `restore_parameters` locks and rechecks every target before writing
        // one of them, so even a racing version change cannot leave a strict
        // or non-strict load partially visible.
        restore_parameters(restores)?;
        report.loaded_keys = loaded_keys;
        Ok(report)
    }

    /// Loads an exact, complete decoded state map transactionally.
    ///
    /// This is the canonical strict state boundary: keys, shapes, and dtypes
    /// must match the module's deterministic traversal, and no cast or partial
    /// parameter update is permitted.
    fn load_state_dict_strict(&self, state: &StateDict) -> Result<LoadReport> {
        self.load_state_dict(state, true, CastPolicy::Exact)
    }

    /// Decodes and strictly loads a bounded safetensors byte stream.
    fn load_safetensors_strict(&self, bytes: &[u8]) -> Result<LoadReport> {
        self.load_safetensors_strict_with_limits(bytes, StrictStateLoadLimits::default())
    }

    /// Decodes and strictly loads safetensors bytes under an explicit limit.
    fn load_safetensors_strict_with_limits(
        &self,
        bytes: &[u8],
        limits: StrictStateLoadLimits,
    ) -> Result<LoadReport> {
        check_safetensors_len(bytes.len(), limits)?;
        let (state, _) = load_safetensors(bytes)?;
        self.load_state_dict_strict(&StateDict::from(state))
    }

    /// Reads, decodes, and strictly loads a bounded local safetensors file.
    fn load_safetensors_file_strict(&self, path: &Path) -> Result<LoadReport> {
        self.load_safetensors_file_strict_with_limits(path, StrictStateLoadLimits::default())
    }

    /// Reads, decodes, and strictly loads a local safetensors file under an
    /// explicit byte limit. The file is copied into bounded owned bytes; no
    /// mapping or device-backed state is created.
    fn load_safetensors_file_strict_with_limits(
        &self,
        path: &Path,
        limits: StrictStateLoadLimits,
    ) -> Result<LoadReport> {
        let bytes = read_safetensors_file_bounded(path, limits)?;
        self.load_safetensors_strict_with_limits(&bytes, limits)
    }
}

fn module_parameter_identities(module: &(impl Module + ?Sized)) -> Result<BTreeSet<ParameterId>> {
    let mut seen = BTreeSet::new();
    let mut error = None;
    module.visit("", &mut |_, parameter, _| match parameter.snapshot() {
        Ok(snapshot) => {
            seen.insert(snapshot.identity);
        }
        Err(err) => error = Some(err),
    });
    match error {
        Some(err) => Err(err),
        None => Ok(seen),
    }
}

pub(crate) fn module_input_node_bindings(
    module: &(impl Module + ?Sized),
    graph: &Graph,
) -> Result<HashMap<String, (NodeId, TensorData)>> {
    Ok(graph.parameter_node_bindings_for(&module_parameter_identities(module)?))
}

/// A statically shaped module that composes one graph input into one graph
/// output. This is the canonical typed forward seam for CPU module workflows;
/// it deliberately does not erase distinct multi-input or stateful signatures.
pub trait ModuleForward: Module {
    /// Returns whether this module can consume the workflow's external input
    /// dtype. Ordinary static CPU modules accept F32 features; modules such as
    /// Embedding may explicitly opt into a different typed input contract.
    fn accepts_input_dtype(&self, dtype: DType) -> bool {
        dtype == DType::F32
    }

    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId>;
}

fn check_safetensors_len(actual: usize, limits: StrictStateLoadLimits) -> Result<()> {
    if actual > limits.max_safetensors_bytes {
        return Err(Error::Serialization {
            reason: format!(
                "safetensors input has {actual} bytes, exceeding strict state limit {}",
                limits.max_safetensors_bytes
            ),
        });
    }
    Ok(())
}

fn read_safetensors_file_bounded(path: &Path, limits: StrictStateLoadLimits) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path).map_err(|error| Error::Serialization {
        reason: format!("failed to inspect safetensors file: {error}"),
    })?;
    let actual = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
    check_safetensors_len(actual, limits)?;
    let file = fs::File::open(path).map_err(|error| Error::Serialization {
        reason: format!("failed to open safetensors file: {error}"),
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(limits.max_safetensors_bytes.min(64 << 10))
        .map_err(|_| Error::Serialization {
            reason: "failed to allocate safetensors input buffer".into(),
        })?;
    file.take(
        u64::try_from(limits.max_safetensors_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    )
    .read_to_end(&mut bytes)
    .map_err(|error| Error::Serialization {
        reason: format!("failed to read safetensors file: {error}"),
    })?;
    check_safetensors_len(bytes.len(), limits)?;
    Ok(bytes)
}

/// Returns live parameter handles in tinygrad's explicit state-dict traversal order.
///
/// This is exactly the values of [`get_state_dict`] at zero prefix.
pub fn get_parameters(module: &dyn Module) -> Vec<Parameter> {
    get_state_dict(module, "")
        .into_entries()
        .into_iter()
        .map(|(_, parameter)| parameter)
        .collect()
}

/// Collects live module handles with tinygrad's raw prefix and dict semantics.
///
/// The prefix is concatenated directly onto each name emitted by a zero-prefix
/// visit. Collection clones handles only: it does not snapshot, lock, sort, or
/// deduplicate them.
pub fn get_state_dict(module: &dyn Module, prefix: &str) -> LiveStateDict {
    let mut state = LiveStateDict::default();
    module.visit("", &mut |name, parameter, _| {
        state.insert(format!("{prefix}{name}"), parameter.clone());
    });
    state
}

/// The only load-time shape adaptation accepted by tinygrad state loading.
/// Both descriptors have exactly one element, so rebuilding the descriptor
/// from cloned storage is a checked descriptor change rather than a broadcast
/// or a value conversion.
fn singleton_scalar_rank_one_pair(value: &TensorData, snapshot: &ParameterSnapshot) -> bool {
    (value.shape().rank() == 0 && snapshot.shape.rank() == 1 && snapshot.shape.dims() == [1])
        || (value.shape().rank() == 1 && value.shape().dims() == [1] && snapshot.shape.rank() == 0)
}

pub(super) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.into()
    } else {
        format!("{prefix}.{name}")
    }
}
