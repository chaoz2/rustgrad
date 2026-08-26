//! Deterministic module traversal and state loading.

use super::{
    Parameter, ParameterRestore, ParameterSnapshot,
    norm::{BatchNorm, PendingBatchNormStats},
    restore_parameters,
};
use crate::{Error, Graph, NodeId, Result, TensorData, load_safetensors};
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

/// Explicit execution mode. It is passed to stateful normalization forwards;
/// RustGrad deliberately has no process-global training flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Training,
    Eval,
}

/// Output from an explicit mode-aware module forward.
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
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>>;
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
        let mut seen = BTreeSet::new();
        let mut error = None;
        self.visit("", &mut |_, parameter, _| match parameter.snapshot() {
            Ok(snapshot) => {
                seen.insert(snapshot.identity);
            }
            Err(err) => error = Some(err),
        });
        match error {
            Some(err) => Err(err),
            None => Ok(graph.parameter_bindings_for(&seen)),
        }
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
        for (name, (parameter, snapshot)) in &entries {
            let Some(value) = state.tensors.get(name) else {
                report.missing_keys.push(name.clone());
                continue;
            };
            if value.shape() != &snapshot.shape {
                report.shape_mismatches.push(name.clone());
                continue;
            }
            let value = if value.dtype() != snapshot.dtype {
                if cast == CastPolicy::Allow {
                    value.cast(snapshot.dtype)
                } else {
                    report.dtype_mismatches.push(name.clone());
                    continue;
                }
            } else {
                value.clone()
            };
            restores.push(ParameterRestore {
                parameter: parameter.clone(),
                data: value,
                expected_version: snapshot.version,
                restored_version: snapshot.version.wrapping_add(1),
            });
            report.loaded_keys.push(name.clone());
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

/// A statically shaped module that composes one graph input into one graph
/// output. This is the canonical typed forward seam for CPU module workflows;
/// it deliberately does not erase distinct multi-input or stateful signatures.
pub trait ModuleForward: Module {
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

pub(super) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.into()
    } else {
        format!("{prefix}.{name}")
    }
}
