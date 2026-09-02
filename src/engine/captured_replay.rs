//! Graph-independent interpreter/native replay and deterministic batching.
use super::capture::{CapturedSchedule, ReplayError};
use super::replay_liveness::ReplayLivenessPlan;
use crate::backend::{JitBackendError, PreparedScheduleItem, TensorValueStore};
use crate::{
    BufferRole, CpuJitBackend, ItemBackend, JitFallback, KernelBindings, KernelBufferDesc,
    ScheduleItem, TensorData,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapturedBackendPolicy {
    Interpreter,
    NativeJit { vectorized: bool },
    JitFallback { vectorized: bool },
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
enum ReplayValue {
    Materialized(TensorData),
    PrunedZeroDomain {
        descriptor: crate::BufferDesc,
        producer_item: u64,
        reason: String,
    },
}
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub(crate) struct ReplayValues(BTreeMap<u64, ReplayValue>);
#[allow(dead_code)]
impl ReplayValues {
    pub(crate) fn from_materialized(values: BTreeMap<u64, TensorData>) -> Self {
        Self(
            values
                .into_iter()
                .map(|(id, value)| (id, ReplayValue::Materialized(value)))
                .collect(),
        )
    }
    pub(crate) fn tensor(&self, id: u64, context: &str) -> Result<&TensorData, ReplayError> {
        match self.0.get(&id) {
            Some(ReplayValue::Materialized(value)) => Ok(value),
            Some(ReplayValue::PrunedZeroDomain { .. }) => Err(ReplayError::Corrupt(format!(
                "{context}: pruned value {id} read"
            ))),
            None => Err(ReplayError::Missing(id.to_string())),
        }
    }
    pub(crate) fn insert_tensor(&mut self, id: u64, value: TensorData) {
        self.0.insert(id, ReplayValue::Materialized(value));
    }
    fn insert_pruned(&mut self, id: u64, descriptor: crate::BufferDesc, producer_item: u64) {
        self.0.insert(
            id,
            ReplayValue::PrunedZeroDomain {
                descriptor,
                producer_item,
                reason: "only demanded by a pure zero-domain result".into(),
            },
        );
    }
    fn requested(&self, requested: &[u64]) -> Result<Vec<TensorData>, ReplayError> {
        requested
            .iter()
            .map(|id| self.tensor(*id, "requested output").cloned())
            .collect()
    }
}
#[allow(dead_code)]
impl TensorValueStore for ReplayValues {
    fn tensor(&self, id: u64, context: &str) -> Result<&TensorData, JitBackendError> {
        self.tensor(id, context)
            .map_err(|e| JitBackendError::Binding(e.to_string()))
    }
}

#[cfg(test)]
mod replay_values_tests {
    use super::*;
    #[test]
    fn pruned_zero_domain_rejects_live_tensor_lookup() {
        let mut values = ReplayValues::default();
        values.0.insert(
            7,
            ReplayValue::PrunedZeroDomain {
                descriptor: crate::BufferDesc {
                    id: 7,
                    shape: crate::Shape::from([0]),
                    dtype: crate::DType::F32,
                    bytes: 0,
                    alignment: 4,
                    read_only: true,
                    view: None,
                },
                producer_item: 3,
                reason: "zero domain".into(),
            },
        );
        assert!(matches!(
            values.tensor(7, "test"),
            Err(ReplayError::Corrupt(_))
        ));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapturedReplayOptions {
    pub backend: CapturedBackendPolicy,
}
impl Default for CapturedReplayOptions {
    fn default() -> Self {
        Self {
            backend: CapturedBackendPolicy::Interpreter,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedItemTrace {
    pub invocation: usize,
    pub item: u64,
    pub backend: ItemBackend,
    pub schedule_cache_key: u64,
    pub native_cache_key: Option<String>,
    pub cache_hit: bool,
    pub lanes: usize,
    pub vector_main: usize,
    pub vector_tail: usize,
    /// Exact packed bytes bound to native code. No dense weight allocation is
    /// hidden behind this count.
    pub packed_weight_bytes: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CapturedReplayTrace {
    pub items: Vec<CapturedItemTrace>,
}

#[derive(Clone, Debug)]
pub struct CapturedReplayResult {
    pub outputs: Vec<TensorData>,
    pub trace: CapturedReplayTrace,
    pub specialization: Option<CapturedSpecializationTrace>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedSpecializationTrace {
    pub source_identity: u64,
    pub concrete_identity: u64,
    pub bindings: Vec<(u64, i64)>,
    pub cache_hit: bool,
}

#[derive(Clone, Debug)]
pub struct CapturedSpecialization {
    capture: Arc<CapturedSchedule>,
    trace: CapturedSpecializationTrace,
}
impl CapturedSpecialization {
    pub fn capture(&self) -> &CapturedSchedule {
        &self.capture
    }
    pub fn trace(&self) -> &CapturedSpecializationTrace {
        &self.trace
    }
}

#[derive(Clone, Debug)]
pub struct CapturedInvocation {
    bindings: BTreeMap<String, TensorData>,
    symbolic_bindings: BTreeMap<String, i64>,
}
impl CapturedInvocation {
    pub fn bindings(&self) -> &BTreeMap<String, TensorData> {
        &self.bindings
    }
    pub fn symbolic_bindings(&self) -> &BTreeMap<String, i64> {
        &self.symbolic_bindings
    }
}

#[derive(Clone, Debug)]
pub struct CapturedBatch {
    artifact_identity: u64,
    invocations: Vec<CapturedInvocation>,
}
impl CapturedBatch {
    pub fn new(
        capture: &CapturedSchedule,
        invocations: impl IntoIterator<Item = BTreeMap<String, TensorData>>,
    ) -> Result<Self, ReplayError> {
        if capture.is_symbolic() {
            return Err(ReplayError::Symbolic(
                "symbolic batches require CapturedBatch::new_symbolic".into(),
            ));
        }
        let invocations = invocations
            .into_iter()
            .enumerate()
            .map(|(index, bindings)| {
                validate_inputs(capture, &bindings).map_err(|error| ReplayError::Batch {
                    invocation: index,
                    reason: error.to_string(),
                })?;
                Ok(CapturedInvocation {
                    bindings,
                    symbolic_bindings: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>, ReplayError>>()?;
        Ok(Self {
            artifact_identity: capture.identity,
            invocations,
        })
    }
    pub fn new_symbolic(
        capture: &CapturedSchedule,
        invocations: impl IntoIterator<Item = (BTreeMap<String, i64>, BTreeMap<String, TensorData>)>,
    ) -> Result<Self, ReplayError> {
        let schema = capture.symbolic.as_ref().ok_or_else(|| {
            ReplayError::Symbolic("concrete artifact cannot form a symbolic batch".into())
        })?;
        let invocations = invocations
            .into_iter()
            .enumerate()
            .map(|(index, (symbolic_bindings, bindings))| {
                let canonical = schema
                    .canonical_bindings(&symbolic_bindings)
                    .map_err(|error| ReplayError::Batch {
                        invocation: index,
                        reason: error.to_string(),
                    })?;
                let specialized = super::symbolic::specialize_capture(capture, &canonical)
                    .map_err(|error| ReplayError::Batch {
                        invocation: index,
                        reason: error.to_string(),
                    })?;
                validate_inputs(&specialized, &bindings).map_err(|error| ReplayError::Batch {
                    invocation: index,
                    reason: error.to_string(),
                })?;
                Ok(CapturedInvocation {
                    bindings,
                    symbolic_bindings,
                })
            })
            .collect::<Result<Vec<_>, ReplayError>>()?;
        Ok(Self {
            artifact_identity: capture.identity,
            invocations,
        })
    }
    pub fn len(&self) -> usize {
        self.invocations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.invocations.is_empty()
    }
    pub fn invocations(&self) -> &[CapturedInvocation] {
        &self.invocations
    }
}

#[derive(Clone, Debug)]
pub struct CapturedBatchResult {
    pub invocations: Vec<CapturedReplayResult>,
}

type SpecializationKey = (u64, Vec<(u64, i64)>);
type SpecializationCache = BTreeMap<SpecializationKey, Arc<CapturedSchedule>>;

pub struct CapturedReplayExecutor {
    scalar: CpuJitBackend,
    vectorized: CpuJitBackend,
    specializations: Mutex<SpecializationCache>,
}
impl Default for CapturedReplayExecutor {
    fn default() -> Self {
        Self {
            scalar: CpuJitBackend::new(JitFallback::Error),
            vectorized: CpuJitBackend::new(JitFallback::Error).vectorized(true),
            specializations: Mutex::new(BTreeMap::new()),
        }
    }
}
impl CapturedReplayExecutor {
    pub fn compile_cache_len(&self, vectorized: bool) -> usize {
        self.jit(vectorized).cache_len()
    }

    pub fn specialization_cache_len(&self) -> usize {
        self.specializations
            .lock()
            .expect("specialization cache lock")
            .len()
    }

    /// Evaluates one complete symbolic environment and returns a concrete,
    /// graph-independent artifact. Canonical symbol IDs and values key this
    /// process-local specialization cache.
    pub fn specialize(
        &self,
        capture: &CapturedSchedule,
        bindings: &BTreeMap<String, i64>,
    ) -> Result<CapturedSpecialization, ReplayError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let schema = capture
            .symbolic
            .as_ref()
            .ok_or_else(|| ReplayError::Symbolic("artifact is already concrete".into()))?;
        let canonical = schema.canonical_bindings(bindings)?;
        let key = (capture.identity, canonical.clone());
        let mut cache = self
            .specializations
            .lock()
            .map_err(|_| ReplayError::Backend("specialization cache lock poisoned".into()))?;
        let (concrete, cache_hit) = if let Some(concrete) = cache.get(&key) {
            (concrete.clone(), true)
        } else {
            let concrete = Arc::new(super::symbolic::specialize_capture(capture, &canonical)?);
            cache.insert(key, concrete.clone());
            (concrete, false)
        };
        Ok(CapturedSpecialization {
            trace: CapturedSpecializationTrace {
                source_identity: capture.identity,
                concrete_identity: concrete.identity,
                bindings: canonical,
                cache_hit,
            },
            capture: concrete,
        })
    }

    pub fn replay(
        &self,
        capture: &CapturedSchedule,
        provided: &BTreeMap<String, TensorData>,
        options: CapturedReplayOptions,
    ) -> Result<CapturedReplayResult, ReplayError> {
        crate::schedule::artifact::validate_for_replay(capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        validate_inputs(capture, provided)?;
        let plan = self.plan(capture, options.backend, None)?;
        execute_invocation(capture, provided, 0, &plan, options.backend, self, None)
    }

    /// Strict-native replay with conservative reverse-demand pruning. This is
    /// crate-private because the optimization is currently owned by the
    /// module-inference adapter; generic capture replay keeps its complete
    /// schedule trace and existing cache behavior.
    pub(crate) fn replay_pruned_native(
        &self,
        capture: &CapturedSchedule,
        provided: &BTreeMap<String, TensorData>,
        vectorized: bool,
    ) -> Result<CapturedReplayResult, ReplayError> {
        let prepared = self.prepare_pruned_native(capture, provided, vectorized)?;
        self.execute_prepared_pruned_native(capture, provided, &prepared)
    }

    /// Preflights and compiles the strict native path while retaining its
    /// existing liveness plan for one later detached execution. This is the
    /// narrow reportable phase boundary used by module inference; it neither
    /// executes an item nor exposes compiled resources.
    pub(crate) fn prepare_pruned_native(
        &self,
        capture: &CapturedSchedule,
        provided: &BTreeMap<String, TensorData>,
        vectorized: bool,
    ) -> Result<PreparedPrunedNativeReplay, ReplayError> {
        crate::schedule::artifact::validate_for_replay(capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        validate_inputs(capture, provided)?;
        let liveness = ReplayLivenessPlan::analyze(capture)?;
        let policy = CapturedBackendPolicy::NativeJit { vectorized };
        let plan = self.plan(capture, policy, Some(&liveness))?;
        Ok(PreparedPrunedNativeReplay {
            plan,
            vectorized,
            zero_pruned_item_count: liveness.pruned_item_count(),
            zero_materialized_item_count: liveness.materialized_zero_item_count(),
        })
    }

    /// Executes a prior strict-native preparation without performing another
    /// validation, liveness analysis, or cache lookup/compile pass.
    pub(crate) fn execute_prepared_pruned_native(
        &self,
        capture: &CapturedSchedule,
        provided: &BTreeMap<String, TensorData>,
        prepared: &PreparedPrunedNativeReplay,
    ) -> Result<CapturedReplayResult, ReplayError> {
        execute_invocation(
            capture,
            provided,
            0,
            &prepared.plan,
            CapturedBackendPolicy::NativeJit {
                vectorized: prepared.vectorized,
            },
            self,
            None,
        )
    }

    pub fn replay_symbolic(
        &self,
        capture: &CapturedSchedule,
        symbolic_bindings: &BTreeMap<String, i64>,
        provided: &BTreeMap<String, TensorData>,
        options: CapturedReplayOptions,
    ) -> Result<CapturedReplayResult, ReplayError> {
        // A concrete specialization is an executable cache artifact. Build
        // and validate it against caller bindings before publishing it, so a
        // failed external rebind cannot leave a newly reachable cache entry.
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let schema = capture
            .symbolic
            .as_ref()
            .ok_or_else(|| ReplayError::Symbolic("artifact is already concrete".into()))?;
        let canonical = schema.canonical_bindings(symbolic_bindings)?;
        let candidate = super::symbolic::specialize_capture(capture, &canonical)?;
        validate_inputs(&candidate, provided)?;
        let specialization =
            self.cache_preflighted_specialization(capture, canonical, candidate)?;
        let plan = self.plan(specialization.capture(), options.backend, None)?;
        execute_invocation(
            specialization.capture(),
            provided,
            0,
            &plan,
            options.backend,
            self,
            Some(specialization.trace.clone()),
        )
    }

    fn cache_preflighted_specialization(
        &self,
        capture: &CapturedSchedule,
        canonical: Vec<(u64, i64)>,
        candidate: CapturedSchedule,
    ) -> Result<CapturedSpecialization, ReplayError> {
        let key = (capture.identity, canonical.clone());
        let mut cache = self
            .specializations
            .lock()
            .map_err(|_| ReplayError::Backend("specialization cache lock poisoned".into()))?;
        let (concrete, cache_hit) = if let Some(concrete) = cache.get(&key) {
            (concrete.clone(), true)
        } else {
            let concrete = Arc::new(candidate);
            cache.insert(key, concrete.clone());
            (concrete, false)
        };
        Ok(CapturedSpecialization {
            trace: CapturedSpecializationTrace {
                source_identity: capture.identity,
                concrete_identity: concrete.identity,
                bindings: canonical,
                cache_hit,
            },
            capture: concrete,
        })
    }

    pub fn replay_batch(
        &self,
        capture: &CapturedSchedule,
        batch: &CapturedBatch,
        options: CapturedReplayOptions,
    ) -> Result<CapturedBatchResult, ReplayError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        if capture.items.iter().any(|item| !item.outputs.is_single()) {
            return Err(ReplayError::Unsupported(
                "multi-output captured schedules have no replay executor".into(),
            ));
        }
        if batch.artifact_identity != capture.identity {
            return Err(ReplayError::Corrupt(
                "batch artifact identity mismatch".into(),
            ));
        }
        for (index, invocation) in batch.invocations.iter().enumerate() {
            let concrete = if let Some(schema) = &capture.symbolic {
                let canonical = schema
                    .canonical_bindings(&invocation.symbolic_bindings)
                    .map_err(|error| ReplayError::Batch {
                        invocation: index,
                        reason: error.to_string(),
                    })?;
                super::symbolic::specialize_capture(capture, &canonical).map_err(|error| {
                    ReplayError::Batch {
                        invocation: index,
                        reason: error.to_string(),
                    }
                })?
            } else {
                if !invocation.symbolic_bindings.is_empty() {
                    return Err(ReplayError::Batch {
                        invocation: index,
                        reason: "concrete artifact received symbolic bindings".into(),
                    });
                }
                capture.clone()
            };
            validate_inputs(&concrete, &invocation.bindings).map_err(|error| {
                ReplayError::Batch {
                    invocation: index,
                    reason: error.to_string(),
                }
            })?;
        }
        // Every invocation is specialized and input-validated first. Every
        // concrete native plan is then compiled before the first execution.
        let mut specialized = Vec::with_capacity(batch.len());
        for (index, invocation) in batch.invocations.iter().enumerate() {
            let (concrete, trace) = if capture.is_symbolic() {
                let specialization = self
                    .specialize(capture, &invocation.symbolic_bindings)
                    .map_err(|error| ReplayError::Batch {
                        invocation: index,
                        reason: error.to_string(),
                    })?;
                (specialization.capture, Some(specialization.trace))
            } else {
                if !invocation.symbolic_bindings.is_empty() {
                    return Err(ReplayError::Batch {
                        invocation: index,
                        reason: "concrete artifact received symbolic bindings".into(),
                    });
                }
                (Arc::new(capture.clone()), None)
            };
            validate_inputs(&concrete, &invocation.bindings).map_err(|error| {
                ReplayError::Batch {
                    invocation: index,
                    reason: error.to_string(),
                }
            })?;
            specialized.push((concrete, trace));
        }
        for (index, (capture, _)) in specialized.iter().enumerate() {
            self.validate_backend_capability(capture, options.backend)
                .map_err(|error| ReplayError::Batch {
                    invocation: index,
                    reason: error.to_string(),
                })?;
        }
        let plans = specialized
            .iter()
            .enumerate()
            .map(|(index, (capture, _))| {
                self.plan(capture, options.backend, None)
                    .map_err(|error| ReplayError::Batch {
                        invocation: index,
                        reason: error.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut invocations = Vec::with_capacity(batch.len());
        for (index, ((invocation, (capture, specialization)), plan)) in batch
            .invocations
            .iter()
            .zip(specialized)
            .zip(plans)
            .enumerate()
        {
            invocations.push(execute_invocation(
                &capture,
                &invocation.bindings,
                index,
                &plan,
                options.backend,
                self,
                specialization,
            )?);
        }
        Ok(CapturedBatchResult { invocations })
    }

    fn plan(
        &self,
        capture: &CapturedSchedule,
        policy: CapturedBackendPolicy,
        liveness: Option<&ReplayLivenessPlan>,
    ) -> Result<Vec<PlannedItem>, ReplayError> {
        let (fallback, vectorized) = match policy {
            CapturedBackendPolicy::Interpreter => {
                return Ok(capture
                    .items
                    .iter()
                    .map(|_| PlannedItem::Interpreter)
                    .collect());
            }
            CapturedBackendPolicy::NativeJit { vectorized } => (false, vectorized),
            CapturedBackendPolicy::JitFallback { vectorized } => (true, vectorized),
        };
        let jit = self.jit(vectorized);
        let mut native = Vec::with_capacity(capture.items.len());
        for item in &capture.items {
            if liveness.is_some_and(|plan| plan.is_pruned(item.id).is_some())
                || liveness.is_some_and(|plan| plan.materializes_zero(item.id))
            {
                native.push(Ok(false));
                continue;
            }
            if item
                .primary_output()
                .shape
                .numel()
                .map_err(|e| ReplayError::Descriptor(e.to_string()))?
                == 0
                && item.boundary.is_none()
                && !item.is_effect()
            {
                native.push(
                    jit.prepare_zero_domain_schedule_item(item)
                        .map_err(|error| error.to_string()),
                );
                continue;
            }
            match jit.validate_schedule_item(item) {
                Ok(()) => native.push(Ok(false)),
                Err(error) if fallback => native.push(Err(error.to_string())),
                Err(error) => return Err(backend_error(error)),
            }
        }
        let mut out = Vec::with_capacity(capture.items.len());
        for (item, capability) in capture.items.iter().zip(native) {
            if let Some(desc) = liveness.and_then(|plan| plan.is_pruned(item.id)) {
                out.push(PlannedItem::PrunedZeroDomain {
                    descriptor: desc.clone(),
                });
                continue;
            }
            if liveness.is_some_and(|plan| plan.materializes_zero(item.id)) {
                out.push(PlannedItem::MaterializedZero);
                continue;
            }
            if item
                .primary_output()
                .shape
                .numel()
                .map_err(|e| ReplayError::Descriptor(e.to_string()))?
                == 0
                && item.boundary.is_none()
                && !item.is_effect()
            {
                out.push(PlannedItem::ZeroDomain {
                    cache_hit: capability.expect("zero-domain plan capability"),
                });
                continue;
            }
            if let Err(reason) = capability {
                out.push(PlannedItem::Fallback(reason));
                continue;
            }
            match jit.prepare_schedule_item(item) {
                Ok(prepared) => out.push(PlannedItem::Native(prepared)),
                Err(error) if fallback => out.push(PlannedItem::Fallback(error.to_string())),
                Err(error) => return Err(backend_error(error)),
            }
        }
        Ok(out)
    }

    fn validate_backend_capability(
        &self,
        capture: &CapturedSchedule,
        policy: CapturedBackendPolicy,
    ) -> Result<(), ReplayError> {
        let CapturedBackendPolicy::NativeJit { vectorized } = policy else {
            return Ok(());
        };
        for item in &capture.items {
            self.jit(vectorized)
                .validate_schedule_item(item)
                .map_err(backend_error)?;
        }
        Ok(())
    }

    fn jit(&self, vectorized: bool) -> &CpuJitBackend {
        if vectorized {
            &self.vectorized
        } else {
            &self.scalar
        }
    }
}

impl CapturedSchedule {
    /// Replays this concrete artifact with an explicit backend policy and a
    /// caller-owned executor whose native compile cache survives across calls.
    pub fn replay_with_options(
        &self,
        provided: &BTreeMap<String, TensorData>,
        executor: &CapturedReplayExecutor,
        options: CapturedReplayOptions,
    ) -> Result<CapturedReplayResult, ReplayError> {
        executor.replay(self, provided, options)
    }
}

enum PlannedItem {
    Interpreter,
    /// A private placeholder for dead pure work. A subsequent attempted read
    /// is a typed invariant failure, never a fabricated tensor.
    PrunedZeroDomain {
        descriptor: crate::BufferDesc,
    },
    /// A requested pure empty output remains public TensorData, but needs no
    /// native preparation or operand loads.
    MaterializedZero,
    ZeroDomain {
        cache_hit: bool,
    },
    Native(PreparedScheduleItem),
    Fallback(String),
}

/// Crate-private preparation ownership for one strict-native invocation.
/// It deliberately carries only already-validated logical plan data and the
/// existing prepared kernels; callers cannot observe or reuse backend handles.
pub(crate) struct PreparedPrunedNativeReplay {
    plan: Vec<PlannedItem>,
    vectorized: bool,
    zero_pruned_item_count: usize,
    zero_materialized_item_count: usize,
}

impl PreparedPrunedNativeReplay {
    pub(crate) fn zero_pruned_item_count(&self) -> usize {
        self.zero_pruned_item_count
    }

    pub(crate) fn zero_materialized_item_count(&self) -> usize {
        self.zero_materialized_item_count
    }
}

/// Fully compiled strict-native pure prefix, kept in the existing executor's
/// ownership domain until detached execution.
pub(crate) struct PlannedNativeItems {
    items: Vec<PreparedScheduleItem>,
    vectorized: bool,
}

impl CapturedReplayExecutor {
    pub(crate) fn plan_native_items(
        &self,
        capture: &CapturedSchedule,
        provided: &BTreeMap<String, TensorData>,
        vectorized: bool,
    ) -> Result<PlannedNativeItems, ReplayError> {
        reject_multi_output_items(capture)?;
        validate_inputs(capture, provided)?;
        if capture
            .items
            .iter()
            .any(|item| item.boundary.is_some() || item.is_effect())
        {
            return Err(ReplayError::Unsupported(
                "ordinary captured native replay cannot execute effect items".into(),
            ));
        }
        let planned = self.plan(
            capture,
            CapturedBackendPolicy::NativeJit { vectorized },
            None,
        )?;
        let items = planned
            .into_iter()
            .map(|item| match item {
                PlannedItem::Native(item) => Ok(item),
                _ => Err(ReplayError::Unsupported(
                    "strict native plan selected non-native item".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PlannedNativeItems { items, vectorized })
    }

    pub(crate) fn execute_planned_native_items(
        &self,
        capture: &CapturedSchedule,
        provided: &BTreeMap<String, TensorData>,
        plan: &PlannedNativeItems,
    ) -> Result<ReplayValues, ReplayError> {
        reject_multi_output_items(capture)?;
        let mut values = initial_values(capture, provided)?;
        for (item, prepared) in capture.items.iter().zip(&plan.items) {
            let (value, _) = self
                .jit(plan.vectorized)
                .execute_prepared_schedule_item(
                    item,
                    &values,
                    &capture.quantized_constants,
                    prepared,
                )
                .map_err(backend_error)?;
            values.insert_tensor(item.primary_output().id, value);
        }
        Ok(values)
    }
}

fn execute_invocation(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
    invocation: usize,
    plan: &[PlannedItem],
    policy: CapturedBackendPolicy,
    executor: &CapturedReplayExecutor,
    specialization: Option<CapturedSpecializationTrace>,
) -> Result<CapturedReplayResult, ReplayError> {
    let mut values = initial_values(capture, provided)?;
    let mut trace = CapturedReplayTrace::default();
    for (item, planned) in capture.items.iter().zip(plan) {
        let output_elements = item
            .primary_output()
            .shape
            .numel()
            .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
        let (value, backend, native_key, cache_hit, lanes, main, tail, reason) = match planned {
            PlannedItem::PrunedZeroDomain { descriptor } => {
                values.insert_pruned(item.primary_output().id, descriptor.clone(), item.id);
                continue;
            }
            PlannedItem::MaterializedZero => (
                TensorData::zeros_with_dtype(
                    item.primary_output().shape.clone(),
                    item.primary_output().dtype,
                )
                .map_err(|e| ReplayError::Descriptor(e.to_string()))?,
                ItemBackend::NativeJit,
                None,
                false,
                1,
                0,
                0,
                "reverse-liveness zero materialization".into(),
            ),
            PlannedItem::ZeroDomain { cache_hit } => (
                TensorData::zeros_with_dtype(
                    item.primary_output().shape.clone(),
                    item.primary_output().dtype,
                )
                .map_err(|e| ReplayError::Descriptor(e.to_string()))?,
                ItemBackend::NativeJit,
                None,
                *cache_hit,
                1,
                0,
                0,
                "native zero-domain skip".into(),
            ),
            PlannedItem::Interpreter => (
                interpret_item(capture, item, &values)?,
                ItemBackend::Interpreter,
                None,
                false,
                1,
                0,
                output_elements,
                "interpreter scalar semantics".into(),
            ),
            PlannedItem::Fallback(reason) => (
                interpret_item(capture, item, &values)?,
                ItemBackend::JitFallback,
                None,
                false,
                1,
                0,
                output_elements,
                reason.clone(),
            ),
            PlannedItem::Native(prepared) => {
                let vectorized = match policy {
                    CapturedBackendPolicy::NativeJit { vectorized }
                    | CapturedBackendPolicy::JitFallback { vectorized } => vectorized,
                    CapturedBackendPolicy::Interpreter => false,
                };
                let (value, execution) = executor
                    .jit(vectorized)
                    .execute_prepared_schedule_item(
                        item,
                        &values,
                        &capture.quantized_constants,
                        prepared,
                    )
                    .map_err(backend_error)?;
                (
                    value,
                    ItemBackend::NativeJit,
                    Some(execution.cache_key),
                    prepared.cache_hit,
                    execution.vector.lanes,
                    execution.vector_main,
                    execution.vector_tail,
                    execution.vector.reason,
                )
            }
        };
        values.insert_tensor(item.primary_output().id, value);
        trace.items.push(CapturedItemTrace {
            invocation,
            item: item.id,
            backend,
            schedule_cache_key: item.cache_key,
            native_cache_key: native_key,
            cache_hit,
            lanes,
            vector_main: main,
            vector_tail: tail,
            packed_weight_bytes: item
                .quantized_input_bindings
                .iter()
                .map(|binding| binding.desc.bytes)
                .sum(),
            reason,
        });
    }
    let outputs = values.requested(&capture.requested)?;
    Ok(CapturedReplayResult {
        outputs,
        trace,
        specialization,
    })
}

/// Crate-private interpreter seam for RGSM. The caller has already validated
/// its mixed topology and injects only detached persistent snapshots; this
/// avoids routing effectful items through the ordinary RGSA contract.
pub(crate) fn replay_interpreter_items(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
) -> Result<ReplayValues, ReplayError> {
    reject_multi_output_items(capture)?;
    validate_inputs(capture, provided)?;
    if capture
        .items
        .iter()
        .any(|item| item.boundary.is_some() || item.is_effect())
    {
        return Err(ReplayError::Unsupported(
            "ordinary captured interpreter cannot execute effect items".into(),
        ));
    }
    let mut values = initial_values(capture, provided)?;
    for item in &capture.items {
        let value = interpret_item(capture, item, &values).map_err(|error| match error {
            ReplayError::Execute(reason) => ReplayError::Execute(format!(
                "item {} output {} operation {:?}: {reason}",
                item.id,
                item.primary_output().id,
                item.kernel.operation()
            )),
            other => other,
        })?;
        values.insert_tensor(item.primary_output().id, value);
    }
    Ok(values)
}

/// Strict-native counterpart for mixed replay. It compiles every pure item
/// before executing one and returns only detached values; persistent state is
/// intentionally outside this module.
pub(crate) fn replay_native_items(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<ReplayValues, ReplayError> {
    let plan = executor.plan_native_items(capture, provided, vectorized)?;
    executor.execute_planned_native_items(capture, provided, &plan)
}

fn reject_multi_output_items(capture: &CapturedSchedule) -> Result<(), ReplayError> {
    if capture.items.iter().any(|item| !item.outputs.is_single()) {
        return Err(ReplayError::Unsupported(
            "multi-output captured replay is unavailable".into(),
        ));
    }
    Ok(())
}

fn validate_inputs(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
) -> Result<(), ReplayError> {
    let expected = capture
        .inputs
        .iter()
        .map(|x| x.name.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(name) = provided.keys().find(|x| !expected.contains(x.as_str())) {
        return Err(ReplayError::Extra(name.clone()));
    }
    for input in &capture.inputs {
        let value = provided
            .get(&input.name)
            .ok_or_else(|| ReplayError::Missing(input.name.clone()))?;
        if value.shape() != &input.desc.shape || value.dtype() != input.desc.dtype {
            return Err(ReplayError::Descriptor(input.name.clone()));
        }
    }
    for item in &capture.items {
        let crate::Operation::Movement(crate::MovementValue::QuantizedRowGather(plan)) =
            item.kernel.operation()
        else {
            continue;
        };
        let input = capture
            .inputs
            .iter()
            .find(|input| input.node == plan.indices)
            .ok_or_else(|| {
                ReplayError::Corrupt("quantized gather indices are not an input".into())
            })?;
        let indices = provided
            .get(&input.name)
            .ok_or_else(|| ReplayError::Missing(input.name.clone()))?;
        plan.preflight_indices(indices)
            .map_err(|error| ReplayError::Execute(error.to_string()))?;
    }
    Ok(())
}

fn initial_values(
    capture: &CapturedSchedule,
    provided: &BTreeMap<String, TensorData>,
) -> Result<ReplayValues, ReplayError> {
    let mut values = ReplayValues::default();
    for (id, value) in &capture.constants {
        values.insert_tensor(*id, value.clone());
    }
    for input in &capture.inputs {
        values.insert_tensor(
            input.desc.id,
            provided
                .get(&input.name)
                .cloned()
                .ok_or_else(|| ReplayError::Missing(input.name.clone()))?,
        );
    }
    for passthrough in &capture.requested_passthroughs {
        let source = values
            .tensor(
                passthrough.source.index() as u64,
                "requested passthrough source",
            )?
            .clone();
        let view =
            passthrough.desc.view.as_ref().ok_or_else(|| {
                ReplayError::Corrupt("requested passthrough view is absent".into())
            })?;
        let projected = source
            .affine_read(view)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        values.insert_tensor(passthrough.requested.index() as u64, projected);
    }
    Ok(values)
}

fn interpret_item(
    capture: &CapturedSchedule,
    item: &ScheduleItem,
    values: &ReplayValues,
) -> Result<TensorData, ReplayError> {
    if let crate::Operation::Movement(crate::MovementValue::QuantizedRowGather(plan)) =
        item.kernel.operation()
    {
        let indices = values.tensor(plan.indices.index() as u64, "quantized gather indices")?;
        let weight = capture
            .quantized_constants
            .get(&(plan.weight.index() as u64))
            .ok_or_else(|| ReplayError::Missing(plan.weight.index().to_string()))?;
        return plan
            .execute(indices, weight)
            .map_err(|error| ReplayError::Execute(error.to_string()));
    }
    if let crate::Operation::Matmul(crate::MatmulValue::Quantized(plan)) = item.kernel.operation() {
        let activation = values.tensor(
            plan.activation.index() as u64,
            "quantized matmul activation",
        )?;
        let weight = capture
            .quantized_constants
            .get(&(plan.weight.index() as u64))
            .ok_or_else(|| ReplayError::Missing(plan.weight.index().to_string()))?;
        return plan
            .execute(activation, weight)
            .map_err(|error| ReplayError::Execute(error.to_string()));
    }
    if let crate::Operation::Movement(crate::MovementValue::Plan(plan)) = item.kernel.operation() {
        let operands = plan
            .input_operands()
            .into_iter()
            .map(|operand| {
                values
                    .tensor(operand.node.index() as u64, "movement operand")
                    .cloned()
            })
            .collect::<Result<Vec<_>, _>>()?;
        return plan
            .execute(&operands)
            .map_err(|error| ReplayError::Execute(error.to_string()));
    }
    if let crate::Operation::PrefixScan(plan) = item.kernel.operation() {
        let input = values.tensor(plan.input.index() as u64, "prefix scan input")?;
        if input.shape() != &plan.input_shape {
            return Err(ReplayError::Descriptor(format!(
                "prefix scan input {} has shape {}, expected {}",
                plan.input,
                input.shape(),
                plan.input_shape
            )));
        }
        return crate::backend::execute_prefix_scan(
            input,
            plan.axis,
            plan.kind,
            plan.output,
            plan.dtype,
        )
        .map_err(|error| ReplayError::Execute(error.to_string()));
    }
    let mut bindings = KernelBindings::default();
    for binding in item.ordered_inputs() {
        let value = values.tensor(binding.desc.id, "kernel input")?.clone();
        let value = super::direct_matmul_input(item, binding, &value)
            .map_err(ReplayError::Execute)?
            .into_owned();
        let role = if capture.constants.contains_key(&binding.desc.id) {
            BufferRole::Constant
        } else {
            BufferRole::Input
        };
        let desc = KernelBufferDesc::concrete(
            binding.desc.id,
            role,
            value.shape().clone(),
            binding.desc.dtype,
            false,
        )
        .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
        bindings
            .insert(&desc, value)
            .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
    }
    crate::kernel::execute_lowered_elementwise(&item.kernel, &bindings)
        .map_err(|e| ReplayError::Execute(e.to_string()))
}

fn backend_error(error: JitBackendError) -> ReplayError {
    match error {
        JitBackendError::Unsupported(reason) => ReplayError::Unsupported(reason),
        other => ReplayError::Backend(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape, Storage};
    use std::collections::HashMap;

    fn captured(graph: &Graph, requested: &[crate::NodeId]) -> CapturedSchedule {
        let schedule = crate::schedule_many(graph, requested).unwrap();
        let capture = CapturedSchedule::capture(graph, &schedule, requested).unwrap();
        CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap()
    }

    fn interpreter_result(
        graph: &Graph,
        output: crate::NodeId,
        bindings: &BTreeMap<String, TensorData>,
    ) -> TensorData {
        CapturedReplayExecutor::default()
            .replay(
                &captured(graph, &[output]),
                bindings,
                CapturedReplayOptions::default(),
            )
            .unwrap()
            .outputs
            .remove(0)
    }

    #[test]
    fn captured_single_reduction_epilogue_replays_as_one_native_item() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let bias = graph.input("bias", [2]);
        let reduced = graph.sum(input, 1).unwrap();
        let shifted = graph.add(reduced, bias).unwrap();
        let output = graph.relu(shifted).unwrap();
        let capture = captured(&graph, &[output]);
        assert_eq!(capture.items.len(), 1);
        assert!(
            capture.items[0]
                .ordered_inputs()
                .iter()
                .all(|binding| binding.desc.id != reduced.index() as u64)
        );

        let input_value = TensorData::new([2, 3], vec![1.0, 2.0, 3.0, -4.0, 1.0, 2.0]).unwrap();
        let bias_value = TensorData::new([2], vec![-7.0, 2.0]).unwrap();
        let bindings = BTreeMap::from([
            ("input".into(), input_value.clone()),
            ("bias".into(), bias_value.clone()),
        ]);
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("input".into(), input_value), ("bias".into(), bias_value)]),
            )
            .unwrap();
        let executor = CapturedReplayExecutor::default();
        let interpreted = executor
            .replay(&capture, &bindings, CapturedReplayOptions::default())
            .unwrap();
        assert_eq!(interpreted.outputs[0].storage(), expected.storage());
        let native = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(native.outputs[0].storage(), expected.storage());
        assert_eq!(native.trace.items[0].backend, ItemBackend::NativeJit);
    }

    fn assert_computed_affine_replay(
        graph: &Graph,
        output: crate::NodeId,
        bindings: BTreeMap<String, TensorData>,
    ) {
        let scheduled = crate::schedule(graph, output).unwrap();
        let capture = CapturedSchedule::capture(graph, &scheduled, &[output]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        let capture = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(capture.to_bytes().unwrap(), bytes);
        let oracle = CpuBackend
            .execute(
                graph,
                output,
                &bindings.clone().into_iter().collect::<HashMap<_, _>>(),
            )
            .unwrap();
        let executor = CapturedReplayExecutor::default();
        let interpreted = executor
            .replay(&capture, &bindings, CapturedReplayOptions::default())
            .unwrap();
        let native = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        let f32_bits = |value: &TensorData| {
            let Storage::F32(values) = value.storage() else {
                panic!("computed affine replay fixture must remain F32")
            };
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let expected = f32_bits(&oracle);
        assert_eq!(f32_bits(&interpreted.outputs[0]), expected);
        assert_eq!(f32_bits(&native.outputs[0]), expected);
        assert!(
            native
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
    }

    #[test]
    fn captured_computed_affine_reads_match_interpreter_and_native() {
        let mut broadcast = Graph::new();
        let input = broadcast.input_dtype("input", [2, 1], DType::F32);
        let producer = broadcast.square(input).unwrap();
        let output = broadcast.expand(producer, [2, 3]).unwrap();
        assert_computed_affine_replay(
            &broadcast,
            output,
            BTreeMap::from([(
                "input".into(),
                TensorData::new([2, 1], vec![2.0, -3.0]).unwrap(),
            )]),
        );

        let mut reverse = Graph::new();
        let condition = reverse.input_dtype("condition", [4], DType::Bool);
        let input = reverse.input_dtype("input", [4], DType::F32);
        let alternative = reverse.input_dtype("alternative", [4], DType::F32);
        let producer = reverse.select(condition, input, alternative).unwrap();
        let output = reverse
            .stride(
                producer,
                [crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            )
            .unwrap();
        assert_computed_affine_replay(
            &reverse,
            output,
            BTreeMap::from([
                (
                    "condition".into(),
                    TensorData::from_storage([4], Storage::Bool(vec![true, true, true, true]))
                        .unwrap(),
                ),
                (
                    "input".into(),
                    TensorData::from_storage(
                        [4],
                        Storage::F32(vec![
                            f32::from_bits(0x8000_0000),
                            f32::from_bits(0x7fc0_0001),
                            f32::from_bits(0x7f80_0000),
                            f32::from_bits(0x3f80_0000),
                        ]),
                    )
                    .unwrap(),
                ),
                (
                    "alternative".into(),
                    TensorData::from_storage([4], Storage::F32(vec![0.0; 4])).unwrap(),
                ),
            ]),
        );

        let mut diagonal = Graph::new();
        let input = diagonal.input_dtype("input", [3, 3], DType::F32);
        let output = diagonal.diagonal_default(input).unwrap();
        assert_computed_affine_replay(
            &diagonal,
            output,
            BTreeMap::from([(
                "input".into(),
                TensorData::new([3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap(),
            )]),
        );
    }

    #[test]
    fn captured_raw_matmul_resolves_a_transposed_consumer_view_once() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 3], DType::F32);
        let rhs_base = graph.input_dtype("rhs", [2, 3], DType::F32);
        let rhs = graph.permute(rhs_base, [1, 0]).unwrap();
        let output = graph.matmul(lhs, rhs).unwrap();
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let mut item = capture
            .items
            .iter()
            .find(|item| matches!(item.kernel.operation(), crate::Operation::Matmul(_)))
            .unwrap()
            .clone();
        let plan = match item.kernel.operation() {
            crate::Operation::Matmul(crate::MatmulValue::Serial(plan)) => plan.as_ref(),
            crate::Operation::Matmul(crate::MatmulValue::Tiled(payload)) => &payload.matmul,
            crate::Operation::Matmul(crate::MatmulValue::TensorCore(payload)) => &payload.matmul,
            _ => unreachable!(),
        };
        assert_eq!(plan.rhs, rhs);
        let view = crate::rangeify::static_view(&graph, rhs).unwrap().view;
        assert_eq!(view.source_shape, Shape::from([2, 3]));
        assert_eq!(view.logical_shape, Shape::from([3, 2]));
        for desc in &mut item.inputs {
            if desc.id == rhs.index() as u64 {
                desc.shape = view.source_shape.clone();
                desc.view = Some(view.clone());
            }
        }
        for binding in &mut item.input_bindings {
            if binding.desc.id == rhs.index() as u64 {
                binding.desc.shape = view.source_shape.clone();
                binding.desc.view = Some(view.clone());
            }
        }
        item.validate_input_bindings().unwrap();

        let lhs_value = TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs_value = TensorData::new([2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let generic_view_item = capture
            .items
            .iter()
            .find(|candidate| {
                !matches!(candidate.kernel.operation(), crate::Operation::Matmul(_))
                    && candidate
                        .input_bindings
                        .iter()
                        .any(|binding| binding.desc.view.is_some())
            })
            .unwrap();
        let generic_view_binding = generic_view_item
            .input_bindings
            .iter()
            .find(|binding| binding.desc.view.is_some())
            .unwrap();
        assert!(matches!(
            crate::engine::direct_matmul_input(
                generic_view_item,
                generic_view_binding,
                &rhs_value,
            )
            .unwrap(),
            std::borrow::Cow::Borrowed(_)
        ));
        let mut values = ReplayValues::default();
        values.insert_tensor(lhs.index() as u64, lhs_value.clone());
        values.insert_tensor(rhs.index() as u64, rhs_value.clone());
        let actual = interpret_item(&capture, &item, &values).unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("lhs".into(), lhs_value), ("rhs".into(), rhs_value)]),
            )
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn captured_contiguous_roundtrips_interpreter_and_dependent_native_execution() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let permuted = graph.permute(input, [1, 0]).unwrap();
        let output = graph.contiguous(permuted).unwrap();
        let capture = captured(&graph, &[output]);
        assert_eq!(capture.items.len(), 1);
        assert!(matches!(
            capture.items[0].kernel.operation(),
            crate::Operation::Movement(crate::MovementValue::Plan(plan))
                if matches!(
                    &plan.kind,
                    crate::MovementKernelKind::AffineCopy { input: operand, view }
                        if operand.node == input
                            && view.logical_shape == Shape::new([2, 2])
                            && view.strides == [1, 2]
                )
        ));
        assert!(capture.items[0].dependencies.is_empty());

        let value = TensorData::from_storage(
            [2, 2],
            Storage::F32(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_0123),
                f32::INFINITY,
                f32::NEG_INFINITY,
            ]),
        )
        .unwrap();
        let provided = BTreeMap::from([("input".into(), value)]);
        let executor = CapturedReplayExecutor::default();
        let interpreted = executor
            .replay(
                &capture,
                &provided,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::Interpreter,
                },
            )
            .unwrap();
        let native = executor
            .replay(
                &capture,
                &provided,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        let raw = |data: &TensorData| match data.storage() {
            Storage::F32(values) => values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            _ => unreachable!("F32 contiguous fixture"),
        };
        assert_eq!(
            raw(&interpreted.outputs[0]),
            [0x8000_0000, 0x7f80_0000, 0x7fc0_0123, 0xff80_0000]
        );
        assert_eq!(raw(&native.outputs[0]), raw(&interpreted.outputs[0]));
        assert!(
            native
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );

        let mut float8 = Graph::new();
        let input = float8.input_dtype("input", [2, 2], DType::F8E4M3);
        let viewed = float8.permute(input, [1, 0]).unwrap();
        let output = float8.contiguous(viewed).unwrap();
        let capture = captured(&float8, &[output]);
        assert_eq!(capture.items.len(), 1);
        let value = TensorData::from_storage(
            [2, 2],
            Storage::Float8(crate::Float8Storage::from_raw(
                crate::Float8Format::E4M3,
                vec![0x00, 0x80, 0x7f, 0xff],
            )),
        )
        .unwrap();
        let replay = executor
            .replay(
                &capture,
                &BTreeMap::from([("input".into(), value)]),
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        let Storage::Float8(value) = replay.outputs[0].storage() else {
            panic!("Float8 contiguous affine output")
        };
        assert_eq!(value.as_raw(), [0x00, 0x7f, 0x80, 0xff]);

        let mut computed = Graph::new();
        let input = computed.input_dtype("input", [2, 1], DType::I32);
        let producer = computed.square(input).unwrap();
        let expanded = computed.expand(producer, [2, 3]).unwrap();
        let output = computed.contiguous(expanded).unwrap();
        let capture = captured(&computed, &[output]);
        assert_eq!(capture.items.len(), 1);
        let fused = &capture.items[0];
        assert_eq!(fused.primary_output().id, output.index() as u64);
        assert!(matches!(fused.kernel.operation(), crate::Operation::Sink));
        assert_eq!(
            fused
                .ordered_inputs()
                .iter()
                .map(|binding| binding.input_node)
                .collect::<Vec<_>>(),
            vec![input]
        );
        assert!(fused.dependencies.is_empty());
        let bindings = BTreeMap::from([(
            "input".into(),
            TensorData::from_storage([2, 1], Storage::I32(vec![2, -3])).unwrap(),
        )]);
        let interpreted = executor
            .replay(&capture, &bindings, CapturedReplayOptions::default())
            .unwrap();
        let native = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            interpreted.outputs[0].storage(),
            &Storage::I32(vec![4, 4, 4, 9, 9, 9])
        );
        assert_eq!(
            native.outputs[0].storage(),
            interpreted.outputs[0].storage()
        );
        assert_eq!(native.trace.items[0].backend, ItemBackend::NativeJit);

        // Requesting the computed producer makes ownership observable and
        // preserves the explicit AffineCopy capture/dependency contract.
        let fallback = captured(&computed, &[producer, output]);
        assert_eq!(fallback.items.len(), 2);
        let copy = fallback
            .items
            .iter()
            .find(|item| item.primary_output().id == output.index() as u64)
            .unwrap();
        assert!(matches!(
            copy.kernel.operation(),
            crate::Operation::Movement(crate::MovementValue::Plan(plan))
                if matches!(
                    &plan.kind,
                    crate::MovementKernelKind::AffineCopy { input: operand, view }
                        if operand.node == producer && view.strides == [1, 0]
                )
        ));
        assert!(copy.dependencies.iter().any(|dependency| {
            fallback.items[*dependency as usize].primary_output().id == producer.index() as u64
        }));
        let mut missing_edge = fallback.clone();
        let copy_position = missing_edge
            .items
            .iter()
            .position(|item| item.primary_output().id == output.index() as u64)
            .unwrap();
        let dependency = missing_edge.items[copy_position].dependencies[0] as usize;
        missing_edge.items[copy_position].dependencies.clear();
        missing_edge.items[dependency].consumers.clear();
        assert!(crate::schedule::artifact::validate_capture(&missing_edge).is_err());
        let fallback_replay = executor
            .replay(
                &fallback,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            fallback_replay.outputs[1].storage(),
            interpreted.outputs[0].storage()
        );
    }

    #[test]
    fn captured_prefix_scan_executes_its_materialized_computed_input_graph_free() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::I16);
        let bias =
            graph.constant(TensorData::from_scalars([], DType::I16, [Scalar::I(1)]).unwrap());
        let shifted = graph.binary(crate::BinaryOp::Add, input, bias).unwrap();
        let output = graph.cumsum(shifted, 0).unwrap();
        let capture = captured(&graph, &[output]);
        let scan_item = capture
            .items
            .iter()
            .find(|item| matches!(item.kernel.operation(), crate::Operation::PrefixScan(_)))
            .unwrap();
        let scan_input = match scan_item.kernel.operation() {
            crate::Operation::PrefixScan(plan) => plan.input,
            _ => unreachable!(),
        };
        assert!(scan_item.dependencies.iter().any(|dependency| {
            capture.items[*dependency as usize].primary_output().id == scan_input.index() as u64
        }));

        let bindings = BTreeMap::from([(
            "input".into(),
            TensorData::from_scalars([2], DType::I16, [Scalar::I(1), Scalar::I(2)]).unwrap(),
        )]);
        let actual = CapturedReplayExecutor::default()
            .replay(&capture, &bindings, CapturedReplayOptions::default())
            .unwrap()
            .outputs
            .remove(0);
        assert_eq!(actual.shape(), &Shape::from([2]));
        assert_eq!(actual.dtype(), DType::I32);
        assert_eq!(actual.to_vec_f64(), vec![2.0, 5.0]);
        let native = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(native.outputs[0].storage(), &Storage::I32(vec![2, 5]));
        assert_eq!(
            native
                .trace
                .items
                .iter()
                .find(|trace| trace.item == scan_item.id)
                .unwrap()
                .backend,
            ItemBackend::NativeJit
        );

        let mut malformed = capture.clone();
        let malformed_item = malformed
            .items
            .iter_mut()
            .find(|item| matches!(item.kernel.operation(), crate::Operation::PrefixScan(_)))
            .unwrap();
        let mutate_desc = |desc: &mut crate::BufferDesc| {
            if desc.id == scan_input.index() as u64 {
                desc.dtype = DType::F32;
                desc.bytes = desc.shape.numel().unwrap() * DType::F32.itemsize();
                desc.alignment = DType::F32.itemsize();
            }
        };
        for desc in &mut malformed_item.inputs {
            mutate_desc(desc);
        }
        for binding in &mut malformed_item.input_bindings {
            mutate_desc(&mut binding.desc);
        }
        assert!(matches!(
            malformed_item.validate_input_bindings(),
            Err(crate::ScheduleError::Binding(reason))
                if reason == "prefix scan descriptor mismatch"
        ));
        assert!(matches!(malformed.to_bytes(), Err(ReplayError::Corrupt(_))));

        let mut extrema_graph = Graph::new();
        let extrema_input = extrema_graph.input_dtype("input", [2], DType::F32);
        let (_, extrema_indices) = extrema_graph.cummax(extrema_input, 0).unwrap();
        let extrema_capture = captured(&extrema_graph, &[extrema_indices]);
        let extrema_item = extrema_capture
            .items
            .iter()
            .find(|item| matches!(item.kernel.operation(), crate::Operation::PrefixScan(_)))
            .unwrap();
        assert_eq!(extrema_item.ordered_inputs()[0].desc.dtype, DType::F32);
        assert_eq!(extrema_item.primary_output().dtype, DType::I32);
        extrema_item.validate_input_bindings().unwrap();
        let extrema_native = CapturedReplayExecutor::default()
            .replay(
                &extrema_capture,
                &BTreeMap::from([(
                    "input".into(),
                    TensorData::from_storage([2], Storage::F32(vec![-0.0, 0.0])).unwrap(),
                )]),
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            extrema_native.outputs[0].storage(),
            &Storage::I32(vec![0, 0])
        );
        assert_eq!(
            extrema_native.trace.items[0].backend,
            ItemBackend::NativeJit
        );

        let mut boolean_graph = Graph::new();
        let boolean_input = boolean_graph.input_dtype("input", [5], DType::Bool);
        let (maximum, maximum_indices) = boolean_graph.cummax(boolean_input, 0).unwrap();
        let (minimum, minimum_indices) = boolean_graph.cummin(boolean_input, 0).unwrap();
        let boolean_capture = captured(
            &boolean_graph,
            &[maximum, maximum_indices, minimum, minimum_indices],
        );
        let boolean_bindings = BTreeMap::from([(
            "input".into(),
            TensorData::from_storage([5], Storage::Bool(vec![false, false, true, true, false]))
                .unwrap(),
        )]);
        let expected = [
            Storage::Bool(vec![false, false, true, true, true]),
            Storage::I32(vec![0, 0, 2, 2, 2]),
            Storage::Bool(vec![false, false, false, false, false]),
            Storage::I32(vec![0, 0, 0, 0, 0]),
        ];
        for backend in [
            CapturedBackendPolicy::Interpreter,
            CapturedBackendPolicy::NativeJit { vectorized: false },
        ] {
            let replay = CapturedReplayExecutor::default()
                .replay(
                    &boolean_capture,
                    &boolean_bindings,
                    CapturedReplayOptions { backend },
                )
                .unwrap();
            let actual = replay
                .outputs
                .iter()
                .map(|output| output.storage().clone())
                .collect::<Vec<_>>();
            assert_eq!(actual.as_slice(), expected.as_slice());
            if matches!(backend, CapturedBackendPolicy::NativeJit { .. }) {
                assert!(
                    replay
                        .trace
                        .items
                        .iter()
                        .all(|trace| trace.backend == ItemBackend::NativeJit)
                );
            }
        }

        let mut scalar_graph = Graph::new();
        let scalar_input = scalar_graph.input_dtype("input", [], DType::F32);
        let scalar_sum = scalar_graph.cumsum(scalar_input, 0).unwrap();
        let scalar_product = scalar_graph.cumprod(scalar_input, 0).unwrap();
        let (scalar_maximum, scalar_maximum_indices) =
            scalar_graph.cummax(scalar_input, 0).unwrap();
        let (scalar_minimum, scalar_minimum_indices) =
            scalar_graph.cummin(scalar_input, 0).unwrap();
        let scalar_capture = captured(
            &scalar_graph,
            &[
                scalar_sum,
                scalar_product,
                scalar_maximum,
                scalar_minimum,
                scalar_maximum_indices,
                scalar_minimum_indices,
            ],
        );
        let scalar_bits = 0x7fc0_1234;
        let scalar_replay = CapturedReplayExecutor::default()
            .replay(
                &scalar_capture,
                &BTreeMap::from([(
                    "input".into(),
                    TensorData::from_storage([], Storage::F32(vec![f32::from_bits(scalar_bits)]))
                        .unwrap(),
                )]),
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        for output in &scalar_replay.outputs[..4] {
            let Storage::F32(values) = output.storage() else {
                panic!("scalar F32 scan value storage")
            };
            assert_eq!(values[0].to_bits(), scalar_bits);
        }
        assert_eq!(scalar_replay.outputs[4].storage(), &Storage::I32(vec![0]));
        assert_eq!(scalar_replay.outputs[5].storage(), &Storage::I32(vec![0]));
        assert!(
            scalar_replay
                .trace
                .items
                .iter()
                .all(|trace| trace.backend == ItemBackend::NativeJit)
        );

        let mut width_graph = Graph::new();
        let width_input = width_graph.input_dtype("input", [3], DType::F32);
        let width_output = width_graph.cumsum(width_input, 0).unwrap();
        let width_capture = captured(&width_graph, &[width_output]);
        let width_native = CapturedReplayExecutor::default()
            .replay(
                &width_capture,
                &BTreeMap::from([(
                    "input".into(),
                    TensorData::from_storage(
                        [3],
                        Storage::F32(vec![16_777_216.0, 1.0, -16_777_216.0]),
                    )
                    .unwrap(),
                )]),
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            width_native.outputs[0].storage(),
            &Storage::F32(vec![16_777_216.0, 16_777_216.0, 0.0])
        );
    }

    #[test]
    fn captured_threefry_random_is_graph_free_and_native_f32_f64_matches_oracle() {
        let _stream_guard = Graph::lock_implicit_random_tests();
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let output = graph.rand([5], dtype, 0x1234_5678).unwrap();
            let capture = captured(&graph, &[output]);
            assert!(capture.inputs.is_empty());
            let oracle = CpuBackend.execute(&graph, output, &HashMap::new()).unwrap();
            let executor = CapturedReplayExecutor::default();
            let first = executor
                .replay(&capture, &BTreeMap::new(), CapturedReplayOptions::default())
                .unwrap();
            let native = executor
                .replay(
                    &capture,
                    &BTreeMap::new(),
                    CapturedReplayOptions {
                        backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                    },
                )
                .unwrap();
            assert_eq!(first.outputs[0], oracle);
            assert_eq!(native.outputs[0], oracle);
            assert_eq!(native.trace.items[0].backend, ItemBackend::NativeJit);
            // Replay reads only the captured reservation, not the mutable stream registry.
            Graph::manual_seed(7);
            assert_eq!(
                executor
                    .replay(&capture, &BTreeMap::new(), CapturedReplayOptions::default())
                    .unwrap()
                    .outputs[0],
                oracle
            );
            let bytes = capture.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }
    }

    #[test]
    fn captured_threefry_empty_and_narrow_replay_match_oracle_without_stream_state() {
        for (shape, dtype) in [
            ([0], DType::F16),
            ([3], DType::BF16),
            ([5], DType::F32),
            ([3], DType::F64),
        ] {
            let mut graph = Graph::new();
            let output = graph.rand(shape, dtype, 99).unwrap();
            let capture = captured(&graph, &[output]);
            let oracle = CpuBackend.execute(&graph, output, &HashMap::new()).unwrap();
            assert_eq!(
                CapturedReplayExecutor::default()
                    .replay(&capture, &BTreeMap::new(), CapturedReplayOptions::default())
                    .unwrap()
                    .outputs[0],
                oracle
            );
        }
    }

    #[test]
    fn captured_threefry_native_full_distribution_surface_matches_cpu() {
        let _stream_guard = Graph::lock_implicit_random_tests();
        enum Distribution {
            Uniform(f64, f64),
            Normal(f64, f64),
            RandInt(i64, i64),
        }
        let cases = [
            (
                "f16 uniform odd",
                [5],
                DType::F16,
                Distribution::Uniform(-1.5, 2.25),
            ),
            (
                "bf16 uniform",
                [4],
                DType::BF16,
                Distribution::Uniform(0.25, 1.5),
            ),
            (
                "f32 normal odd",
                [3],
                DType::F32,
                Distribution::Normal(-0.5, 1.25),
            ),
            (
                "f64 normal",
                [4],
                DType::F64,
                Distribution::Normal(2.0, 0.5),
            ),
            (
                "f16 normal",
                [3],
                DType::F16,
                Distribution::Normal(0.0, 1.0),
            ),
            (
                "bf16 normal",
                [3],
                DType::BF16,
                Distribution::Normal(0.0, 1.0),
            ),
            (
                "i8 randint negative",
                [5],
                DType::I8,
                Distribution::RandInt(-3, 5),
            ),
            ("u8 randint", [3], DType::U8, Distribution::RandInt(1, 10)),
            (
                "i16 randint",
                [3],
                DType::I16,
                Distribution::RandInt(-70, 31),
            ),
            (
                "u16 randint",
                [3],
                DType::U16,
                Distribution::RandInt(31, 700),
            ),
            (
                "i32 randint",
                [3],
                DType::I32,
                Distribution::RandInt(-7000, 9000),
            ),
            ("u32 randint", [4], DType::U32, Distribution::RandInt(3, 19)),
            (
                "i64 randint",
                [3],
                DType::I64,
                Distribution::RandInt(-9, -1),
            ),
            ("u64 randint", [3], DType::U64, Distribution::RandInt(0, 99)),
            ("zero randint", [0], DType::U64, Distribution::RandInt(0, 7)),
        ];
        let executor = CapturedReplayExecutor::default();
        for (name, shape, dtype, distribution) in cases {
            let mut graph = Graph::new();
            let output = match distribution {
                Distribution::Uniform(low, high) => graph.uniform(shape, low, high, dtype, 91),
                Distribution::Normal(mean, std) => graph.normal(shape, mean, std, dtype, 91),
                Distribution::RandInt(low, high) => graph.randint(shape, low, high, dtype, 91),
            }
            .unwrap();
            let capture = captured(&graph, &[output]);
            let oracle = CpuBackend.execute(&graph, output, &HashMap::new()).unwrap();
            let first = executor
                .replay(
                    &capture,
                    &BTreeMap::new(),
                    CapturedReplayOptions {
                        backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                    },
                )
                .unwrap();
            let second = executor
                .replay(
                    &capture,
                    &BTreeMap::new(),
                    CapturedReplayOptions {
                        backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                    },
                )
                .unwrap();
            assert_eq!(
                first.outputs[0].to_le_bytes().unwrap(),
                oracle.to_le_bytes().unwrap(),
                "{name}"
            );
            assert_eq!(second.outputs[0], first.outputs[0], "{name} replay");
            assert_eq!(
                first.trace.items[0].backend,
                ItemBackend::NativeJit,
                "{name}"
            );
            assert_eq!(
                first.trace.items[0].native_cache_key, second.trace.items[0].native_cache_key,
                "{name} key"
            );
            Graph::manual_seed(7);
            assert_eq!(
                executor
                    .replay(
                        &capture,
                        &BTreeMap::new(),
                        CapturedReplayOptions {
                            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                        },
                    )
                    .unwrap()
                    .outputs[0],
                oracle,
                "{name} captured state"
            );
        }
    }

    #[test]
    fn artifact_interpreter_executes_all_movement_kinds_against_cpu_oracle() {
        let mut concat_graph = Graph::new();
        let lhs = concat_graph.input_dtype("lhs", [2, 0], DType::I32);
        let rhs = concat_graph.input_dtype("rhs", [2, 3], DType::I32);
        let concat = concat_graph.concat([lhs, rhs], 1).unwrap();
        let concat_bindings = BTreeMap::from([
            (
                "lhs".into(),
                TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_storage([2, 3], Storage::I32(vec![0; 6])).unwrap(),
            ),
        ]);
        let concat_oracle = concat_bindings
            .clone()
            .into_iter()
            .collect::<HashMap<_, _>>();
        assert_eq!(
            interpreter_result(&concat_graph, concat, &concat_bindings),
            CpuBackend
                .execute(&concat_graph, concat, &concat_oracle)
                .unwrap()
        );

        let mut mixed_graph = Graph::new();
        let lhs = mixed_graph.input_dtype("lhs", [1, 2], DType::I8);
        let rhs = mixed_graph.input_dtype("rhs", [1, 1], DType::U8);
        let mixed = mixed_graph.concat([lhs, rhs], 1).unwrap();
        let mixed_bindings = BTreeMap::from([
            (
                "lhs".into(),
                TensorData::from_storage([1, 2], Storage::I8(vec![-2, 3])).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_storage([1, 1], Storage::U8(vec![250])).unwrap(),
            ),
        ]);
        let mixed_capture = captured(&mixed_graph, &[mixed]);
        let mixed_oracle = mixed_bindings
            .clone()
            .into_iter()
            .collect::<HashMap<_, _>>();
        assert_eq!(
            CapturedReplayExecutor::default()
                .replay(
                    &mixed_capture,
                    &mixed_bindings,
                    CapturedReplayOptions::default()
                )
                .unwrap()
                .outputs[0],
            CpuBackend
                .execute(&mixed_graph, mixed, &mixed_oracle)
                .unwrap()
        );
        assert!(matches!(
            CapturedReplayExecutor::default().replay(
                &mixed_capture,
                &mixed_bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Unsupported(reason)) if reason.contains("homogeneous")
        ));

        let mut gather_graph = Graph::new();
        let input = gather_graph.input_dtype("input", [2, 3], DType::F16);
        let index = gather_graph.input_dtype("index", [2, 2], DType::U16);
        let gather = gather_graph.gather(input, index, 1).unwrap();
        let gather_bindings = BTreeMap::from([
            (
                "input".into(),
                TensorData::from_storage(
                    [2, 3],
                    Storage::F16(vec![0x8000, 0x7e01, 0x3c00, 0x4000, 0x4200, 0x4400]),
                )
                .unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_storage([2, 2], Storage::U16(vec![2, 0, 1, 1])).unwrap(),
            ),
        ]);
        let gather_oracle = gather_bindings
            .clone()
            .into_iter()
            .collect::<HashMap<_, _>>();
        assert_eq!(
            interpreter_result(&gather_graph, gather, &gather_bindings),
            CpuBackend
                .execute(&gather_graph, gather, &gather_oracle)
                .unwrap()
        );

        for add in [false, true] {
            let mut scatter_graph = Graph::new();
            let base = scatter_graph.input_dtype("base", [1, 3], DType::F64);
            let index = scatter_graph.input_dtype("index", [1, 3], DType::I8);
            let updates = scatter_graph.input_dtype("updates", [1, 3], DType::F64);
            let scatter = if add {
                scatter_graph.scatter_add(base, index, updates, 1).unwrap()
            } else {
                scatter_graph.scatter(base, index, updates, 1).unwrap()
            };
            let scatter_bindings = BTreeMap::from([
                (
                    "base".into(),
                    TensorData::from_storage([1, 3], Storage::F64(vec![1.0, 2.0, 3.0])).unwrap(),
                ),
                (
                    "index".into(),
                    TensorData::from_storage([1, 3], Storage::I8(vec![1, 1, 1])).unwrap(),
                ),
                (
                    "updates".into(),
                    TensorData::from_storage([1, 3], Storage::F64(vec![0.25, 0.5, 4.0])).unwrap(),
                ),
            ]);
            let scatter_oracle = scatter_bindings
                .clone()
                .into_iter()
                .collect::<HashMap<_, _>>();
            assert_eq!(
                interpreter_result(&scatter_graph, scatter, &scatter_bindings),
                CpuBackend
                    .execute(&scatter_graph, scatter, &scatter_oracle)
                    .unwrap(),
                "add={add}"
            );
        }
    }

    #[test]
    fn artifact_interpreter_preflights_every_movement_index() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 3], DType::I32);
        let index = graph.input_dtype("index", [1, 3], DType::I64);
        let output = graph.gather(input, index, 1).unwrap();
        let capture = captured(&graph, &[output]);
        let bindings = BTreeMap::from([
            (
                "input".into(),
                TensorData::from_storage([1, 3], Storage::I32(vec![10, 20, 30])).unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_storage([1, 3], Storage::I64(vec![0, 1, -1])).unwrap(),
            ),
        ]);
        assert!(matches!(
            CapturedReplayExecutor::default().replay(
                &capture,
                &bindings,
                CapturedReplayOptions::default()
            ),
            Err(ReplayError::Execute(reason)) if reason.contains("IndexOutOfBounds")
        ));
    }

    #[test]
    fn deserialized_native_multi_item_replay_matches_oracle_and_hits_cache() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([5]), DType::F32);
        let shared = graph.square(x).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(shared, one).unwrap();
        let right = graph.mul(shared, one).unwrap();
        let capture = captured(&graph, &[left, right]);
        let bindings = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars([5], DType::F32, [-2., -1., 0., 1., 2.].map(Scalar::F))
                .unwrap(),
        )]);
        let oracle_bindings = bindings.clone().into_iter().collect::<HashMap<_, _>>();
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &bindings, options).unwrap();
        let second = executor.replay(&capture, &bindings, options).unwrap();
        for ((actual, again), node) in first.outputs.iter().zip(&second.outputs).zip([left, right])
        {
            let expected = CpuBackend.execute(&graph, node, &oracle_bindings).unwrap();
            assert_eq!(actual.storage(), expected.storage());
            assert_eq!(again.storage(), expected.storage());
        }
        assert!(first.trace.items.iter().all(|x| {
            x.backend == ItemBackend::NativeJit
                && !x.cache_hit
                && x.schedule_cache_key == capture.items[x.item as usize].cache_key
        }));
        assert!(second.trace.items.iter().all(|x| x.cache_hit));
        assert_eq!(executor.compile_cache_len(false), capture.items.len());
    }

    #[test]
    fn native_view_reduction_and_zero_domain_match_interpreter() {
        let executor = CapturedReplayExecutor::default();
        let native = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let interpreter = CapturedReplayOptions::default();

        let mut view_graph = Graph::new();
        let x = view_graph.input_dtype("x", Shape::from([5]), DType::F32);
        let view = view_graph.shrink(x, [(1, 5)]).unwrap();
        let output = view_graph.neg(view).unwrap();
        let view_capture = captured(&view_graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::new([5], vec![0., 1., 2., 3., 4.]).unwrap(),
        )]);
        let view_result = executor
            .replay(
                &view_capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::JitFallback { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            view_result.outputs[0].storage(),
            executor
                .replay(&view_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );
        assert_eq!(view_result.trace.items[0].backend, ItemBackend::NativeJit);

        let mut reduction_graph = Graph::new();
        let x = reduction_graph.input_dtype("x", Shape::from([2, 3]), DType::F32);
        let output = reduction_graph
            .reduce(x, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let reduction_capture = captured(&reduction_graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
        )]);
        assert_eq!(
            executor
                .replay(&reduction_capture, &values, native)
                .unwrap()
                .outputs[0]
                .storage(),
            executor
                .replay(&reduction_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );

        let mut empty_graph = Graph::new();
        let x = empty_graph.input_dtype("x", Shape::from([0]), DType::F32);
        let output = empty_graph.square(x).unwrap();
        let empty_capture = captured(&empty_graph, &[output]);
        let values = BTreeMap::from([("x".into(), TensorData::new([0], vec![]).unwrap())]);
        assert_eq!(
            executor
                .replay(&empty_capture, &values, native)
                .unwrap()
                .outputs[0]
                .storage(),
            executor
                .replay(&empty_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );

        let mut vector_graph = Graph::new();
        let x = vector_graph.input_dtype("x", Shape::from([5]), DType::F32);
        let output = vector_graph.square(x).unwrap();
        let vector_capture = captured(&vector_graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::new([5], vec![-2., -1., 0., 1., 2.]).unwrap(),
        )]);
        let vector = executor
            .replay(
                &vector_capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(
            vector.outputs[0].storage(),
            executor
                .replay(&vector_capture, &values, interpreter)
                .unwrap()
                .outputs[0]
                .storage()
        );
        assert_eq!(vector.trace.items[0].backend, ItemBackend::NativeJit);
        assert!(vector.trace.items[0].lanes > 1);
        assert_eq!(vector.trace.items[0].vector_main, 4);
        assert_eq!(vector.trace.items[0].vector_tail, 1);
    }

    #[test]
    fn captured_typed_unbroadcast_matches_interpreter_and_native() {
        let mut graph = Graph::new();
        let target = graph.input_dtype("target", [1], DType::BF16);
        let other = graph.input_dtype("other", [3], DType::BF16);
        let output = graph.add(target, other).unwrap();
        let seed = graph.input_dtype_requires_grad("seed", [3], DType::BF16, false);
        let gradient = graph.grad_with(output, target, Some(seed), true).unwrap();
        let capture = captured(&graph, &[gradient]);
        assert!(!capture.items.is_empty());
        assert!(capture.items.iter().all(|item| item.boundary.is_none()));
        assert_eq!(capture.inputs.len(), 1);
        assert_eq!(capture.inputs[0].name, "seed");

        let bindings = BTreeMap::from([(
            "seed".into(),
            TensorData::from_scalars(
                [3],
                DType::BF16,
                [Scalar::F(256.0), Scalar::F(1.0), Scalar::F(-256.0)],
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let interpreted = executor
            .replay(&capture, &bindings, CapturedReplayOptions::default())
            .unwrap();
        let native = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        let expected = TensorData::from_scalars([1], DType::BF16, [Scalar::F(1.0)]).unwrap();
        assert_eq!(interpreted.outputs, vec![expected.clone()]);
        assert_eq!(native.outputs, vec![expected]);
        assert!(
            native
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
    }

    #[test]
    fn captured_broadcast_matmul_vjp_matches_interpreter_and_native() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1, 2, 3], dtype);
            let rhs = graph.input_dtype("rhs", [1, 2, 3, 2], dtype);
            let output = graph.matmul(lhs, rhs).unwrap();
            let seed = graph.input_dtype_requires_grad("seed", [2, 2, 2, 2], dtype, false);
            let gradient = graph.grad_with(output, lhs, Some(seed), true).unwrap();
            let capture = captured(&graph, &[gradient]);
            assert!(capture.items.iter().all(|item| item.boundary.is_none()));
            assert!(
                (0..graph.node_count())
                    .map(crate::NodeId::from_index)
                    .all(|node| !matches!(
                        graph.op(node).unwrap(),
                        crate::Op::MatmulGrad { .. } | crate::Op::MatmulGradVjp { .. }
                    ))
            );

            let bindings = BTreeMap::from([
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [1, 2, 3, 2],
                        dtype,
                        [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
                            .into_iter()
                            .map(Scalar::F),
                    )
                    .unwrap(),
                ),
                (
                    "seed".into(),
                    TensorData::from_scalars(
                        [2, 2, 2, 2],
                        dtype,
                        std::iter::repeat_n(Scalar::F(1.0), 16),
                    )
                    .unwrap(),
                ),
            ]);
            let executor = CapturedReplayExecutor::default();
            let interpreted = executor
                .replay(&capture, &bindings, CapturedReplayOptions::default())
                .unwrap();
            let native = executor
                .replay(
                    &capture,
                    &bindings,
                    CapturedReplayOptions {
                        backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                    },
                )
                .unwrap();
            let expected = TensorData::from_scalars(
                [2, 1, 2, 3],
                dtype,
                [18., 26., 34., 18., 26., 34., 18., 26., 34., 18., 26., 34.]
                    .into_iter()
                    .map(Scalar::F),
            )
            .unwrap();
            assert_eq!(interpreted.outputs, vec![expected.clone()]);
            assert_eq!(native.outputs, vec![expected]);
            assert!(
                native
                    .trace
                    .items
                    .iter()
                    .all(|item| item.backend == ItemBackend::NativeJit)
            );
        }
    }

    #[test]
    fn unsupported_native_policy_is_explicit() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2]), DType::F32);
        let y = graph.input_dtype("y", Shape::from([2]), DType::F32);
        // Raw Pow remains captured-interpreter-complete but outside the native
        // C renderer's deliberately bounded GraphBinary subset.
        let output = graph.binary(crate::BinaryOp::Pow, x, y).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([
            ("x".into(), TensorData::new([2], vec![2.0, 3.0]).unwrap()),
            ("y".into(), TensorData::new([2], vec![3.0, 2.0]).unwrap()),
        ]);
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            executor.replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Unsupported(_))
        ));
        let fallback = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::JitFallback { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(fallback.trace.items[0].backend, ItemBackend::JitFallback);
        assert_eq!(
            fallback.outputs[0].storage(),
            capture.replay(&values).unwrap()[0].storage()
        );
        assert_eq!(executor.compile_cache_len(false), 0);
    }

    #[test]
    fn native_log2_replay_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.log2(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                Shape::from([3]),
                DType::F32,
                [0.5, 1.0, 8.0].into_iter().map(Scalar::F),
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, options).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, options).unwrap();
        assert_eq!(
            first.outputs[0].storage(),
            capture.replay(&values).unwrap()[0].storage()
        );
        assert!(
            first
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        assert_eq!(
            first.trace.items[0].native_cache_key,
            second.trace.items[0].native_cache_key
        );
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(vector.outputs[0].storage(), first.outputs[0].storage());
        assert!(
            vector
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_exact_negation_is_strict_wrapping_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([1]), DType::I64);
        let output = graph.neg(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(Shape::from([1]), DType::I64, [Scalar::I(i64::MIN)]).unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let scalar = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, scalar).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, scalar).unwrap();
        assert_eq!(
            first.outputs[0].storage(),
            capture.replay(&values).unwrap()[0].storage()
        );
        assert_eq!(first.outputs[0].storage(), values["x"].storage());
        assert_eq!(first.trace.items[0].backend, ItemBackend::NativeJit);
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        let mut warm_trace = second.trace.clone();
        warm_trace.items[0].cache_hit = false;
        assert_eq!(first.trace, warm_trace);
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(vector.outputs[0].storage(), first.outputs[0].storage());
        assert_eq!(vector.trace.items[0].backend, ItemBackend::NativeJit);
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_signed_right_shift_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([1]), DType::I64);
        let shift = graph.input_dtype("shift", Shape::from([1]), DType::I64);
        let output = graph.shr(x, shift).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([
            (
                "x".into(),
                TensorData::from_scalars(Shape::from([1]), DType::I64, [Scalar::I(i64::MIN)])
                    .unwrap(),
            ),
            (
                "shift".into(),
                TensorData::from_scalars(Shape::from([1]), DType::I64, [Scalar::I(63)]).unwrap(),
            ),
        ]);
        let executor = CapturedReplayExecutor::default();
        let scalar = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, scalar).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, scalar).unwrap();
        assert_eq!(
            first.outputs[0].storage(),
            capture.replay(&values).unwrap()[0].storage()
        );
        assert!(matches!(first.outputs[0].scalar_at(0), Scalar::I(-1)));
        assert_eq!(first.trace.items[0].backend, ItemBackend::NativeJit);
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        let mut warm_trace = second.trace.clone();
        warm_trace.items[0].cache_hit = false;
        assert_eq!(first.trace, warm_trace);
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(vector.outputs[0].storage(), first.outputs[0].storage());
        assert_eq!(vector.trace.items[0].backend, ItemBackend::NativeJit);
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_exp2_replay_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.exp2(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                Shape::from([3]),
                DType::F32,
                [-1.0, 0.0, 3.0].into_iter().map(Scalar::F),
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, options).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, options).unwrap();
        assert_eq!(
            first.outputs[0].storage(),
            capture.replay(&values).unwrap()[0].storage()
        );
        assert!(
            first
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        assert_eq!(
            first.trace.items[0].native_cache_key,
            second.trace.items[0].native_cache_key
        );
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(vector.outputs[0].storage(), first.outputs[0].storage());
        assert!(
            vector
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_sin_replay_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.sin(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                Shape::from([3]),
                DType::F32,
                [-1.0, 0.0, 0.5].into_iter().map(Scalar::F),
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, options).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, options).unwrap();
        let expected = capture.replay(&values).unwrap();
        for index in 0..first.outputs[0].len() {
            assert!(
                (first.outputs[0].scalar_at(index).as_f64()
                    - expected[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "index={index}"
            );
        }
        assert!(
            first
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        let mut warm_trace = second.trace.clone();
        warm_trace.items[0].cache_hit = false;
        assert_eq!(first.trace, warm_trace);
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        for index in 0..vector.outputs[0].len() {
            assert!(
                (vector.outputs[0].scalar_at(index).as_f64()
                    - first.outputs[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "vector index={index}"
            );
        }
        assert!(
            vector
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_tan_replay_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.tan(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                Shape::from([3]),
                DType::F32,
                [-1.0, 0.0, 0.5].into_iter().map(Scalar::F),
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, options).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, options).unwrap();
        let expected = capture.replay(&values).unwrap();
        for index in 0..first.outputs[0].len() {
            assert!(
                (first.outputs[0].scalar_at(index).as_f64()
                    - expected[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "index={index}"
            );
        }
        assert!(
            first
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        let mut warm_trace = second.trace.clone();
        warm_trace.items[0].cache_hit = false;
        assert_eq!(first.trace, warm_trace);
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        for index in 0..vector.outputs[0].len() {
            assert!(
                (vector.outputs[0].scalar_at(index).as_f64()
                    - first.outputs[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "vector index={index}"
            );
        }
        assert!(
            vector
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_cos_replay_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.cos(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                Shape::from([3]),
                DType::F32,
                [-1.0, 0.0, 0.5].into_iter().map(Scalar::F),
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, options).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, options).unwrap();
        let expected = capture.replay(&values).unwrap();
        for index in 0..first.outputs[0].len() {
            assert!(
                (first.outputs[0].scalar_at(index).as_f64()
                    - expected[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "index={index}"
            );
        }
        assert!(
            first
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        let mut warm_trace = second.trace.clone();
        warm_trace.items[0].cache_hit = false;
        assert_eq!(first.trace, warm_trace);
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        for index in 0..vector.outputs[0].len() {
            assert!(
                (vector.outputs[0].scalar_at(index).as_f64()
                    - first.outputs[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "vector index={index}"
            );
        }
        assert!(
            vector
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_log_replay_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.log(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                Shape::from([3]),
                DType::F32,
                [0.5, 1.0, 2.0].into_iter().map(Scalar::F),
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, options).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, options).unwrap();
        let expected = capture.replay(&values).unwrap();
        for index in 0..first.outputs[0].len() {
            assert!(
                (first.outputs[0].scalar_at(index).as_f64()
                    - expected[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "index={index}"
            );
        }
        assert!(
            first
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        let mut warm_trace = second.trace.clone();
        warm_trace.items[0].cache_hit = false;
        assert_eq!(first.trace, warm_trace);
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        for index in 0..vector.outputs[0].len() {
            assert!(
                (vector.outputs[0].scalar_at(index).as_f64()
                    - first.outputs[0].scalar_at(index).as_f64())
                .abs()
                    <= 1e-6,
                "vector index={index}"
            );
        }
        assert!(
            vector
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_trunc_replay_is_strict_and_cacheable() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.trunc(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                Shape::from([3]),
                DType::F32,
                [-1.75, -0.0, 2.5].into_iter().map(Scalar::F),
            )
            .unwrap(),
        )]);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let first = executor.replay(&capture, &values, options).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = executor.replay(&capture, &values, options).unwrap();
        assert_eq!(
            first.outputs[0].storage(),
            capture.replay(&values).unwrap()[0].storage()
        );
        assert!(
            first
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert!(!first.trace.items[0].cache_hit);
        assert!(second.trace.items[0].cache_hit);
        let mut warm_trace = second.trace.clone();
        warm_trace.items[0].cache_hit = false;
        assert_eq!(first.trace, warm_trace);
        assert_eq!(cached, executor.compile_cache_len(false));

        let vector = executor
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(vector.outputs[0].storage(), first.outputs[0].storage());
        assert!(
            vector
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_ne!(
            vector.trace.items[0].native_cache_key,
            first.trace.items[0].native_cache_key
        );
        assert_eq!(executor.compile_cache_len(true), 1);
    }

    #[test]
    fn native_replay_translates_schedule_operand_order_to_native_abi() {
        let mut graph = Graph::new();
        let right = graph.input_dtype("right", Shape::from([2]), DType::F32);
        let left = graph.input_dtype("left", Shape::from([2]), DType::F32);
        let output = graph.sub(left, right).unwrap();
        let capture = captured(&graph, &[output]);
        assert_eq!(capture.items[0].input_bindings[0].input_node, left);
        assert_eq!(capture.items[0].input_bindings[1].input_node, right);
        let values = BTreeMap::from([
            ("left".into(), TensorData::new([2], vec![7., 11.]).unwrap()),
            ("right".into(), TensorData::new([2], vec![2., 3.]).unwrap()),
        ]);
        let result = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &values,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(result.outputs[0].values(), &[5., 8.]);
    }

    #[test]
    fn batch_preflight_order_and_owned_outputs_are_deterministic() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let output = graph.square(x).unwrap();
        let capture = captured(&graph, &[output]);
        let first = BTreeMap::from([("x".into(), TensorData::new([3], vec![1., 2., 3.]).unwrap())]);
        let second =
            BTreeMap::from([("x".into(), TensorData::new([3], vec![4., 5., 6.]).unwrap())]);
        let executor = CapturedReplayExecutor::default();
        let malformed = CapturedBatch::new(
            &capture,
            [
                first.clone(),
                BTreeMap::from([("x".into(), TensorData::scalar(1.0))]),
            ],
        );
        assert!(matches!(
            malformed,
            Err(ReplayError::Batch { invocation: 1, .. })
        ));
        assert_eq!(executor.compile_cache_len(false), 0);

        let batch = CapturedBatch::new(&capture, [first, second]).unwrap();
        let mut wrong_artifact = batch.clone();
        wrong_artifact.artifact_identity ^= 1;
        assert!(matches!(
            executor.replay_batch(
                &capture,
                &wrong_artifact,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Corrupt(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
        let result = executor
            .replay_batch(
                &capture,
                &batch,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(result.invocations[0].outputs[0].values(), &[1., 4., 9.]);
        assert_eq!(result.invocations[1].outputs[0].values(), &[16., 25., 36.]);
        assert_eq!(result.invocations[0].trace.items[0].invocation, 0);
        assert_eq!(result.invocations[1].trace.items[0].invocation, 1);
        assert!(!result.invocations[0].trace.items[0].cache_hit);
        assert!(result.invocations[1].trace.items[0].cache_hit);
        assert_ne!(
            result.invocations[0].outputs[0].values().as_ptr(),
            result.invocations[1].outputs[0].values().as_ptr()
        );
    }

    #[test]
    fn matmul_artifacts_replay_interpreter_native_and_batches() {
        struct Case {
            name: &'static str,
            dtype: DType,
            lhs: Vec<usize>,
            rhs: Vec<usize>,
        }
        let cases = [
            Case {
                name: "dot",
                dtype: DType::F32,
                lhs: vec![3],
                rhs: vec![3],
            },
            Case {
                name: "matvec",
                dtype: DType::F64,
                lhs: vec![2, 3],
                rhs: vec![3],
            },
            Case {
                name: "vecmat",
                dtype: DType::F32,
                lhs: vec![3],
                rhs: vec![3, 2],
            },
            Case {
                name: "broadcast batch",
                dtype: DType::F64,
                lhs: vec![2, 1, 2, 3],
                rhs: vec![1, 4, 3, 2],
            },
            Case {
                name: "zero k",
                dtype: DType::F32,
                lhs: vec![2, 0],
                rhs: vec![0, 3],
            },
        ];
        for case in cases {
            let mut graph = Graph::new();
            let lhs_node = graph.input_dtype("lhs", case.lhs.clone(), case.dtype);
            let rhs_node = graph.input_dtype("rhs", case.rhs.clone(), case.dtype);
            let output = graph.matmul(lhs_node, rhs_node).unwrap();
            let schedule = crate::schedule(&graph, output).unwrap();
            assert_eq!(schedule.items.len(), 1, "{} item count", case.name);
            assert!(
                schedule.items[0].boundary.is_none(),
                "{} boundary",
                case.name
            );
            assert!(matches!(
                schedule.items[0].kernel.operation(),
                crate::Operation::Matmul(_)
            ));
            assert_eq!(
                schedule.items[0]
                    .ordered_inputs()
                    .iter()
                    .map(|binding| binding.input_node)
                    .collect::<Vec<_>>(),
                vec![lhs_node, rhs_node],
                "{} ABI",
                case.name
            );
            let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
            let bytes = capture.to_bytes().unwrap();
            let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
            assert_eq!(bytes, decoded.to_bytes().unwrap(), "{} bytes", case.name);
            let lhs = TensorData::from_scalars(
                case.lhs,
                case.dtype,
                (0..graph.shape(lhs_node).unwrap().numel().unwrap())
                    .map(|index| Scalar::F(index as f64 * 0.25 - 1.0)),
            )
            .unwrap();
            let rhs = TensorData::from_scalars(
                case.rhs,
                case.dtype,
                (0..graph.shape(rhs_node).unwrap().numel().unwrap())
                    .map(|index| Scalar::F(index as f64 * -0.125 + 0.75)),
            )
            .unwrap();
            let bindings =
                BTreeMap::from([("lhs".into(), lhs.clone()), ("rhs".into(), rhs.clone())]);
            let oracle = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]),
                )
                .unwrap();
            let executor = CapturedReplayExecutor::default();
            let interpreted = executor
                .replay(&decoded, &bindings, CapturedReplayOptions::default())
                .unwrap();
            let options = CapturedReplayOptions {
                backend: CapturedBackendPolicy::NativeJit { vectorized: false },
            };
            let first = executor.replay(&decoded, &bindings, options).unwrap();
            let second = executor.replay(&decoded, &bindings, options).unwrap();
            assert_eq!(
                interpreted.outputs[0].storage(),
                oracle.storage(),
                "{} interpreter",
                case.name
            );
            assert_eq!(
                first.outputs[0].storage(),
                oracle.storage(),
                "{} native",
                case.name
            );
            assert_eq!(first.trace.items[0].backend, ItemBackend::NativeJit);
            assert!(!first.trace.items[0].cache_hit);
            assert!(second.trace.items[0].cache_hit);
        }

        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 2], DType::F32);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let capture = captured(&graph, &[output]);
        let invocation = |offset: f32| {
            BTreeMap::from([
                (
                    "lhs".into(),
                    TensorData::new([2, 2], vec![offset, 1., 2., 3.]).unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::new([2, 2], vec![1., 2., 3., offset]).unwrap(),
                ),
            ])
        };
        let batch = CapturedBatch::new(&capture, [invocation(4.), invocation(5.)]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let result = executor
            .replay_batch(
                &capture,
                &batch,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            result.invocations[0].outputs[0].values(),
            &[7., 12., 11., 16.]
        );
        assert_eq!(
            result.invocations[1].outputs[0].values(),
            &[8., 15., 11., 19.]
        );
        assert!(!result.invocations[0].trace.items[0].cache_hit);
        assert!(result.invocations[1].trace.items[0].cache_hit);
        assert_eq!(executor.compile_cache_len(false), 1);

        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::F32);
        let bias = graph.input_dtype("bias", [2, 2], DType::F32);
        let squared = graph.square(input).unwrap();
        let product = graph.matmul(squared, rhs).unwrap();
        let output = graph.add(product, bias).unwrap();
        let capture = captured(&graph, &[output]);
        let bindings = BTreeMap::from([
            (
                "input".into(),
                TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::new([2, 2], vec![2., 1., 0., 3.]).unwrap(),
            ),
            (
                "bias".into(),
                TensorData::new([2, 2], vec![1., 1., 1., 1.]).unwrap(),
            ),
        ]);
        let oracle = CpuBackend
            .execute(
                &graph,
                output,
                &bindings.clone().into_iter().collect::<HashMap<_, _>>(),
            )
            .unwrap();
        let executor = CapturedReplayExecutor::default();
        let replay = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(replay.outputs[0].storage(), oracle.storage());
        assert_eq!(replay.trace.items.len(), 3);
        assert!(
            replay
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        );
        assert_eq!(executor.compile_cache_len(false), 3);
    }

    #[test]
    fn matmul_native_dtype_and_artifact_abi_fail_before_compilation() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 2], DType::F16);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::F16);
        let output = graph.matmul(lhs, rhs).unwrap();
        let capture = captured(&graph, &[output]);
        let bindings = BTreeMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([2, 2], DType::F16, [1., 2., 3., 4.].map(Scalar::F))
                    .unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([2, 2], DType::F16, [4., 3., 2., 1.].map(Scalar::F))
                    .unwrap(),
            ),
        ]);
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            executor.replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Unsupported(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
        let fallback = executor
            .replay(
                &capture,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::JitFallback { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(fallback.trace.items[0].backend, ItemBackend::JitFallback);
        assert_eq!(
            fallback.outputs[0].storage(),
            capture.replay(&bindings).unwrap()[0].storage()
        );

        let mut malformed_abi = capture.clone();
        malformed_abi.items[0].input_bindings.swap(0, 1);
        assert!(matches!(
            executor.replay(
                &malformed_abi,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Corrupt(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);

        let mut malformed_plan = capture;
        let crate::Operation::Matmul(crate::MatmulValue::Serial(plan)) =
            malformed_plan.items[0].kernel.operation()
        else {
            panic!("matmul payload missing");
        };
        let mut plan = plan.clone();
        plan.output_shape = Shape::from([4]);
        malformed_plan.items[0].kernel = crate::UOp::from_operation(
            crate::Operation::Matmul(crate::MatmulValue::Serial(plan)),
            Some(crate::UType::scalar(DType::F16)),
            vec![],
        );
        assert!(matches!(
            executor.replay(
                &malformed_plan,
                &bindings,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false }
                }
            ),
            Err(ReplayError::Corrupt(_))
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
    }

    fn symbolic_family(
        extent: usize,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [extent, 3], DType::F32);
        let bias = graph.input_dtype("bias", [1, 3], DType::F32);
        let weight = graph.input_dtype("weight", [3, extent], DType::F32);
        let shifted = graph.add(x, bias).unwrap();
        let reduced = graph
            .reduce(shifted, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let product = graph.matmul(shifted, weight).unwrap();
        let bindings = BTreeMap::from([
            (
                "x".into(),
                TensorData::from_scalars(
                    [extent, 3],
                    DType::F32,
                    (0..extent * 3).map(|index| Scalar::F(index as f64 * 0.25 - 1.0)),
                )
                .unwrap(),
            ),
            (
                "bias".into(),
                TensorData::new([1, 3], vec![0.5, -0.25, 1.0]).unwrap(),
            ),
            (
                "weight".into(),
                TensorData::from_scalars(
                    [3, extent],
                    DType::F32,
                    (0..extent * 3).map(|index| Scalar::F(index as f64 * -0.125 + 0.75)),
                )
                .unwrap(),
            ),
        ]);
        (graph, reduced, product, bindings)
    }

    fn symbolic_view_family(extent: usize) -> (Graph, crate::NodeId, BTreeMap<String, TensorData>) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [extent, 4], DType::F32);
        let reshape = graph.reshape(input, [extent, 2, 2]).unwrap();
        let permute = graph.permute(reshape, [0, 2, 1]).unwrap();
        let stride = graph
            .stride(
                permute,
                [
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 2,
                    },
                ],
            )
            .unwrap();
        let expand = graph.expand(stride, [extent, 2, extent]).unwrap();
        let first = graph
            .shrink(expand, [(0, extent), (0, 2), (0, extent)])
            .unwrap();
        let second = graph
            .shrink(first, [(0, extent), (0, 2), (0, extent)])
            .unwrap();
        let output = graph.neg(second).unwrap();
        let values = TensorData::from_scalars(
            [extent, 4],
            DType::F32,
            (0..extent * 4).map(|index| Scalar::F(index as f64 + 0.25)),
        )
        .unwrap();
        (graph, output, BTreeMap::from([("input".into(), values)]))
    }

    #[test]
    fn symbolic_source_view_contiguous_specializes_the_affine_copy_plan() {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let viewed = graph.permute(input, [1, 0]).unwrap();
        let output = graph.contiguous(viewed).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        assert_eq!(schedule.items.len(), 1);
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                crate::SymbolicShape::new(vec![2usize.into(), extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 3)]),
        )
        .unwrap();
        let bytes = capture.to_bytes().unwrap();
        let capture = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(capture.to_bytes().unwrap(), bytes);

        let executor = CapturedReplayExecutor::default();
        let empty = executor
            .replay_symbolic(
                &capture,
                &BTreeMap::from([("extent".into(), 0)]),
                &BTreeMap::from([(
                    "input".into(),
                    TensorData::from_storage([2, 0], Storage::F32(vec![])).unwrap(),
                )]),
                CapturedReplayOptions::default(),
            )
            .unwrap();
        assert_eq!(empty.outputs[0].shape(), &Shape::new([0, 2]));

        let rebound = executor
            .replay_symbolic(
                &capture,
                &BTreeMap::from([("extent".into(), 4)]),
                &BTreeMap::from([(
                    "input".into(),
                    TensorData::new([2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
                )]),
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(rebound.outputs[0].shape(), &Shape::new([4, 2]));
        assert_eq!(
            rebound.outputs[0].values(),
            &[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]
        );
    }

    fn symbolic_computed_permute_family(
        extent: usize,
        contiguous: bool,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [extent, 2], DType::F32);
        let producer = graph.square(input).unwrap();
        let viewed = graph.permute(producer, [1, 0]).unwrap();
        let output = if contiguous {
            graph.contiguous(viewed).unwrap()
        } else {
            viewed
        };
        let values = (0..extent * 2)
            .map(|index| index as f32 - 2.0)
            .collect::<Vec<_>>();
        (
            graph,
            input,
            producer,
            output,
            BTreeMap::from([(
                "input".into(),
                TensorData::new([extent, 2], values).unwrap(),
            )]),
        )
    }

    #[test]
    fn symbolic_computed_affine_copy_specializes_direct_and_contiguous_outputs() {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        for contiguous in [false, true] {
            let (template, input, producer, output, _) =
                symbolic_computed_permute_family(3, contiguous);
            let requested = if contiguous {
                vec![producer, output]
            } else {
                vec![output]
            };
            let schedule = crate::schedule_many(&template, &requested).unwrap();
            assert_eq!(schedule.items.len(), 2);
            let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
                schedule.items[1].kernel.operation()
            else {
                panic!("computed view must retain its affine copy boundary")
            };
            let crate::MovementKernelKind::AffineCopy { input: operand, .. } = &plan.kind else {
                panic!("computed view must retain its affine copy plan")
            };
            assert_eq!(operand.node, producer);
            let capture = CapturedSchedule::capture_symbolic(
                &template,
                &schedule,
                &requested,
                &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                    input,
                    crate::SymbolicShape::new(vec![extent.clone().into(), 2usize.into()]),
                )])),
                &BTreeMap::from([("extent".into(), 3)]),
            )
            .unwrap();
            let bytes = capture.to_bytes().unwrap();
            let capture = CapturedSchedule::from_bytes(&bytes).unwrap();
            assert_eq!(capture.to_bytes().unwrap(), bytes);

            let executor = CapturedReplayExecutor::default();
            for rebound in [0usize, 1, 3, 8] {
                let (oracle_graph, _, _, oracle_output, bindings) =
                    symbolic_computed_permute_family(rebound, contiguous);
                let replayed = executor
                    .replay_symbolic(
                        &capture,
                        &BTreeMap::from([("extent".into(), rebound as i64)]),
                        &bindings,
                        CapturedReplayOptions {
                            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                        },
                    )
                    .unwrap();
                let oracle = CpuBackend
                    .execute(
                        &oracle_graph,
                        oracle_output,
                        &bindings.into_iter().collect::<HashMap<_, _>>(),
                    )
                    .unwrap();
                let output = replayed.outputs.last().unwrap();
                assert_eq!(output.storage(), oracle.storage(), "{rebound}");
                assert_eq!(output.shape(), oracle.shape(), "{rebound}");
            }

            let mut tampered = capture.clone();
            let view = tampered
                .symbolic
                .as_mut()
                .unwrap()
                .views
                .values_mut()
                .next()
                .unwrap();
            view.strides[0] = crate::SymbolicExpr::constant(99);
            tampered.identity = 0;
            tampered.identity = crate::schedule::artifact::identity(&tampered).unwrap();
            assert!(crate::schedule::artifact::validate_capture(&tampered).is_err());
        }
    }

    fn symbolic_computed_reverse_family(
        extent: usize,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [extent], DType::F32);
        let producer = graph.square(input).unwrap();
        let output = graph.flip(producer, [0]).unwrap();
        let values = (0..extent)
            .map(|index| index as f32 + 1.0)
            .collect::<Vec<_>>();
        (
            graph,
            input,
            output,
            BTreeMap::from([("input".into(), TensorData::new([extent], values).unwrap())]),
        )
    }

    #[test]
    fn symbolic_computed_full_reverse_specializes_signed_affine_copy() {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        let (template, input, output, _) = symbolic_computed_reverse_family(3);
        let schedule = crate::schedule(&template, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                crate::SymbolicShape::new(vec![extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 3)]),
        )
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        for rebound in [0usize, 1, 3, 8] {
            let (oracle_graph, _, oracle_output, bindings) =
                symbolic_computed_reverse_family(rebound);
            let specialized = executor
                .specialize(
                    &capture,
                    &BTreeMap::from([("extent".into(), rebound as i64)]),
                )
                .unwrap();
            let affine = specialized
                .capture()
                .items
                .iter()
                .find_map(|item| match item.kernel.operation() {
                    crate::Operation::Movement(crate::MovementValue::Plan(plan)) => {
                        match &plan.kind {
                            crate::MovementKernelKind::AffineCopy { view, .. } => Some(view),
                            _ => None,
                        }
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                affine.offset,
                if rebound == 0 { 0 } else { rebound as i64 - 1 }
            );
            assert_eq!(affine.strides, vec![-1]);
            let replayed = executor
                .replay_symbolic(
                    &capture,
                    &BTreeMap::from([("extent".into(), rebound as i64)]),
                    &bindings,
                    CapturedReplayOptions::default(),
                )
                .unwrap();
            let oracle = CpuBackend
                .execute(
                    &oracle_graph,
                    oracle_output,
                    &bindings.into_iter().collect::<HashMap<_, _>>(),
                )
                .unwrap();
            assert_eq!(replayed.outputs[0].storage(), oracle.storage(), "{rebound}");
        }
    }

    fn symbolic_computed_broadcast_shrink_family(
        extent: usize,
    ) -> (
        Graph,
        crate::NodeId,
        crate::NodeId,
        BTreeMap<String, TensorData>,
    ) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [extent], DType::F32);
        let producer = graph.square(input).unwrap();
        let reshaped = graph.reshape(producer, [extent, 1]).unwrap();
        let expanded = graph.expand(reshaped, [extent, 3]).unwrap();
        let output = graph.shrink(expanded, [(0, extent), (0, 2)]).unwrap();
        let values = (0..extent)
            .map(|index| index as f32 - 1.0)
            .collect::<Vec<_>>();
        (
            graph,
            input,
            output,
            BTreeMap::from([("input".into(), TensorData::new([extent], values).unwrap())]),
        )
    }

    #[test]
    fn symbolic_computed_reshape_expand_and_shrink_share_one_affine_copy() {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        // Keep the template extent distinct from the fixed expanded width so
        // shape lifting does not have two source-exact interpretations.
        let (template, input, output, _) = symbolic_computed_broadcast_shrink_family(2);
        let schedule = crate::schedule(&template, output).unwrap();
        assert_eq!(schedule.items.len(), 2);
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                crate::SymbolicShape::new(vec![extent.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        for rebound in [0usize, 1, 2, 3, 8] {
            let (oracle_graph, _, oracle_output, bindings) =
                symbolic_computed_broadcast_shrink_family(rebound);
            let replayed = executor
                .replay_symbolic(
                    &capture,
                    &BTreeMap::from([("extent".into(), rebound as i64)]),
                    &bindings,
                    CapturedReplayOptions {
                        backend: CapturedBackendPolicy::NativeJit { vectorized: false },
                    },
                )
                .unwrap();
            let oracle = CpuBackend
                .execute(
                    &oracle_graph,
                    oracle_output,
                    &bindings.into_iter().collect::<HashMap<_, _>>(),
                )
                .unwrap();
            assert_eq!(replayed.outputs[0].storage(), oracle.storage(), "{rebound}");
            assert_eq!(replayed.outputs[0].shape(), &Shape::new([rebound, 2]));
        }
    }

    #[test]
    fn symbolic_rebind_failure_does_not_publish_a_specialization() {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let output = graph.square(input).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &crate::schedule(&graph, output).unwrap(),
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                crate::SymbolicShape::new(vec![extent.clone().into()]),
            )])),
            &BTreeMap::from([("extent".into(), 2)]),
        )
        .unwrap();
        let executor = CapturedReplayExecutor::default();
        let symbols = BTreeMap::from([("extent".into(), 2)]);
        let bad = BTreeMap::from([("input".into(), TensorData::new([1], vec![3.0]).unwrap())]);

        assert!(matches!(
            executor.replay_symbolic(
                &capture,
                &symbols,
                &bad,
                CapturedReplayOptions::default(),
            ),
            Err(ReplayError::Descriptor(name)) if name == "input"
        ));
        assert_eq!(bad["input"].values(), &[3.0]);
        assert_eq!(executor.specialization_cache_len(), 0);
        assert_eq!(executor.compile_cache_len(false), 0);

        let good = BTreeMap::from([(
            "input".into(),
            TensorData::new([2], vec![2.0, -3.0]).unwrap(),
        )]);
        let first = executor
            .replay_symbolic(&capture, &symbols, &good, CapturedReplayOptions::default())
            .unwrap();
        let second = executor
            .replay_symbolic(&capture, &symbols, &good, CapturedReplayOptions::default())
            .unwrap();
        assert_eq!(first.outputs[0].values(), &[4.0, 9.0]);
        assert!(!first.specialization.unwrap().cache_hit);
        assert!(second.specialization.unwrap().cache_hit);
        assert_eq!(executor.specialization_cache_len(), 1);
    }

    #[test]
    fn symbolic_artifact_specializes_replays_and_separates_caches() {
        let n = crate::SymbolicExpr::variable("n", 0, 8).unwrap();
        let m = crate::SymbolicExpr::variable("m", 0, 8).unwrap();
        let (template, reduced, product, _) = symbolic_family(2);
        let x = template
            .op(reduced)
            .ok()
            .and_then(|op| match op {
                crate::Op::Reduce { input, .. } => template.op(*input).ok(),
                _ => None,
            })
            .and_then(|op| match op {
                crate::Op::Binary { lhs, .. } => Some(*lhs),
                _ => None,
            })
            .unwrap();
        let weight = match template.op(product).unwrap() {
            crate::Op::Matmul { rhs, .. } => *rhs,
            _ => unreachable!(),
        };
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([
            (
                x,
                crate::SymbolicShape::new(vec![n.clone().into(), 3usize.into()]),
            ),
            (
                weight,
                crate::SymbolicShape::new(vec![3usize.into(), m.clone().into()]),
            ),
        ]))
        .with_guard(crate::SymbolicGuard::equal(n.clone(), m.clone()))
        .with_guard(crate::SymbolicGuard::divisible(n, 2).unwrap());
        let schedule = crate::schedule_many(&template, &[reduced, product]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[reduced, product],
            &spec,
            &BTreeMap::from([("n".into(), 2), ("m".into(), 2)]),
        )
        .unwrap();
        assert!(capture.is_symbolic());
        assert_eq!(capture.symbolic_parameters().len(), 2);
        let bytes = capture.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, decoded.to_bytes().unwrap());

        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        let probe = CapturedReplayExecutor::default();
        let probe_bindings = BTreeMap::from([("n".into(), 2), ("m".into(), 2)]);
        let first_specialization = probe.specialize(&decoded, &probe_bindings).unwrap();
        let second_specialization = probe.specialize(&decoded, &probe_bindings).unwrap();
        assert!(!first_specialization.trace().cache_hit);
        assert!(second_specialization.trace().cache_hit);
        assert_eq!(
            first_specialization.trace().concrete_identity,
            second_specialization.trace().concrete_identity
        );
        let specialized_bytes = first_specialization.capture().to_bytes().unwrap();
        assert_eq!(
            specialized_bytes,
            CapturedSchedule::from_bytes(&specialized_bytes)
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        let mut concrete_identities = BTreeSet::new();
        for (case, extent) in [("first", 2usize), ("second", 4), ("zero", 0)] {
            let (oracle_graph, oracle_reduced, oracle_product, bindings) = symbolic_family(extent);
            let symbols =
                BTreeMap::from([("n".into(), extent as i64), ("m".into(), extent as i64)]);
            let first = executor
                .replay_symbolic(&decoded, &symbols, &bindings, options)
                .unwrap();
            let second = executor
                .replay_symbolic(&decoded, &symbols, &bindings, options)
                .unwrap();
            let interpreted = CapturedReplayExecutor::default()
                .replay_symbolic(
                    &decoded,
                    &symbols,
                    &bindings,
                    CapturedReplayOptions::default(),
                )
                .unwrap();
            let oracle_bindings = bindings.clone().into_iter().collect::<HashMap<_, _>>();
            for (index, output) in [oracle_reduced, oracle_product].into_iter().enumerate() {
                let oracle = CpuBackend
                    .execute(&oracle_graph, output, &oracle_bindings)
                    .unwrap();
                assert_eq!(first.outputs[index].storage(), oracle.storage(), "{case}");
                assert_eq!(second.outputs[index].storage(), oracle.storage(), "{case}");
                assert_eq!(
                    interpreted.outputs[index].storage(),
                    oracle.storage(),
                    "{case} interpreter"
                );
            }
            assert!(!first.specialization.as_ref().unwrap().cache_hit, "{case}");
            assert!(second.specialization.as_ref().unwrap().cache_hit, "{case}");
            concrete_identities.insert(first.specialization.as_ref().unwrap().concrete_identity);
            assert!(
                second.trace.items.iter().all(|item| item.cache_hit),
                "{case}"
            );
        }
        assert_eq!(executor.specialization_cache_len(), 3);
        assert_eq!(executor.compile_cache_len(false), 9);
        assert_eq!(concrete_identities.len(), 3);

        let (_, _, _, wrong) = symbolic_family(2);
        for symbols in [
            BTreeMap::from([("n".into(), 2), ("m".into(), 4)]),
            BTreeMap::from([("n".into(), 3), ("m".into(), 3)]),
        ] {
            assert!(matches!(
                executor.replay_symbolic(&decoded, &symbols, &wrong, options),
                Err(ReplayError::Symbolic(_))
            ));
        }
        assert!(matches!(
            executor.replay_symbolic(&decoded, &BTreeMap::new(), &wrong, options),
            Err(ReplayError::Missing(_))
        ));
        assert!(matches!(
            executor.replay_symbolic(
                &decoded,
                &BTreeMap::from([("n".into(), 2), ("m".into(), 2), ("extra".into(), 1)]),
                &wrong,
                options
            ),
            Err(ReplayError::Extra(_))
        ));
        assert!(matches!(
            executor.replay_symbolic(
                &decoded,
                &BTreeMap::from([("n".into(), 10), ("m".into(), 10)]),
                &wrong,
                options
            ),
            Err(ReplayError::Symbolic(_))
        ));
        assert_eq!(executor.specialization_cache_len(), 3);
        assert_eq!(executor.compile_cache_len(false), 9);
    }

    #[test]
    fn symbolic_batch_preflights_every_binding_before_compilation() {
        let n = crate::SymbolicExpr::variable("n", 0, 8).unwrap();
        let (template, reduced, product, _) = symbolic_family(2);
        let (x, weight) = match (template.op(reduced).unwrap(), template.op(product).unwrap()) {
            (crate::Op::Reduce { input, .. }, crate::Op::Matmul { rhs: weight, .. }) => {
                let crate::Op::Binary { lhs: x, .. } = template.op(*input).unwrap() else {
                    unreachable!()
                };
                (*x, *weight)
            }
            _ => unreachable!(),
        };
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([
            (
                x,
                crate::SymbolicShape::new(vec![n.clone().into(), 3usize.into()]),
            ),
            (
                weight,
                crate::SymbolicShape::new(vec![3usize.into(), n.clone().into()]),
            ),
        ]))
        .with_guard(crate::SymbolicGuard::divisible(n, 2).unwrap());
        let schedule = crate::schedule_many(&template, &[reduced, product]).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[reduced, product],
            &spec,
            &BTreeMap::from([("n".into(), 2)]),
        )
        .unwrap();
        let (_, _, _, two_a) = symbolic_family(2);
        let (_, _, _, two_b) = symbolic_family(2);
        let (_, _, _, four) = symbolic_family(4);
        let mut batch = CapturedBatch::new_symbolic(
            &capture,
            [
                (BTreeMap::from([("n".into(), 2)]), two_a),
                (BTreeMap::from([("n".into(), 2)]), two_b),
                (BTreeMap::from([("n".into(), 4)]), four),
            ],
        )
        .unwrap();
        batch.invocations[2].symbolic_bindings.insert("n".into(), 3);
        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: false },
        };
        assert!(matches!(
            executor.replay_batch(&capture, &batch, options),
            Err(ReplayError::Batch { invocation: 2, .. })
        ));
        assert_eq!(executor.compile_cache_len(false), 0);

        batch.invocations[2].symbolic_bindings.insert("n".into(), 4);
        let executor = CapturedReplayExecutor::default();
        let result = executor.replay_batch(&capture, &batch, options).unwrap();
        assert_eq!(result.invocations.len(), 3);
        assert!(
            !result.invocations[0]
                .specialization
                .as_ref()
                .unwrap()
                .cache_hit
        );
        assert!(
            result.invocations[1]
                .specialization
                .as_ref()
                .unwrap()
                .cache_hit
        );
        assert!(
            !result.invocations[2]
                .specialization
                .as_ref()
                .unwrap()
                .cache_hit
        );
        assert!(
            result.invocations[1]
                .trace
                .items
                .iter()
                .all(|item| item.cache_hit)
        );
        assert_eq!(executor.specialization_cache_len(), 2);
        assert_eq!(executor.compile_cache_len(false), 6);
    }

    #[test]
    fn symbolic_capture_rejects_any_domain_with_possible_checked_overflow() {
        let extent = crate::SymbolicExpr::variable("extent", 0, i64::MAX).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1], DType::F32);
        let output = graph.square(input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([(
            input,
            crate::SymbolicShape::new(vec![(extent.clone() * extent).into()]),
        )]));
        assert!(matches!(
            CapturedSchedule::capture_symbolic(
                &graph,
                &schedule,
                &[output],
                &spec,
                &BTreeMap::from([("extent".into(), 1)])
            ),
            Err(ReplayError::Symbolic(_))
        ));
    }

    #[test]
    fn symbolic_affine_views_round_trip_and_replay_across_zero_and_tails() {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        let (template, output, _) = symbolic_view_family(3);
        let input = template
            .op(output)
            .ok()
            .and_then(|op| match op {
                crate::Op::Unary { input, .. } => Some(*input),
                _ => None,
            })
            .and_then(|mut node| {
                loop {
                    match template.op(node).ok()? {
                        crate::Op::Shrink { input, .. }
                        | crate::Op::Reshape { input, .. }
                        | crate::Op::Permute { input, .. }
                        | crate::Op::Expand { input, .. }
                        | crate::Op::Stride { input, .. } => node = *input,
                        crate::Op::Input { .. } => break Some(node),
                        _ => break None,
                    }
                }
            })
            .unwrap();
        let schedule = crate::schedule(&template, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template,
            &schedule,
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                crate::SymbolicShape::new(vec![extent.into(), 4usize.into()]),
            )])),
            &BTreeMap::from([("extent".into(), 3)]),
        )
        .unwrap();
        let bytes = capture.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, decoded.to_bytes().unwrap());
        assert!(decoded.items[0].input_bindings[0].desc.view.is_some());

        let executor = CapturedReplayExecutor::default();
        let options = CapturedReplayOptions {
            backend: CapturedBackendPolicy::NativeJit { vectorized: true },
        };
        for extent in [0usize, 1, 3, 8] {
            let (oracle_graph, oracle_output, bindings) = symbolic_view_family(extent);
            let symbols = BTreeMap::from([("extent".into(), extent as i64)]);
            let native = executor
                .replay_symbolic(&decoded, &symbols, &bindings, options)
                .unwrap();
            let cached = executor
                .replay_symbolic(&decoded, &symbols, &bindings, options)
                .unwrap();
            let interpreted = CapturedReplayExecutor::default()
                .replay_symbolic(
                    &decoded,
                    &symbols,
                    &bindings,
                    CapturedReplayOptions::default(),
                )
                .unwrap();
            let oracle = CpuBackend
                .execute(
                    &oracle_graph,
                    oracle_output,
                    &bindings.clone().into_iter().collect::<HashMap<_, _>>(),
                )
                .unwrap();
            assert_eq!(native.outputs[0].storage(), oracle.storage(), "{extent}");
            assert_eq!(
                interpreted.outputs[0].storage(),
                oracle.storage(),
                "{extent} interpreter"
            );
            assert_eq!(native.trace.items[0].backend, ItemBackend::NativeJit);
            assert!(cached.specialization.as_ref().unwrap().cache_hit);
            assert!(cached.trace.items.iter().all(|item| item.cache_hit));
        }
        assert_eq!(executor.specialization_cache_len(), 4);
        assert_eq!(executor.compile_cache_len(true), 4);

        let invocations = [1usize, 3, 8].map(|extent| {
            let (_, _, bindings) = symbolic_view_family(extent);
            (BTreeMap::from([("extent".into(), extent as i64)]), bindings)
        });
        let mut batch = CapturedBatch::new_symbolic(&decoded, invocations).unwrap();
        batch.invocations[2]
            .symbolic_bindings
            .insert("extent".into(), 9);
        let batch_executor = CapturedReplayExecutor::default();
        assert!(matches!(
            batch_executor.replay_batch(&decoded, &batch, options),
            Err(ReplayError::Batch { invocation: 2, .. })
        ));
        assert_eq!(batch_executor.compile_cache_len(true), 0);
        batch.invocations[2]
            .symbolic_bindings
            .insert("extent".into(), 8);
        let result = batch_executor
            .replay_batch(&decoded, &batch, options)
            .unwrap();
        assert_eq!(result.invocations.len(), 3);
        assert!(result.invocations.iter().all(|invocation| {
            invocation
                .trace
                .items
                .iter()
                .all(|item| item.backend == ItemBackend::NativeJit)
        }));
    }

    #[test]
    fn symbolic_exact_splat_constants_resize_and_vector_scalar_broadcasts_are_native() {
        let extent = crate::SymbolicExpr::variable("extent", 0, 8).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3], DType::F32);
        let constant = graph.constant(TensorData::new([3], vec![2.0, 2.0, 2.0]).unwrap());
        let output = graph.add(input, constant).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let shape = crate::SymbolicShape::new(vec![extent.clone().into()]);
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([(input, shape.clone())]))
            .with_constant_shape(constant, shape);
        let capture = CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &spec,
            &BTreeMap::from([("extent".into(), 3)]),
        )
        .unwrap();
        let decoded = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        let executor = CapturedReplayExecutor::default();
        for len in [0usize, 1, 3, 8] {
            let input = TensorData::from_scalars(
                [len],
                DType::F32,
                (0..len).map(|index| Scalar::F(index as f64)),
            )
            .unwrap();
            let result = executor
                .replay_symbolic(
                    &decoded,
                    &BTreeMap::from([("extent".into(), len as i64)]),
                    &BTreeMap::from([("input".into(), input)]),
                    CapturedReplayOptions {
                        backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                    },
                )
                .unwrap();
            assert_eq!(result.outputs[0].shape(), &Shape::from([len]));
            assert_eq!(
                result.outputs[0].to_vec_f64(),
                (0..len).map(|index| index as f64 + 2.0).collect::<Vec<_>>()
            );
        }

        let mut scalar_graph = Graph::new();
        let vector = scalar_graph.input_dtype("vector", [7], DType::F32);
        let scalar = scalar_graph.input_dtype("scalar", [1], DType::F32);
        let output = scalar_graph.add(vector, scalar).unwrap();
        let capture = captured(&scalar_graph, &[output]);
        let result = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &BTreeMap::from([
                    (
                        "vector".into(),
                        TensorData::new([7], vec![0., 1., 2., 3., 4., 5., 6.]).unwrap(),
                    ),
                    ("scalar".into(), TensorData::new([1], vec![0.5]).unwrap()),
                ]),
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(
            result.outputs[0].values(),
            &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        );
        assert_eq!(result.trace.items[0].backend, ItemBackend::NativeJit);
        assert_eq!(result.trace.items[0].lanes, 4);
        assert_eq!(result.trace.items[0].vector_main, 4);
        assert_eq!(result.trace.items[0].vector_tail, 3);

        let mut view_graph = Graph::new();
        let input = view_graph.input_dtype("input", [12], DType::F32);
        let view = view_graph.shrink(input, [(4, 11)]).unwrap();
        let output = view_graph.neg(view).unwrap();
        let capture = captured(&view_graph, &[output]);
        let result = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &BTreeMap::from([(
                    "input".into(),
                    TensorData::new([12], vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.])
                        .unwrap(),
                )]),
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(
            result.outputs[0].values(),
            &[-4., -5., -6., -7., -8., -9., -10.]
        );
        assert_eq!(result.trace.items[0].backend, ItemBackend::NativeJit);
        assert_eq!(result.trace.items[0].lanes, 4);
        assert_eq!(result.trace.items[0].vector_main, 4);
        assert_eq!(result.trace.items[0].vector_tail, 3);

        let mut malformed = Graph::new();
        let input = malformed.input_dtype("input", [3], DType::F32);
        let constant = malformed.constant(TensorData::new([3], vec![1.0, 2.0, 1.0]).unwrap());
        let output = malformed.add(input, constant).unwrap();
        let schedule = crate::schedule(&malformed, output).unwrap();
        let shape = crate::SymbolicShape::new(vec![
            crate::SymbolicExpr::variable("bad_extent", 0, 8)
                .unwrap()
                .into(),
        ]);
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([(input, shape.clone())]))
            .with_constant_shape(constant, shape);
        assert!(matches!(
            CapturedSchedule::capture_symbolic(
                &malformed,
                &schedule,
                &[output],
                &spec,
                &BTreeMap::from([("bad_extent".into(), 3)])
            ),
            Err(ReplayError::Unsupported(_))
        ));
    }

    struct SymbolicMovementFamily {
        graph: Graph,
        outputs: [crate::NodeId; 6],
        inputs: [crate::NodeId; 4],
        bindings: BTreeMap<String, TensorData>,
    }

    fn symbolic_movement_family(rows: usize) -> SymbolicMovementFamily {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [rows, 3], DType::F32);
        let tail = graph.input_dtype("tail", [rows, 1], DType::F32);
        let index = graph.input_dtype("index", [rows, 2], DType::I32);
        let updates = graph.input_dtype("updates", [rows, 2], DType::F32);
        let padded = graph.pad(input, [(0, 0), (1, 1)], Scalar::F(-1.0)).unwrap();
        let concatenated = graph.concat([input, tail], 1).unwrap();
        let gathered = graph.gather(input, index, 1).unwrap();
        let scattered = graph.scatter_add(input, index, updates, 1).unwrap();
        let contiguous_source = graph.neg(input).unwrap();
        let contiguous = graph.contiguous(contiguous_source).unwrap();
        let bitcast = graph.bitcast(input, DType::I32).unwrap();
        let input_values = (0..rows)
            .flat_map(|row| (0..3).map(move |column| (row * 10 + column) as f32))
            .collect::<Vec<_>>();
        let tail_values = (0..rows).map(|row| 50.0 + row as f32).collect::<Vec<_>>();
        let index_values = (0..rows).flat_map(|_| [1i32, 1]).collect::<Vec<_>>();
        let update_values = (0..rows)
            .flat_map(|row| [100.0 + row as f32, 200.0 + row as f32])
            .collect::<Vec<_>>();
        SymbolicMovementFamily {
            graph,
            outputs: [
                padded,
                concatenated,
                gathered,
                scattered,
                contiguous,
                bitcast,
            ],
            inputs: [input, tail, index, updates],
            bindings: BTreeMap::from([
                (
                    "input".into(),
                    TensorData::new([rows, 3], input_values).unwrap(),
                ),
                (
                    "tail".into(),
                    TensorData::new([rows, 1], tail_values).unwrap(),
                ),
                (
                    "index".into(),
                    TensorData::from_storage([rows, 2], Storage::I32(index_values)).unwrap(),
                ),
                (
                    "updates".into(),
                    TensorData::new([rows, 2], update_values).unwrap(),
                ),
            ]),
        }
    }

    #[test]
    fn symbolic_movement_plans_specialize_without_graph_reconstruction() {
        let rows = crate::SymbolicExpr::variable("rows", 0, 8).unwrap();
        let template = symbolic_movement_family(2);
        let schedule = crate::schedule_many(&template.graph, &template.outputs).unwrap();
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([
            (
                template.inputs[0],
                crate::SymbolicShape::new(vec![rows.clone().into(), 3usize.into()]),
            ),
            (
                template.inputs[1],
                crate::SymbolicShape::new(vec![rows.clone().into(), 1usize.into()]),
            ),
            (
                template.inputs[2],
                crate::SymbolicShape::new(vec![rows.clone().into(), 2usize.into()]),
            ),
            (
                template.inputs[3],
                crate::SymbolicShape::new(vec![rows.into(), 2usize.into()]),
            ),
        ]));
        let capture = CapturedSchedule::capture_symbolic(
            &template.graph,
            &schedule,
            &template.outputs,
            &spec,
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        let bytes = capture.to_bytes().unwrap();
        let capture = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, capture.to_bytes().unwrap());

        let mut malformed_plan = capture
            .items
            .iter()
            .find_map(|item| match item.kernel.operation() {
                crate::Operation::Movement(crate::MovementValue::Plan(plan)) => {
                    Some((**plan).clone())
                }
                _ => None,
            })
            .unwrap();
        malformed_plan.cache_key ^= 1;
        assert!(malformed_plan.validate().is_err());

        let mut tampered_schema = capture.clone();
        let schema = tampered_schema.symbolic.as_mut().unwrap();
        let input_buffer = template.inputs[0].index() as u64;
        let row_dimension = schema.buffer_shapes[&input_buffer].dims()[0].clone();
        schema.buffer_shapes.insert(
            input_buffer,
            crate::SymbolicShape::new(vec![row_dimension, 4usize.into()]),
        );
        tampered_schema.identity = 0;
        tampered_schema.identity = crate::schedule::artifact::identity(&tampered_schema).unwrap();
        assert!(tampered_schema.to_bytes().is_err());

        let executor = CapturedReplayExecutor::default();
        let mut identities = BTreeSet::new();
        for rows in [0usize, 1, 2, 8] {
            let oracle = symbolic_movement_family(rows);
            let symbols = BTreeMap::from([("rows".into(), rows as i64)]);
            let specialized = executor.specialize(&capture, &symbols).unwrap();
            let mut movement_items = 0;
            for item in &specialized.capture().items {
                let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
                    item.kernel.operation()
                else {
                    continue;
                };
                movement_items += 1;
                plan.validate().unwrap();
                assert_eq!(plan.output_shape, item.primary_output().shape);
                assert!(
                    plan.input_operands()
                        .iter()
                        .all(|operand| item.inputs.iter().any(|input| {
                            input.id == operand.node.index() as u64 && input.shape == operand.shape
                        }))
                );
            }
            assert_eq!(movement_items, 6);
            identities.insert(specialized.trace().concrete_identity);
            let replayed = executor
                .replay_symbolic(
                    &capture,
                    &symbols,
                    &oracle.bindings,
                    CapturedReplayOptions::default(),
                )
                .unwrap();
            let oracle_bindings = oracle.bindings.into_iter().collect::<HashMap<_, _>>();
            for (actual, output) in replayed.outputs.iter().zip(oracle.outputs) {
                let expected = CpuBackend
                    .execute(&oracle.graph, output, &oracle_bindings)
                    .unwrap();
                assert_eq!(actual.shape(), expected.shape(), "rows={rows}");
                assert_eq!(actual.storage(), expected.storage(), "rows={rows}");
            }
        }
        assert_eq!(identities.len(), 4);

        let mut invalid = symbolic_movement_family(1).bindings;
        invalid.insert(
            "index".into(),
            TensorData::from_storage([1, 2], Storage::I32(vec![3, 0])).unwrap(),
        );
        assert!(matches!(
            CapturedReplayExecutor::default().replay_symbolic(
                &capture,
                &BTreeMap::from([("rows".into(), 1)]),
                &invalid,
                CapturedReplayOptions::default(),
            ),
            Err(ReplayError::Execute(_))
        ));
    }

    struct SymbolicEmbeddingFamily {
        graph: Graph,
        output: crate::NodeId,
        index: crate::NodeId,
        bindings: BTreeMap<String, TensorData>,
    }

    fn symbolic_embedding_family(tokens: usize) -> SymbolicEmbeddingFamily {
        let mut graph = Graph::new();
        let weight = graph.input_dtype("weight", [5, 3], DType::F32);
        let index_node = graph.input_dtype("index", [tokens], DType::I32);
        let expanded = graph.reshape(index_node, [tokens, 1]).unwrap();
        let expanded = graph.expand(expanded, [tokens, 3]).unwrap();
        let gathered = graph.gather(weight, expanded, 0).unwrap();
        let output = graph.reshape(gathered, [tokens, 3]).unwrap();
        let weight = TensorData::new(
            [5, 3],
            (0..15).map(|value| value as f32).collect::<Vec<_>>(),
        )
        .unwrap();
        let index = TensorData::from_storage(
            [tokens],
            Storage::I32((0..tokens).map(|token| (token % 5) as i32).collect()),
        )
        .unwrap();
        SymbolicEmbeddingFamily {
            graph,
            output,
            index: index_node,
            bindings: BTreeMap::from([("weight".into(), weight), ("index".into(), index)]),
        }
    }

    #[test]
    fn symbolic_embedding_gather_replays_variable_token_counts() {
        let tokens = crate::SymbolicExpr::variable("tokens", 0, 8).unwrap();
        let template = symbolic_embedding_family(2);
        let schedule = crate::schedule(&template.graph, template.output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &template.graph,
            &schedule,
            &[template.output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                template.index,
                crate::SymbolicShape::new(vec![tokens.into()]),
            )])),
            &BTreeMap::from([("tokens".into(), 2)]),
        )
        .unwrap();
        let capture = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        for tokens in [0usize, 1, 2, 8] {
            let oracle = symbolic_embedding_family(tokens);
            let actual = CapturedReplayExecutor::default()
                .replay_symbolic(
                    &capture,
                    &BTreeMap::from([("tokens".into(), tokens as i64)]),
                    &oracle.bindings,
                    CapturedReplayOptions::default(),
                )
                .unwrap_or_else(|error| panic!("tokens={tokens}: {error:?}"))
                .outputs
                .remove(0);
            let expected = CpuBackend
                .execute(
                    &oracle.graph,
                    oracle.output,
                    &oracle.bindings.into_iter().collect::<HashMap<_, _>>(),
                )
                .unwrap();
            assert_eq!(actual, expected, "tokens={tokens}");
        }
    }

    #[test]
    fn symbolic_computed_affine_is_admitted_but_shape_changing_bitcast_fails_closed() {
        let rows = crate::SymbolicExpr::variable("rows", 0, 8).unwrap();

        let mut affine = Graph::new();
        let input = affine.input_dtype("input", [2, 3], DType::F32);
        let computed = affine.square(input).unwrap();
        let output = affine.permute(computed, [1, 0]).unwrap();
        let schedule = crate::schedule(&affine, output).unwrap();
        let capture = CapturedSchedule::capture_symbolic(
            &affine,
            &schedule,
            &[output],
            &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                input,
                crate::SymbolicShape::new(vec![rows.clone().into(), 3usize.into()]),
            )])),
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();
        assert_eq!(capture.symbolic.as_ref().unwrap().views.len(), 1);

        let mut bitcast = Graph::new();
        let input = bitcast.input_dtype("input", [2, 1], DType::F32);
        let output = bitcast.bitcast(input, DType::U8).unwrap();
        let schedule = crate::schedule(&bitcast, output).unwrap();
        assert!(matches!(
            CapturedSchedule::capture_symbolic(
                &bitcast,
                &schedule,
                &[output],
                &crate::SymbolicCaptureSpec::new(BTreeMap::from([(
                    input,
                    crate::SymbolicShape::new(vec![rows.into(), 1usize.into()]),
                )])),
                &BTreeMap::from([("rows".into(), 2)]),
            ),
            Err(ReplayError::Unsupported(_))
        ));
    }

    #[test]
    fn symbolic_gather_requires_an_all_domain_extent_proof() {
        let rows = crate::SymbolicExpr::variable("rows", 0, 8).unwrap();
        let mut correlated = Graph::new();
        let input = correlated.input_dtype("input", [3, 3], DType::F32);
        let index = correlated.input_dtype("index", [2, 2], DType::I32);
        let output = correlated.gather(input, index, 1).unwrap();
        let schedule = crate::schedule(&correlated, output).unwrap();
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([
            (
                input,
                crate::SymbolicShape::new(vec![
                    (rows.clone() + crate::SymbolicExpr::constant(1)).into(),
                    3usize.into(),
                ]),
            ),
            (
                index,
                crate::SymbolicShape::new(vec![rows.clone().into(), 2usize.into()]),
            ),
        ]));
        let capture = CapturedSchedule::capture_symbolic(
            &correlated,
            &schedule,
            &[output],
            &spec,
            &BTreeMap::from([("rows".into(), 2)]),
        )
        .unwrap();

        let mut tampered = capture.clone();
        tampered.symbolic.as_mut().unwrap().buffer_shapes.insert(
            input.index() as u64,
            crate::SymbolicShape::new(vec![3usize.into(), 3usize.into()]),
        );
        tampered.identity = 0;
        tampered.identity = crate::schedule::artifact::identity(&tampered).unwrap();
        assert_eq!(
            crate::schedule::artifact::identity(&tampered).unwrap(),
            tampered.identity
        );
        assert!(crate::schedule::artifact::validate_capture(&tampered).is_err());

        let mut unproven = Graph::new();
        let input = unproven.input_dtype("input", [3, 3], DType::F32);
        let index = unproven.input_dtype("index", [2, 2], DType::I32);
        let output = unproven.gather(input, index, 1).unwrap();
        let schedule = crate::schedule(&unproven, output).unwrap();
        let spec = crate::SymbolicCaptureSpec::new(BTreeMap::from([
            (
                input,
                crate::SymbolicShape::new(vec![3usize.into(), 3usize.into()]),
            ),
            (
                index,
                crate::SymbolicShape::new(vec![rows.into(), 2usize.into()]),
            ),
        ]));
        assert!(matches!(
            CapturedSchedule::capture_symbolic(
                &unproven,
                &schedule,
                &[output],
                &spec,
                &BTreeMap::from([("rows".into(), 2)]),
            ),
            Err(ReplayError::Unsupported(_))
        ));
    }
}
