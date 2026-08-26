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
        let specialization = self.specialize(capture, symbolic_bindings)?;
        validate_inputs(specialization.capture(), provided)?;
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

    pub fn replay_batch(
        &self,
        capture: &CapturedSchedule,
        batch: &CapturedBatch,
        options: CapturedReplayOptions,
    ) -> Result<CapturedBatchResult, ReplayError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
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
                .output
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
                .output
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
            values.insert_tensor(item.output.id, value);
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
            .output
            .shape
            .numel()
            .map_err(|e| ReplayError::Descriptor(e.to_string()))?;
        let (value, backend, native_key, cache_hit, lanes, main, tail, reason) = match planned {
            PlannedItem::PrunedZeroDomain { descriptor } => {
                values.insert_pruned(item.output.id, descriptor.clone(), item.id);
                continue;
            }
            PlannedItem::MaterializedZero => (
                TensorData::zeros_with_dtype(item.output.shape.clone(), item.output.dtype)
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
                TensorData::zeros_with_dtype(item.output.shape.clone(), item.output.dtype)
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
        values.insert_tensor(item.output.id, value);
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
        let value = interpret_item(capture, item, &values)?;
        values.insert_tensor(item.output.id, value);
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
        let Some(plan) = item.kernel.arg().quantized_row_gather_plan() else {
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
    Ok(values)
}

fn interpret_item(
    capture: &CapturedSchedule,
    item: &ScheduleItem,
    values: &ReplayValues,
) -> Result<TensorData, ReplayError> {
    if let Some(plan) = item.kernel.arg().quantized_row_gather_plan() {
        let indices = values.tensor(plan.indices.index() as u64, "quantized gather indices")?;
        let weight = capture
            .quantized_constants
            .get(&(plan.weight.index() as u64))
            .ok_or_else(|| ReplayError::Missing(plan.weight.index().to_string()))?;
        return plan
            .execute(indices, weight)
            .map_err(|error| ReplayError::Execute(error.to_string()));
    }
    if let Some(plan) = item.kernel.arg().quantized_matmul_plan() {
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
    if let crate::UArg::Movement(plan) = item.kernel.arg() {
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
    let mut bindings = KernelBindings::default();
    for binding in item.ordered_inputs() {
        let value = values.tensor(binding.desc.id, "kernel input")?.clone();
        let role = if capture.constants.contains_key(&binding.desc.id) {
            BufferRole::Constant
        } else {
            BufferRole::Input
        };
        let desc = KernelBufferDesc::concrete(
            binding.desc.id,
            role,
            binding.desc.shape.clone(),
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
    use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape, Storage, UArg};
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
    fn captured_threefry_random_is_graph_free_and_native_f32_f64_matches_oracle() {
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
    fn unsupported_native_policy_is_explicit() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2]), DType::F32);
        let output = graph.tan(x).unwrap();
        let capture = captured(&graph, &[output]);
        let values = BTreeMap::from([("x".into(), TensorData::new([2], vec![0., 1.]).unwrap())]);
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
        assert_eq!(first.outputs[0].scalar_at(0), Scalar::I(i64::MIN));
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
                schedule.items[0].kernel.kind(),
                crate::UOpKind::Matmul
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
        let Some(plan) = malformed_plan.items[0].kernel.arg().matmul_plan() else {
            panic!("matmul payload missing");
        };
        let mut plan = plan.clone();
        plan.output_shape = Shape::from([4]);
        malformed_plan.items[0].kernel = crate::UOp::new(
            crate::UOpKind::Matmul,
            Some(crate::UType::scalar(DType::F16)),
            vec![],
            UArg::Matmul(Box::new(plan)),
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
}
