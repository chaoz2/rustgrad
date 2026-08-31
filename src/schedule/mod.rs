//! A deterministic, non-mutating producer-aware schedule DAG. Pure
//! elementwise/view regions fuse into their consumers while materialization
//! roots retain stable buffer and UOp identities for realization.
use crate::{DType, Graph, NodeId, Op, Shape, UOp, UOpError};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};
pub mod artifact;
pub(crate) mod dynamic;
pub mod execution_summary;
mod identity;
pub mod mixed;
pub use execution_summary::{
    ExecutionPlanItemSummary, ExecutionPlanSummary, ExecutionPlanSummaryError,
};
pub use mixed::{
    ScheduleStateBinding, ScheduleValueBinding, bind_states as bind_schedule_states,
    combine as combine_mixed_schedules,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BufferDesc {
    pub id: u64,
    pub shape: Shape,
    pub dtype: DType,
    pub bytes: usize,
    pub alignment: usize,
    pub read_only: bool,
    pub view: Option<crate::AffineView>,
}

/// Validates the physical descriptor shared by ordinary schedules, memory
/// planning, and portable schedule artifacts. A logical view still describes
/// reads from this base descriptor, so its source shape—not its logical
/// shape—must agree with `shape`.
pub(crate) fn validate_buffer_desc(desc: &BufferDesc) -> Result<(), ScheduleError> {
    let expected_bytes = desc
        .shape
        .numel()
        .map_err(|_| ScheduleError::Overflow)?
        .checked_mul(desc.dtype.itemsize())
        .ok_or(ScheduleError::Overflow)?;
    if desc.bytes != expected_bytes {
        return Err(ScheduleError::Binding(
            "buffer descriptor byte size mismatch".into(),
        ));
    }
    if desc.alignment == 0 || !desc.alignment.is_power_of_two() {
        return Err(ScheduleError::Binding(
            "buffer descriptor alignment is invalid".into(),
        ));
    }
    if let Some(view) = &desc.view {
        view.validate_read()
            .map_err(|_| ScheduleError::Binding("buffer descriptor view is invalid".into()))?;
        if view.source_shape != desc.shape {
            return Err(ScheduleError::Binding(
                "buffer descriptor view source shape mismatch".into(),
            ));
        }
    }
    Ok(())
}
/// Immutable input-pointer order for a lowered kernel. `inputs` remains a
/// set-like inventory for dependency planning; this is the only operand/ABI
/// order and must never be reconstructed by sorting node or buffer IDs.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ScheduleInputBinding {
    pub input_node: NodeId,
    pub desc: BufferDesc,
    pub abi_index: usize,
}
/// A packed GGML input occupies one immutable pointer ABI slot but is not a
/// dense `BufferDesc` and therefore cannot acquire a fake scalar `DType`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct QuantizedScheduleInputBinding {
    pub input_node: NodeId,
    pub desc: crate::QuantizedBufferDesc,
    pub abi_index: usize,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ScheduleBoundary {
    Unsupported(&'static str),
    NonScalarUOpBridge,
    Effect,
}
/// Immutable, ordered outputs owned by one scheduled producer.
///
/// This is the sole output descriptor inventory for a scheduled producer.
/// Single-output consumers use [`ScheduleItem::primary_output`] explicitly.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ScheduledOutputs(Vec<BufferDesc>);

impl ScheduledOutputs {
    pub fn new(outputs: Vec<BufferDesc>) -> Result<Self, ScheduleError> {
        if outputs.is_empty() {
            return Err(ScheduleError::Binding("scheduled outputs are empty".into()));
        }
        let mut ids = BTreeSet::new();
        if outputs.iter().any(|output| !ids.insert(output.id)) {
            return Err(ScheduleError::Binding(
                "scheduled outputs are duplicated".into(),
            ));
        }
        Ok(Self(outputs))
    }

    pub fn single(output: BufferDesc) -> Self {
        Self(vec![output])
    }

    pub fn primary(&self) -> &BufferDesc {
        &self.0[0]
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_single(&self) -> bool {
        self.len() == 1
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &BufferDesc> {
        self.0.iter()
    }
}

#[derive(Clone, Debug)]
pub struct ScheduleItem {
    pub id: u64,
    pub node: NodeId,
    pub dependencies: Vec<u64>,
    pub consumers: Vec<u64>,
    pub inputs: Vec<BufferDesc>,
    pub input_bindings: Vec<ScheduleInputBinding>,
    pub quantized_input_bindings: Vec<QuantizedScheduleInputBinding>,
    /// Caller-owned computed buffers intentionally substituted for producer
    /// lowering in this item.
    pub external_materializations: Vec<NodeId>,
    pub outputs: ScheduledOutputs,
    pub kernel: UOp,
    pub boundary: Option<ScheduleBoundary>,
    pub cache_key: u64,
}
#[derive(Clone, Debug)]
pub struct Schedule {
    pub items: Vec<ScheduleItem>,
    /// Explicit edges from a materialized pure output to an effect STORE
    /// source. Ordinary pure schedules keep this empty.
    pub value_bindings: Vec<ScheduleValueBinding>,
    /// Explicit immutable persistent-state snapshots consumed by pure items.
    pub state_bindings: Vec<ScheduleStateBinding>,
}
impl Schedule {
    /// Validates deterministic DAG and universal effect-item invariants before
    /// a backend is allowed to inspect a kernel.
    pub fn validate(&self) -> Result<(), ScheduleError> {
        let ids = self
            .items
            .iter()
            .map(|item| item.id)
            .collect::<BTreeSet<_>>();
        if ids.len() != self.items.len()
            || ids
                .iter()
                .copied()
                .enumerate()
                .any(|(want, got)| want as u64 != got)
            || self
                .items
                .iter()
                .enumerate()
                .any(|(position, item)| item.id != position as u64)
        {
            return Err(ScheduleError::Binding(
                "schedule item IDs are not contiguous and ordered".into(),
            ));
        }
        let mut output_producers = BTreeMap::new();
        for item in &self.items {
            for output in item.outputs.iter() {
                if output_producers.insert(output.id, item.id).is_some() {
                    return Err(ScheduleError::Binding(
                        "scheduled output has multiple producers".into(),
                    ));
                }
            }
        }
        self.validate_dag_edges(&ids)?;
        for item in &self.items {
            for output in item.outputs.iter() {
                validate_buffer_desc(output)?;
            }
            for input in &item.inputs {
                validate_buffer_desc(input)?;
            }
            item.validate_input_bindings()?;
            // Effect items address persistent state buffers. An earlier
            // effect may read the initial value of a buffer that a later
            // effect overwrites under the same stable ID, so future output
            // ownership is not a pure producer relation. Effect-specific
            // ordering and value bindings are validated below.
            if !item.is_effect() {
                for binding in &item.input_bindings {
                    if let Some(producer) = output_producers.get(&binding.desc.id)
                        && (*producer >= item.id || !item.dependencies.contains(producer))
                    {
                        return Err(ScheduleError::Binding(
                            "scheduled input producer edge is absent".into(),
                        ));
                    }
                }
            }
            item.kernel.validate().map_err(ScheduleError::UOp)?;
            if item.is_effect() {
                if !item.outputs.is_single()
                    || item.boundary != Some(ScheduleBoundary::Effect)
                    || !matches!(item.kernel.operation(), crate::Operation::After(_))
                {
                    return Err(ScheduleError::Binding(
                        "invalid effect item boundary".into(),
                    ));
                }
                let store = item.kernel.sources().first().ok_or_else(|| {
                    ScheduleError::Binding("effect AFTER has no STORE source".into())
                })?;
                let (crate::Operation::After(after), crate::Operation::EffectStore(store_payload)) =
                    (item.kernel.operation(), store.operation())
                else {
                    return Err(ScheduleError::Binding("effect payload is absent".into()));
                };
                let pure_bound = self.value_bindings.iter().any(|binding| {
                    binding.effect_item == item.id
                        && binding.source_position == 0
                        && item.inputs.first() == Some(&binding.producer_output)
                });
                if !matches!(store.operation(), crate::Operation::EffectStore(_))
                    || after != store_payload
                    || item.primary_output().id != after.target.buffer
                    || (!pure_bound
                        && item.inputs.first().map(|desc| desc.id) != Some(after.source.buffer))
                {
                    return Err(ScheduleError::Binding("STORE/AFTER item mismatch".into()));
                }
            }
        }
        self.validate_value_bindings()?;
        Ok(())
    }

    /// Consumer lists are a derived, ordered mirror of dependency edges. The
    /// engine walks `items` in ID order while MemoryPlan uses `consumers` to
    /// determine lifetimes, so accepting a stale mirror would make one logical
    /// schedule have conflicting execution and allocation semantics.
    fn validate_dag_edges(&self, ids: &BTreeSet<u64>) -> Result<(), ScheduleError> {
        let mut expected_consumers = BTreeMap::<u64, Vec<u64>>::new();
        for item in &self.items {
            if item.dependencies.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(ScheduleError::Binding(
                    "schedule dependencies are not strictly ordered".into(),
                ));
            }
            for dependency in &item.dependencies {
                if !ids.contains(dependency) {
                    return Err(ScheduleError::Binding(
                        "schedule dependency is absent".into(),
                    ));
                }
                if *dependency >= item.id {
                    return Err(ScheduleError::Binding(
                        "schedule dependency is not topological".into(),
                    ));
                }
                expected_consumers
                    .entry(*dependency)
                    .or_default()
                    .push(item.id);
            }
        }
        for item in &self.items {
            let expected = expected_consumers.remove(&item.id).unwrap_or_default();
            if item.consumers != expected {
                return Err(ScheduleError::Binding(
                    "schedule consumer edges are not canonical".into(),
                ));
            }
        }
        Ok(())
    }
    /// Returns only compiler-owned outputs that can become candidates for a
    /// future allocator. Requested outputs and external identities are kept
    /// out of this list, so a planner cannot accidentally reuse them.
    pub fn internal_temporaries(&self, requested: &[NodeId]) -> Vec<BufferDesc> {
        let requested = requested
            .iter()
            .map(|node| node.index() as u64)
            .collect::<BTreeSet<_>>();
        self.items
            .iter()
            .flat_map(|item| item.outputs.iter())
            .filter(|output| !requested.contains(&output.id))
            .cloned()
            .collect()
    }

    fn validate_value_bindings(&self) -> Result<(), ScheduleError> {
        let mut targets = BTreeSet::new();
        for binding in &self.value_bindings {
            binding.validate().map_err(ScheduleError::Binding)?;
            let producer = self
                .items
                .get(binding.producer_item as usize)
                .ok_or_else(|| ScheduleError::Binding("value binding producer is absent".into()))?;
            let effect = self
                .items
                .get(binding.effect_item as usize)
                .ok_or_else(|| ScheduleError::Binding("value binding effect is absent".into()))?;
            if producer.id != binding.producer_item
                || producer.node != binding.producer_node
                || producer.primary_output() != &binding.producer_output
                || binding.abi_index != 0
            {
                return Err(ScheduleError::Binding(
                    "value binding producer identity mismatch".into(),
                ));
            }
            if !effect.is_effect()
                || binding.source_position != 0
                || effect.inputs.first() != Some(&binding.producer_output)
            {
                return Err(ScheduleError::Binding(
                    "value binding effect source mismatch".into(),
                ));
            }
            if binding.producer_item >= binding.effect_item
                || !effect.dependencies.contains(&binding.producer_item)
            {
                return Err(ScheduleError::Binding(
                    "value binding use-before-produce".into(),
                ));
            }
            if !targets.insert((binding.effect_item, binding.source_position)) {
                return Err(ScheduleError::Binding(
                    "duplicate value binding target".into(),
                ));
            }
        }
        let mut state_abis = BTreeSet::new();
        for binding in &self.state_bindings {
            binding.validate().map_err(ScheduleError::Binding)?;
            let item = self
                .items
                .get(binding.consumer_item as usize)
                .ok_or_else(|| ScheduleError::Binding("state binding consumer is absent".into()))?;
            if item.is_effect()
                || item.node != binding.consumer_node
                || !state_abis.insert((binding.consumer_item, binding.abi_index))
            {
                return Err(ScheduleError::Binding(
                    "state binding ABI identity mismatch".into(),
                ));
            }
            if item
                .input_bindings
                .get(binding.abi_index)
                .map(|input| (&input.input_node, &input.desc))
                != Some((&binding.input_node, &binding.desc))
            {
                return Err(ScheduleError::Binding(
                    "state binding input descriptor mismatch".into(),
                ));
            }
        }
        Ok(())
    }
}
/// Deterministic, conservative allocation assignment for compiler-created
/// temporaries. Callers supply only internal buffers; graph inputs, constants,
/// requested outputs and aliases are therefore never candidates for reuse.
pub use crate::memory_plan::{MemoryPlan, TemporaryAllocation};
/// Assigns allocations in buffer-ID order, reusing only an earlier compatible
/// allocation whose last use precedes the candidate's first use.
pub fn plan_temporary_reuse(
    items: &[ScheduleItem],
    temporaries: &[BufferDesc],
) -> Result<MemoryPlan, crate::MemoryPlanError> {
    MemoryPlan::from_temporaries(items, temporaries, true)
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScheduleError {
    Graph(crate::Error),
    Overflow,
    UOp(UOpError),
    Binding(String),
}
impl ScheduleItem {
    /// Canonical first descriptor for single-output execution paths.
    pub fn primary_output(&self) -> &BufferDesc {
        self.outputs.primary()
    }

    pub fn is_effect(&self) -> bool {
        matches!(
            self.kernel.operation(),
            crate::Operation::EffectStore(_) | crate::Operation::After(_)
        )
    }
    pub fn ordered_inputs(&self) -> &[ScheduleInputBinding] {
        &self.input_bindings
    }
    pub fn ordered_quantized_inputs(&self) -> &[QuantizedScheduleInputBinding] {
        &self.quantized_input_bindings
    }
    pub fn validate_input_bindings(&self) -> Result<(), ScheduleError> {
        // A visible unsupported boundary deliberately carries no lowered ABI;
        // its inventory is dependency metadata, not callable pointers.
        if self.boundary.is_some() && self.input_bindings.is_empty() {
            return Ok(());
        }
        // `inputs` is the complete leaf inventory used for dependency and
        // provenance planning, while `input_bindings` contains only buffers
        // present in the lowered callable ABI. In particular, rank-zero graph
        // constants remain in the inventory after lowering embeds their typed
        // payloads as dependency-free UOp constants. Validate every binding
        // against the inventory below without requiring equal cardinality.
        if self.boundary.is_none()
            && input_bindings(&self.kernel, &self.inputs, self.primary_output())?
                != self.input_bindings
        {
            return Err(ScheduleError::Binding(
                "bindings do not match lowered kernel resources".into(),
            ));
        }
        let mut nodes = BTreeSet::new();
        let mut buffers = BTreeSet::new();
        let mut indices = BTreeSet::new();
        for binding in &self.input_bindings {
            if self
                .outputs
                .iter()
                .any(|output| output.id == binding.desc.id)
                && !self.is_effect()
            {
                return Err(ScheduleError::Binding(
                    "output appears as input binding".into(),
                ));
            }
            if binding.input_node.index() as u64 != binding.desc.id {
                return Err(ScheduleError::Binding(
                    "binding node/descriptor mismatch".into(),
                ));
            }
            if !nodes.insert(binding.input_node.index())
                || !buffers.insert(binding.desc.id)
                || !indices.insert(binding.abi_index)
            {
                return Err(ScheduleError::Binding(
                    "duplicate input binding identity".into(),
                ));
            }
            if !self.inputs.contains(&binding.desc) {
                return Err(ScheduleError::Binding(
                    "binding descriptor absent from inventory".into(),
                ));
            }
            if let Some(view) = &binding.desc.view
                && view.source_shape != binding.desc.shape
            {
                return Err(ScheduleError::Binding(
                    "view source/logical shape mismatch".into(),
                ));
            }
        }
        for binding in &self.quantized_input_bindings {
            binding
                .desc
                .validate_metadata()
                .map_err(|error| ScheduleError::Binding(error.to_string()))?;
            if self
                .outputs
                .iter()
                .any(|output| output.id == binding.input_node.index() as u64)
                || !nodes.insert(binding.input_node.index())
                || !buffers.insert(binding.input_node.index() as u64)
                || !indices.insert(binding.abi_index)
            {
                return Err(ScheduleError::Binding(
                    "duplicate quantized input binding identity".into(),
                ));
            }
        }
        if indices
            .into_iter()
            .enumerate()
            .any(|(want, got)| want != got)
        {
            return Err(ScheduleError::Binding(
                "ABI indices are not contiguous".into(),
            ));
        }
        Ok(())
    }
}

/// Lowers graph-adjacent STORE/AFTER records into ordinary schedule items.
/// They retain the normal deterministic DAG/item/cache identity but carry an
/// explicit effect boundary, so pure renderers cannot consume them by mistake.
pub fn schedule_effects(graph: &crate::EffectGraph) -> Result<Schedule, ScheduleError> {
    let effect = crate::EffectSchedule::lower(graph)
        .map_err(|error| ScheduleError::Binding(error.to_string()))?;
    let mut items = Vec::new();
    for (position, node) in effect.nodes().iter().enumerate() {
        let payload = node.payload();
        let output_node = NodeId::from_index(
            usize::try_from(payload.target.buffer).map_err(|_| ScheduleError::Overflow)?,
        );
        let source_node = NodeId::from_index(
            usize::try_from(payload.source.buffer).map_err(|_| ScheduleError::Overflow)?,
        );
        let desc = |state: &crate::BufferState, read_only: bool| BufferDesc {
            id: state.buffer,
            shape: state.shape.clone(),
            dtype: state.dtype,
            bytes: state.bytes,
            alignment: state.dtype.itemsize().max(1),
            read_only,
            view: None,
        };
        let source = desc(&payload.source, true);
        let output = desc(&payload.target, false);
        // Effect step IDs are stable construction identities.  Resolve them
        // through this table rather than assuming they happen to equal item
        // positions.
        let dependencies = node
            .predecessors()
            .iter()
            .map(|step| {
                effect
                    .nodes()
                    .iter()
                    .position(|candidate| candidate.payload().step == *step)
                    .map(|position| position as u64)
                    .ok_or_else(|| ScheduleError::Binding("effect predecessor is absent".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut item = ScheduleItem {
            id: position as u64,
            node: output_node,
            dependencies,
            consumers: vec![],
            inputs: vec![source.clone()],
            input_bindings: vec![ScheduleInputBinding {
                input_node: source_node,
                desc: source,
                abi_index: 0,
            }],
            quantized_input_bindings: vec![],
            external_materializations: vec![],
            outputs: ScheduledOutputs::single(output),
            kernel: node.after_uop(),
            boundary: Some(ScheduleBoundary::Effect),
            cache_key: 0,
        };
        item.cache_key = item_cache_key(&item)?;
        items.push(item);
    }
    let ids = items.iter().map(|item| item.id).collect::<BTreeSet<_>>();
    if ids.len() != items.len()
        || ids
            .iter()
            .copied()
            .enumerate()
            .any(|(want, got)| want as u64 != got)
    {
        return Err(ScheduleError::Overflow);
    }
    for item in items.clone() {
        for dependency in item.dependencies {
            let producer = items
                .iter_mut()
                .find(|candidate| candidate.id == dependency)
                .ok_or_else(|| ScheduleError::Binding("effect predecessor is absent".into()))?;
            producer.consumers.push(item.id);
        }
    }
    let schedule = Schedule {
        items,
        value_bindings: vec![],
        state_bindings: vec![],
    };
    schedule.validate()?;
    Ok(schedule)
}
pub(crate) fn item_cache_key(item: &ScheduleItem) -> Result<u64, ScheduleError> {
    identity::item_key(item)
        .map_err(|error| ScheduleError::Binding(format!("schedule identity: {error}")))
}
pub(crate) fn specialized_item_cache_key(
    item: &ScheduleItem,
    source_identity: u64,
    bindings: &[(u64, i64)],
) -> Result<u64, ScheduleError> {
    identity::specialized_item_key(item, source_identity, bindings)
        .map_err(|error| ScheduleError::Binding(format!("schedule identity: {error}")))
}
pub(crate) fn state_bound_item_cache_key(
    source_key: u64,
    bindings: &[&ScheduleStateBinding],
) -> Result<u64, ScheduleError> {
    identity::state_bound_item_key(source_key, bindings)
        .map_err(|error| ScheduleError::Binding(format!("schedule identity: {error}")))
}
pub(crate) fn rekey_schedule_items(
    items: &mut [ScheduleItem],
    state_bindings: &[ScheduleStateBinding],
    specialization: Option<(u64, &[(u64, i64)])>,
) -> Result<(), ScheduleError> {
    // Derive the base key exactly once before wrapping it with the canonical
    // state-binding sidecar. A symbolic specialization is the base identity;
    // state metadata then wraps that specialized key exactly once.
    for item in items {
        item.cache_key = match specialization {
            Some((source_identity, bindings)) => {
                specialized_item_cache_key(item, source_identity, bindings)?
            }
            None => item_cache_key(item)?,
        };
        let relevant = state_bindings
            .iter()
            .filter(|binding| binding.consumer_item == item.id)
            .collect::<Vec<_>>();
        if !relevant.is_empty() {
            item.cache_key = state_bound_item_cache_key(item.cache_key, &relevant)?;
        }
    }
    Ok(())
}
fn input_bindings(
    kernel: &UOp,
    inputs: &[BufferDesc],
    output: &BufferDesc,
) -> Result<Vec<ScheduleInputBinding>, ScheduleError> {
    if let crate::Operation::PrefixScan(plan) = kernel.operation() {
        let desc = inputs
            .iter()
            .find(|desc| desc.id == plan.input.index() as u64)
            .cloned()
            .ok_or_else(|| ScheduleError::Binding("prefix scan input is absent".into()))?;
        let expected_dtype =
            crate::ir::prefix_scan_output_dtype(desc.dtype, plan.kind, plan.output);
        if desc.shape != plan.input_shape
            || !desc.read_only
            || desc.view.is_some()
            || output.shape != plan.output_shape
            || output.dtype != plan.dtype
            || expected_dtype != Some(plan.dtype)
            || output.read_only
            || output.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "prefix scan descriptor mismatch".into(),
            ));
        }
        return Ok(vec![ScheduleInputBinding {
            input_node: plan.input,
            desc,
            abi_index: 0,
        }]);
    }
    if let crate::Operation::TensorGuard(crate::TensorGuardValue {
        input,
        input_shape,
        dtype,
        ..
    }) = kernel.operation()
    {
        let desc = inputs
            .iter()
            .find(|desc| desc.id == input.index() as u64)
            .cloned()
            .ok_or_else(|| ScheduleError::Binding("tensor guard input is absent".into()))?;
        if desc.shape != *input_shape
            || desc.dtype != *dtype
            || !desc.read_only
            || desc.view.is_some()
            || output.id == desc.id
            || output.shape != *input_shape
            || output.dtype != *dtype
            || output.read_only
            || output.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "tensor guard descriptor mismatch".into(),
            ));
        }
        return Ok(vec![ScheduleInputBinding {
            input_node: *input,
            desc,
            abi_index: 0,
        }]);
    }
    if let crate::Operation::Sort(crate::SortValue {
        input,
        input_shape,
        values,
        dtype,
        ..
    }) = kernel.operation()
    {
        let desc = inputs
            .iter()
            .find(|desc| desc.id == input.index() as u64)
            .cloned()
            .ok_or_else(|| ScheduleError::Binding("sort input is absent".into()))?;
        if desc.shape != *input_shape
            || desc.dtype != *dtype
            || !desc.read_only
            || desc.view.is_some()
            || output.id != values.index() as u64
            || output.shape != *input_shape
            || output.dtype != *dtype
        {
            return Err(ScheduleError::Binding("sort descriptor mismatch".into()));
        }
        return Ok(vec![ScheduleInputBinding {
            input_node: *input,
            desc,
            abi_index: 0,
        }]);
    }
    if let crate::Operation::Movement(crate::MovementValue::QuantizedRowGather(plan)) =
        kernel.operation()
    {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        let desc = inputs
            .iter()
            .find(|desc| desc.id == plan.indices.index() as u64)
            .cloned()
            .ok_or_else(|| ScheduleError::Binding("quantized gather indices absent".into()))?;
        if desc.shape != plan.indices_shape
            || desc.dtype != plan.indices_dtype
            || !desc.read_only
            || desc.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "quantized gather indices descriptor mismatch".into(),
            ));
        }
        if output.id != plan.output.index() as u64
            || output.shape != plan.output_shape
            || output.dtype != plan.output_dtype
            || output.read_only
            || output.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "quantized gather output descriptor mismatch".into(),
            ));
        }
        return Ok(vec![ScheduleInputBinding {
            input_node: plan.indices,
            desc,
            abi_index: 0,
        }]);
    }
    if let crate::Operation::Matmul(crate::MatmulValue::Quantized(plan)) = kernel.operation() {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        let desc = inputs
            .iter()
            .find(|desc| desc.id == plan.activation.index() as u64)
            .cloned()
            .ok_or_else(|| ScheduleError::Binding("quantized activation absent".into()))?;
        if desc.shape != plan.activation_shape
            || desc.dtype != plan.activation_dtype
            || !desc.read_only
            || desc.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "quantized activation descriptor mismatch".into(),
            ));
        }
        if output.id != plan.output.index() as u64
            || output.shape != plan.output_shape
            || output.dtype != plan.output_dtype
            || output.read_only
            || output.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "quantized output descriptor mismatch".into(),
            ));
        }
        return Ok(vec![ScheduleInputBinding {
            input_node: plan.activation,
            desc,
            abi_index: 0,
        }]);
    }
    if let crate::Operation::Movement(crate::MovementValue::Plan(plan)) = kernel.operation() {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        let mut out = Vec::new();
        for operand in plan.input_operands() {
            let buffer = operand.node.index() as u64;
            if out
                .iter()
                .any(|binding: &ScheduleInputBinding| binding.desc.id == buffer)
            {
                continue;
            }
            let desc = inputs
                .iter()
                .find(|desc| desc.id == buffer)
                .cloned()
                .ok_or_else(|| {
                    ScheduleError::Binding(format!("movement input buffer {buffer} absent"))
                })?;
            if desc.shape != operand.shape
                || desc.dtype != operand.dtype
                || !desc.read_only
                || desc.view.is_some()
            {
                return Err(ScheduleError::Binding(format!(
                    "movement input buffer {buffer} descriptor mismatch"
                )));
            }
            out.push(ScheduleInputBinding {
                input_node: operand.node,
                desc,
                abi_index: out.len(),
            });
        }
        if plan.output.index() as u64 != output.id
            || plan.output_shape != output.shape
            || plan.dtype != output.dtype
        {
            return Err(ScheduleError::Binding(
                "movement output descriptor mismatch".into(),
            ));
        }
        return Ok(out);
    }
    let matmul = match kernel.operation() {
        crate::Operation::Matmul(crate::MatmulValue::Serial(plan)) => Some(plan.as_ref()),
        crate::Operation::Matmul(crate::MatmulValue::Tiled(payload)) => Some(&payload.matmul),
        crate::Operation::Matmul(crate::MatmulValue::TensorCore(payload)) => Some(&payload.matmul),
        _ => None,
    };
    if let Some(plan) = matmul {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        let mut out = Vec::new();
        for node in [plan.lhs, plan.rhs] {
            let buffer = node.index() as u64;
            if out
                .iter()
                .any(|binding: &ScheduleInputBinding| binding.desc.id == buffer)
            {
                continue;
            }
            let desc = inputs
                .iter()
                .find(|desc| desc.id == buffer)
                .cloned()
                .ok_or_else(|| {
                    ScheduleError::Binding(format!("matmul input buffer {buffer} absent"))
                })?;
            let (shape, dtype) = if node == plan.lhs {
                (&plan.lhs_shape, plan.lhs_dtype)
            } else {
                (&plan.rhs_shape, plan.rhs_dtype)
            };
            let logical_shape = desc
                .view
                .as_ref()
                .map(|view| &view.logical_shape)
                .unwrap_or(&desc.shape);
            if logical_shape != shape || desc.dtype != dtype || !desc.read_only {
                return Err(ScheduleError::Binding(format!(
                    "matmul input buffer {buffer} descriptor mismatch"
                )));
            }
            out.push(ScheduleInputBinding {
                input_node: node,
                desc,
                abi_index: out.len(),
            });
        }
        if plan.output.index() as u64 != output.id
            || plan.output_shape != output.shape
            || plan.dtype != output.dtype
            || output.read_only
            || output.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "matmul output descriptor mismatch".into(),
            ));
        }
        return Ok(out);
    }
    if let crate::Operation::Conv2d(plan) = kernel.operation() {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        let mut out = Vec::new();
        for node in [Some(plan.input), Some(plan.weight), plan.bias]
            .into_iter()
            .flatten()
        {
            let buffer = node.index() as u64;
            let desc = inputs
                .iter()
                .find(|desc| desc.id == buffer)
                .cloned()
                .ok_or_else(|| {
                    ScheduleError::Binding(format!("static conv input buffer {buffer} absent"))
                })?;
            let expected = if node == plan.input {
                &plan.input_shape
            } else if node == plan.weight {
                &plan.weight_shape
            } else {
                plan.bias_shape.as_ref().expect("validated optional bias")
            };
            if desc.shape != *expected
                || desc.dtype != crate::DType::F32
                || !desc.read_only
                || desc.view.is_some()
            {
                return Err(ScheduleError::Binding(format!(
                    "static conv input buffer {buffer} descriptor mismatch"
                )));
            }
            out.push(ScheduleInputBinding {
                input_node: node,
                desc,
                abi_index: out.len(),
            });
        }
        if output.id != plan.output.index() as u64
            || output.shape != plan.output_shape
            || output.dtype != crate::DType::F32
            || output.read_only
            || output.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "static conv output descriptor mismatch".into(),
            ));
        }
        return Ok(out);
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for node in kernel.topological().map_err(ScheduleError::UOp)? {
        if !matches!(node.operation(), crate::Operation::Load) {
            continue;
        }
        let Some(index) = node.sources().first() else {
            return Err(ScheduleError::Binding("load lacks index".into()));
        };
        let buffer = match index.operation() {
            crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
            | crate::Operation::Index(crate::IndexValue::View { buffer, .. }) => *buffer,
            _ => return Err(ScheduleError::Binding("load index lacks buffer".into())),
        };
        if buffer == output.id {
            return Err(ScheduleError::Binding("load aliases output".into()));
        }
        if seen.insert(buffer) {
            let desc = inputs
                .iter()
                .find(|desc| desc.id == buffer)
                .cloned()
                .ok_or_else(|| {
                    ScheduleError::Binding(format!("lowered input buffer {buffer} absent"))
                })?;
            out.push(ScheduleInputBinding {
                input_node: NodeId::from_index(buffer as usize),
                desc,
                abi_index: out.len(),
            });
        }
    }
    Ok(out)
}

pub(crate) fn quantized_input_bindings(
    kernel: &UOp,
) -> Result<Vec<QuantizedScheduleInputBinding>, ScheduleError> {
    if let crate::Operation::Matmul(crate::MatmulValue::Quantized(plan)) = kernel.operation() {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        return Ok(vec![QuantizedScheduleInputBinding {
            input_node: plan.weight,
            desc: plan.weight_desc.clone(),
            abi_index: 1,
        }]);
    }
    if let crate::Operation::Movement(crate::MovementValue::QuantizedRowGather(plan)) =
        kernel.operation()
    {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        return Ok(vec![QuantizedScheduleInputBinding {
            input_node: plan.weight,
            desc: plan.weight_desc.clone(),
            abi_index: 1,
        }]);
    }
    Ok(Vec::new())
}
impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "schedule error: {self:?}")
    }
}
impl std::error::Error for ScheduleError {}
fn buffer(graph: &Graph, id: NodeId, read_only: bool) -> Result<BufferDesc, ScheduleError> {
    let shape = graph.shape(id).map_err(ScheduleError::Graph)?.clone();
    let dtype = graph.dtype(id).map_err(ScheduleError::Graph)?;
    let bytes = shape
        .numel()
        .map_err(ScheduleError::Graph)?
        .checked_mul(dtype.itemsize())
        .ok_or(ScheduleError::Overflow)?;
    Ok(BufferDesc {
        id: id.index() as u64,
        shape,
        dtype,
        bytes,
        alignment: dtype.itemsize().max(1),
        read_only,
        view: None,
    })
}
fn supported(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. }
            | Op::Constant(_)
            | Op::Random { .. }
            | Op::Cast { .. }
            | Op::Bitcast { .. }
            | Op::Contiguous { .. }
            | Op::ContiguousBackward { .. }
            | Op::Detach { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Compare { .. }
            | Op::Logical { .. }
            | Op::Select { .. }
            | Op::Shrink { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
            | Op::Expand { .. }
            | Op::Stride { .. }
            | Op::Pad { .. }
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::Scatter { .. }
            | Op::Reduce { .. }
            | Op::PrefixScan { .. }
            | Op::TensorGuard { .. }
            | Op::Sort { .. }
            | Op::Matmul { .. }
            | Op::Conv2d { .. }
    )
}
/// Creates one conservative fused item for a pure elementwise output. Anything
/// else is a visible schedule boundary, never an implicit mislowering.
pub fn schedule(graph: &Graph, output: NodeId) -> Result<Schedule, ScheduleError> {
    schedule_many(graph, &[output])
}
/// Schedules requested graph outputs as a stable producer-aware DAG. Pure
/// elementwise/view chains are fused until an explicit materialization root.
pub fn schedule_many(graph: &Graph, outputs: &[NodeId]) -> Result<Schedule, ScheduleError> {
    schedule_many_with_external(graph, outputs, &BTreeSet::new())
}
/// Schedules with explicit caller-owned computed buffers. Only these named
/// nodes become lowered Load boundaries; ordinary scheduling stays unchanged.
pub fn schedule_with_external_materializations(
    graph: &Graph,
    outputs: &[NodeId],
    materialized: &[NodeId],
) -> Result<Schedule, ScheduleError> {
    let mut external = BTreeSet::new();
    for node in materialized {
        if !external.insert(node.index()) {
            return Err(ScheduleError::Binding(
                "duplicate external materialization".into(),
            ));
        }
        match graph.op(*node).map_err(ScheduleError::Graph)? {
            Op::Input { .. } | Op::Constant(_) => {
                return Err(ScheduleError::Binding(
                    "external materialization must be computed".into(),
                ));
            }
            _ => {}
        }
    }
    if outputs
        .iter()
        .any(|output| external.contains(&output.index()))
    {
        return Err(ScheduleError::Binding(
            "requested output cannot be external".into(),
        ));
    }
    // A materialized static view can be hidden behind the ordinary scheduler's
    // computed-shrink boundary, so schedule-buffer reachability is too weak
    // here. Validate against typed graph ancestry before traversal treats the
    // requested node as a leaf.
    fn reaches_external(
        graph: &Graph,
        node: NodeId,
        external: &BTreeSet<usize>,
        seen: &mut BTreeSet<usize>,
    ) -> Result<bool, ScheduleError> {
        if !seen.insert(node.index()) {
            return Ok(false);
        }
        if external.contains(&node.index()) {
            return Ok(true);
        }
        let children: Vec<NodeId> = match graph.op(node).map_err(ScheduleError::Graph)? {
            Op::Cast { input, .. }
            | Op::Bitcast { input, .. }
            | Op::Contiguous { input }
            | Op::ContiguousBackward { input }
            | Op::Detach { input }
            | Op::Unary { input, .. }
            | Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. }
            | Op::Reduce { input, .. }
            | Op::PrefixScan { input, .. }
            | Op::TensorGuard { input, .. }
            | Op::Sort { input, .. }
            | Op::Pad { input, .. } => vec![*input],
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Matmul { lhs, rhs } => vec![*lhs, *rhs],
            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            } => bias.iter().copied().chain([*input, *weight]).collect(),
            Op::Concat { inputs, .. } => inputs.clone(),
            Op::Gather { input, index, .. } => vec![*input, *index],
            Op::Scatter {
                base,
                index,
                updates,
                ..
            } => vec![*base, *index, *updates],
            Op::Logical { lhs, rhs, .. } => rhs.iter().copied().chain([*lhs]).collect(),
            Op::Select {
                condition,
                on_true,
                on_false,
            } => vec![*condition, *on_true, *on_false],
            _ => vec![],
        };
        for child in children {
            if reaches_external(graph, child, external, seen)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    for node in &external {
        let one = BTreeSet::from([*node]);
        let reachable = outputs.iter().try_fold(false, |found, output| {
            if found {
                Ok(true)
            } else {
                reaches_external(graph, *output, &one, &mut BTreeSet::new())
            }
        })?;
        if !reachable {
            return Err(ScheduleError::Binding(
                "external materialization is unreachable".into(),
            ));
        }
    }
    schedule_many_with_external(graph, outputs, &external)
}
fn schedule_many_with_external(
    graph: &Graph,
    outputs: &[NodeId],
    external: &BTreeSet<usize>,
) -> Result<Schedule, ScheduleError> {
    if outputs.is_empty() {
        return Ok(Schedule {
            items: vec![],
            value_bindings: vec![],
            state_bindings: vec![],
        });
    }
    let mut needed = BTreeSet::new();
    let mut consumers = vec![0usize; graph.node_count()];
    fn mark(
        g: &Graph,
        id: NodeId,
        needed: &mut BTreeSet<usize>,
        consumers: &mut [usize],
        external: &BTreeSet<usize>,
    ) -> Result<(), ScheduleError> {
        if !needed.insert(id.index()) {
            return Ok(());
        }
        if external.contains(&id.index()) {
            return Ok(());
        }
        let mut child = |child: NodeId| -> Result<(), ScheduleError> {
            consumers[child.index()] += 1;
            mark(g, child, needed, consumers, external)
        };
        match g.op(id).map_err(ScheduleError::Graph)? {
            Op::Cast { input, .. }
            | Op::Bitcast { input, .. }
            | Op::Contiguous { input }
            | Op::ContiguousBackward { input }
            | Op::Detach { input }
            | Op::Unary { input, .. }
            | Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. }
            | Op::Reduce { input, .. }
            | Op::PrefixScan { input, .. }
            | Op::TensorGuard { input, .. }
            | Op::Sort { input, .. }
            | Op::Pad { input, .. } => child(*input)?,
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Matmul { lhs, rhs } => {
                child(*lhs)?;
                child(*rhs)?;
            }
            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            } => {
                child(*input)?;
                child(*weight)?;
                if let Some(bias) = bias {
                    child(*bias)?;
                }
            }
            Op::Concat { inputs, .. } => {
                for input in inputs {
                    child(*input)?;
                }
            }
            Op::Gather { input, index, .. } => {
                child(*input)?;
                child(*index)?;
            }
            Op::Scatter {
                base,
                index,
                updates,
                ..
            } => {
                child(*base)?;
                child(*index)?;
                child(*updates)?;
            }
            Op::Logical { lhs, rhs, .. } => {
                child(*lhs)?;
                if let Some(rhs) = rhs {
                    child(*rhs)?;
                }
            }
            Op::Select {
                condition,
                on_true,
                on_false,
            } => {
                child(*condition)?;
                child(*on_true)?;
                child(*on_false)?;
            }
            _ => {}
        };
        Ok(())
    }
    for output in outputs {
        graph.op(*output).map_err(ScheduleError::Graph)?;
        mark(graph, *output, &mut needed, &mut consumers, external)?;
    }
    // Sort selectors are one coupled producer. Preserve the user-requested
    // node as an observable output while making its sibling available to the
    // same schedule item and its downstream consumers.
    let sort_siblings = |id: NodeId| -> Result<Option<NodeId>, ScheduleError> {
        let Op::Sort { pair, output, .. } = graph.op(id).map_err(ScheduleError::Graph)? else {
            return Ok(None);
        };
        let want = match output {
            crate::SortOutput::Values => crate::SortOutput::Indices,
            crate::SortOutput::Indices => crate::SortOutput::Values,
        };
        (0..graph.node_count())
            .map(NodeId::from_index)
            .find(|candidate| {
                matches!(
                    graph.op(*candidate),
                    Ok(Op::Sort { pair: candidate_pair, output: candidate_output, .. })
                        if candidate_pair == pair && *candidate_output == want
                )
            })
            .map(Some)
            .ok_or_else(|| ScheduleError::Binding("sort pair sibling is absent".into()))
    };
    let marked = needed.iter().copied().collect::<Vec<_>>();
    for index in marked {
        if let Some(sibling) = sort_siblings(NodeId::from_index(index))? {
            needed.insert(sibling.index());
        }
    }
    let requested: BTreeSet<usize> = outputs.iter().map(|id| id.index()).collect();
    // A matmul payload consumes materialized dense operands; computed operands
    // therefore become roots even when they have only this one consumer.
    let matmul_operands = needed
        .iter()
        .filter_map(|index| match graph.op(NodeId::from_index(*index)) {
            Ok(Op::Matmul { lhs, rhs }) => Some([lhs.index(), rhs.index()]),
            _ => None,
        })
        .flatten()
        .filter(|index| {
            !matches!(
                graph.op(NodeId::from_index(*index)),
                Ok(Op::Input { .. } | Op::Constant(_))
            )
        })
        .collect::<BTreeSet<_>>();
    // Prefix-scan payloads likewise name one dense logical input directly;
    // unlike an elementwise Store DAG they cannot reconstruct a computed
    // operand from its leaves during graph-free captured replay.
    let prefix_scan_operands = needed
        .iter()
        .filter_map(|index| match graph.op(NodeId::from_index(*index)) {
            Ok(Op::PrefixScan { input, .. }) => Some(input.index()),
            _ => None,
        })
        .filter(|index| {
            !matches!(
                graph.op(NodeId::from_index(*index)),
                Ok(Op::Input { .. } | Op::Constant(_))
            )
        })
        .collect::<BTreeSet<_>>();
    // Materializing movement kernels consume dense direct operands. Any
    // computed operand, including a source-backed view, therefore becomes its
    // own producer item instead of being silently replaced by its leaves.
    let movement_operands = needed
        .iter()
        .flat_map(|index| {
            let id = NodeId::from_index(*index);
            match graph.op(id) {
                Ok(
                    Op::Bitcast { input, .. } | Op::Contiguous { input } | Op::Pad { input, .. },
                ) => vec![input.index()],
                Ok(Op::Concat { inputs, .. }) => inputs.iter().map(|input| input.index()).collect(),
                Ok(Op::Gather { input, index, .. }) => vec![input.index(), index.index()],
                Ok(Op::Scatter {
                    base,
                    index,
                    updates,
                    ..
                }) => vec![base.index(), index.index(), updates.index()],
                _ => vec![],
            }
        })
        .filter(|index| {
            !matches!(
                graph.op(NodeId::from_index(*index)),
                Ok(Op::Input { .. } | Op::Constant(_))
            )
        })
        .collect::<BTreeSet<_>>();
    // A computed affine view is materialized as its own dense movement item.
    // Its producer must consequently be a schedule root even when the view is
    // its only consumer, so the copy has an owned input ABI.
    let computed_view_sources = needed
        .iter()
        .filter_map(|index| {
            let id = NodeId::from_index(*index);
            matches!(
                graph.op(id),
                Ok(Op::Shrink { .. }
                    | Op::Reshape { .. }
                    | Op::Permute { .. }
                    | Op::Expand { .. }
                    | Op::Stride { .. })
            )
            .then(|| crate::rangeify::computed_view(graph, id).ok())
            .flatten()
            .map(|view| view.source.index())
        })
        .collect::<BTreeSet<_>>();
    let roots: BTreeSet<usize> = needed
        .iter()
        .copied()
        .filter(|index| {
            let id = NodeId::from_index(*index);
            !external.contains(index)
                // Inputs and constants are caller/graph-owned values, not
                // scheduled producers. A requested source value is retained
                // by capture as an explicit passthrough instead of becoming
                // a fake in-place kernel whose input and output IDs alias.
                && !matches!(graph.op(id), Ok(Op::Input { .. } | Op::Constant(_)))
                && !matches!(
                    graph.op(id),
                    Ok(Op::Sort {
                        output: crate::SortOutput::Indices,
                        ..
                    })
                )
                && (requested.contains(index)
                    || matmul_operands.contains(index)
                    || prefix_scan_operands.contains(index)
                    || movement_operands.contains(index)
                    || computed_view_sources.contains(index)
                    || (consumers[*index] > 1
                        && !matches!(graph.op(id), Ok(Op::Input { .. } | Op::Constant(_))))
                    || matches!(
                        graph.op(id),
                        Ok(Op::Random { .. }
                            | Op::Reduce { .. }
                            | Op::PrefixScan { .. }
                            | Op::Sort { .. }
                            | Op::Matmul { .. }
                            | Op::Conv2d { .. }
                            | Op::Bitcast { .. }
                            | Op::Contiguous { .. }
                            | Op::Pad { .. }
                            | Op::Concat { .. }
                            | Op::Gather { .. }
                            | Op::Scatter { .. })
                    )
                    || !matches!(graph.op(id), Ok(op) if supported(op)))
        })
        .collect();
    let mut node_to_item: std::collections::BTreeMap<usize, u64> = roots
        .iter()
        .enumerate()
        .map(|(item, node)| (*node, item as u64))
        .collect();
    for (item, node) in roots.iter().copied().enumerate() {
        if let Some(sibling) = sort_siblings(NodeId::from_index(node))? {
            node_to_item.insert(sibling.index(), item as u64);
        }
    }
    fn leaves(
        g: &Graph,
        id: NodeId,
        roots: &BTreeSet<usize>,
        here: usize,
        out: &mut BTreeSet<usize>,
        boundary: &mut Option<ScheduleBoundary>,
        external: &BTreeSet<usize>,
    ) -> Result<(), ScheduleError> {
        if id.index() != here && roots.contains(&id.index()) {
            out.insert(id.index());
            return Ok(());
        }
        if external.contains(&id.index()) {
            out.insert(id.index());
            return Ok(());
        }
        let op = g.op(id).map_err(ScheduleError::Graph)?;
        if !supported(op) {
            *boundary = Some(ScheduleBoundary::Unsupported(
                "operation requires materialization",
            ));
            if id.index() != here {
                out.insert(id.index());
            }
            return Ok(());
        }
        match op {
            Op::Input { .. } | Op::Constant(_) => {
                out.insert(id.index());
            }
            Op::Random { .. } => {}
            Op::Cast { input, .. }
            | Op::Bitcast { input, .. }
            | Op::Contiguous { input }
            | Op::ContiguousBackward { input }
            | Op::Detach { input }
            | Op::Unary { input, .. }
            | Op::Reduce { input, .. }
            | Op::PrefixScan { input, .. }
            | Op::TensorGuard { input, .. }
            | Op::Sort { input, .. }
            | Op::Pad { input, .. } => leaves(g, *input, roots, here, out, boundary, external)?,
            Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. } => match crate::rangeify::static_view(g, id) {
                Ok(view) => {
                    out.insert(view.source.index());
                }
                Err(_) => match crate::rangeify::computed_view(g, id) {
                    Ok(view) => {
                        out.insert(view.source.index());
                    }
                    Err(_) => {
                        *boundary = Some(ScheduleBoundary::Unsupported(
                            "view is outside static owned affine materialization",
                        ));
                        out.insert(input.index());
                    }
                },
            },
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Matmul { lhs, rhs } => {
                leaves(g, *lhs, roots, here, out, boundary, external)?;
                leaves(g, *rhs, roots, here, out, boundary, external)?;
            }
            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            } => {
                leaves(g, *input, roots, here, out, boundary, external)?;
                leaves(g, *weight, roots, here, out, boundary, external)?;
                if let Some(bias) = bias {
                    leaves(g, *bias, roots, here, out, boundary, external)?;
                }
            }
            Op::Concat { inputs, .. } => {
                for input in inputs {
                    leaves(g, *input, roots, here, out, boundary, external)?;
                }
            }
            Op::Gather { input, index, .. } => {
                leaves(g, *input, roots, here, out, boundary, external)?;
                leaves(g, *index, roots, here, out, boundary, external)?;
            }
            Op::Scatter {
                base,
                index,
                updates,
                ..
            } => {
                leaves(g, *base, roots, here, out, boundary, external)?;
                leaves(g, *index, roots, here, out, boundary, external)?;
                leaves(g, *updates, roots, here, out, boundary, external)?;
            }
            Op::Logical { lhs, rhs, .. } => {
                leaves(g, *lhs, roots, here, out, boundary, external)?;
                if let Some(rhs) = rhs {
                    leaves(g, *rhs, roots, here, out, boundary, external)?;
                }
            }
            Op::Select {
                condition,
                on_true,
                on_false,
            } => {
                leaves(g, *condition, roots, here, out, boundary, external)?;
                leaves(g, *on_true, roots, here, out, boundary, external)?;
                leaves(g, *on_false, roots, here, out, boundary, external)?;
            }
            _ => unreachable!(),
        };
        Ok(())
    }
    let mut items = Vec::with_capacity(roots.len());
    for &index in &roots {
        let node = NodeId::from_index(index);
        let mut leaf_ids = BTreeSet::new();
        let mut boundary = None;
        leaves(
            graph,
            node,
            &roots,
            index,
            &mut leaf_ids,
            &mut boundary,
            external,
        )?;
        let materialized = leaf_ids
            .iter()
            .filter(|leaf| roots.contains(leaf))
            .copied()
            .collect::<BTreeSet<_>>();
        let materialized = materialized
            .union(external)
            .copied()
            .collect::<BTreeSet<_>>();
        let mut inputs = leaf_ids
            .into_iter()
            .map(|leaf| buffer(graph, NodeId::from_index(leaf), true))
            .collect::<Result<Vec<_>, _>>()?;
        let output = buffer(graph, node, false)?;
        let paired_output = sort_siblings(node)?
            .map(|sibling| buffer(graph, sibling, false))
            .transpose()?;
        let kernel =
            if boundary.is_none() {
                match graph.op(node).map_err(ScheduleError::Graph)? {
                    Op::Random { .. } => crate::kernel::lower_graph_random(graph, node)
                        .map_err(ScheduleError::UOp)?,
                    Op::Matmul { .. } => crate::kernel::lower_graph_matmul(graph, node)
                        .map_err(ScheduleError::UOp)?,
                    Op::Conv2d { .. } => crate::kernel::lower_graph_static_conv2d(graph, node)
                        .map_err(ScheduleError::UOp)?,
                    Op::Bitcast { .. }
                    | Op::Contiguous { .. }
                    | Op::Pad { .. }
                    | Op::Concat { .. }
                    | Op::Gather { .. }
                    | Op::Scatter { .. } => crate::kernel::lower_graph_movement(graph, node)
                        .map_err(ScheduleError::UOp)?,
                    Op::Shrink { .. }
                    | Op::Reshape { .. }
                    | Op::Permute { .. }
                    | Op::Expand { .. }
                    | Op::Stride { .. }
                        if crate::rangeify::computed_view(graph, node).is_ok() =>
                    {
                        crate::kernel::lower_graph_computed_affine_view(graph, node)
                            .map_err(ScheduleError::UOp)?
                    }
                    Op::Reduce { .. } => crate::kernel::lower_graph_reduction_with_materialized(
                        graph,
                        node,
                        &materialized,
                    )
                    .map_err(ScheduleError::UOp)?,
                    Op::PrefixScan { .. } => crate::kernel::lower_graph_prefix_scan(graph, node)
                        .map_err(ScheduleError::UOp)?,
                    Op::TensorGuard { .. } => crate::kernel::lower_graph_tensor_guard(graph, node)
                        .map_err(ScheduleError::UOp)?,
                    Op::Sort {
                        output: crate::SortOutput::Values,
                        ..
                    } => {
                        let indices = paired_output.as_ref().ok_or_else(|| {
                            ScheduleError::Binding("sort indices output is absent".into())
                        })?;
                        crate::kernel::lower_graph_sort_pair(
                            graph,
                            node,
                            NodeId::from_index(indices.id as usize),
                        )
                        .map_err(ScheduleError::UOp)?
                    }
                    _ => crate::kernel::lower_graph_elementwise_with_materialized(
                        graph,
                        node,
                        &materialized,
                    )
                    .map_err(ScheduleError::UOp)?,
                }
            } else {
                UOp::sink(vec![])
            };
        for value in kernel.topological().map_err(ScheduleError::UOp)? {
            if let crate::Operation::Index(crate::IndexValue::View { buffer, view, .. }) =
                value.operation()
                && let Some(desc) = inputs.iter_mut().find(|desc| desc.id == *buffer)
            {
                desc.view = Some(view.clone());
            }
        }
        let id = *node_to_item.get(&index).ok_or(ScheduleError::Overflow)?;
        let external_materializations = input_bindings(&kernel, &inputs, &output)?
            .iter()
            .filter(|binding| external.contains(&binding.input_node.index()))
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>();
        let input_bindings = input_bindings(&kernel, &inputs, &output)?;
        let quantized_input_bindings = quantized_input_bindings(&kernel)?;
        // Executable dependencies follow the final ordered pointer ABI, not
        // the earlier traversal inventory. This keeps lifetime/reuse edges
        // exact when a computed-base affine view rewrites a leaf into its
        // materialized producer. Unsupported boundary items have no lowered
        // ABI, so retain their diagnostic traversal dependencies instead.
        let dependencies = if boundary.is_none() {
            input_bindings
                .iter()
                .filter_map(|binding| node_to_item.get(&(binding.desc.id as usize)).copied())
                .collect::<BTreeSet<_>>()
        } else {
            inputs
                .iter()
                .filter_map(|desc| node_to_item.get(&(desc.id as usize)).copied())
                .collect::<BTreeSet<_>>()
        }
        .into_iter()
        .collect::<Vec<_>>();
        let mut item = ScheduleItem {
            id,
            node,
            dependencies,
            consumers: vec![],
            inputs,
            input_bindings,
            quantized_input_bindings,
            external_materializations,
            outputs: if let Some(indices) = paired_output {
                ScheduledOutputs::new(vec![output.clone(), indices])?
            } else {
                ScheduledOutputs::single(output.clone())
            },
            kernel,
            boundary,
            cache_key: 0,
        };
        item.cache_key = item_cache_key(&item)?;
        item.validate_input_bindings()?;
        items.push(item);
    }
    let positions: std::collections::BTreeMap<u64, usize> = items
        .iter()
        .enumerate()
        .map(|(position, item)| (item.id, position))
        .collect();
    for item in items.clone() {
        for dependency in item.dependencies {
            items[*positions.get(&dependency).ok_or(ScheduleError::Overflow)?]
                .consumers
                .push(item.id);
        }
    }
    let schedule = Schedule {
        items,
        value_bindings: vec![],
        state_bindings: vec![],
    };
    schedule.validate()?;
    Ok(schedule)
}
/* legacy single-root lowering retained below for reference during the DAG transition. */
#[allow(dead_code)]
fn schedule_single_legacy(graph: &Graph, output: NodeId) -> Result<Schedule, ScheduleError> {
    let mut leaves = BTreeSet::new();
    let mut boundary = None;
    fn walk(
        g: &Graph,
        id: NodeId,
        leaves: &mut BTreeSet<usize>,
        boundary: &mut Option<ScheduleBoundary>,
    ) -> Result<(), ScheduleError> {
        let op = g.op(id).map_err(ScheduleError::Graph)?;
        if !supported(op) {
            *boundary = Some(ScheduleBoundary::Unsupported(match op {
                Op::Reduce {
                    kind: crate::ReduceKind::Product,
                    ..
                } => "product reductions are outside sum/mean lowering",
                Op::Reduce {
                    kind: crate::ReduceKind::Min | crate::ReduceKind::Max,
                    ..
                } => "min/max reductions are outside sum/mean lowering",
                _ => "operation is outside phase-one elementwise lowering",
            }));
            leaves.insert(id.index());
            return Ok(());
        }
        match op {
            Op::Input { .. } | Op::Constant(_) => {
                leaves.insert(id.index());
            }
            Op::Cast { input, .. }
            | Op::Bitcast { input, .. }
            | Op::Contiguous { input }
            | Op::ContiguousBackward { input }
            | Op::Unary { input, .. } => walk(g, *input, leaves, boundary)?,
            Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. } => match crate::rangeify::static_view(g, id) {
                Ok(view) => {
                    leaves.insert(view.source.index());
                }
                Err(_) => {
                    *boundary = Some(ScheduleBoundary::Unsupported(
                        "view of a computed value requires materialization",
                    ));
                    leaves.insert(input.index());
                }
            },
            Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } => {
                walk(g, *lhs, leaves, boundary)?;
                walk(g, *rhs, leaves, boundary)?
            }
            Op::Logical { lhs, rhs, .. } => {
                walk(g, *lhs, leaves, boundary)?;
                if let Some(rhs) = rhs {
                    walk(g, *rhs, leaves, boundary)?
                }
            }
            Op::Select {
                condition,
                on_true,
                on_false,
            } => {
                walk(g, *condition, leaves, boundary)?;
                walk(g, *on_true, leaves, boundary)?;
                walk(g, *on_false, leaves, boundary)?
            }
            Op::Reduce { input, .. } | Op::PrefixScan { input, .. } | Op::Sort { input, .. } => {
                walk(g, *input, leaves, boundary)?
            }
            _ => unreachable!(),
        }
        Ok(())
    }
    walk(graph, output, &mut leaves, &mut boundary)?;
    let mut inputs = leaves
        .into_iter()
        .map(|i| buffer(graph, NodeId::from_index(i), true))
        .collect::<Result<Vec<_>, _>>()?;
    let out = buffer(graph, output, false)?;
    let kernel =
        if boundary.is_none() {
            match graph.op(output).map_err(ScheduleError::Graph)? {
                Op::Reduce { .. } => crate::kernel::lower_graph_reduction(graph, output)
                    .map_err(ScheduleError::UOp)?,
                Op::PrefixScan { .. } => crate::kernel::lower_graph_prefix_scan(graph, output)
                    .map_err(ScheduleError::UOp)?,
                _ => crate::kernel::lower_graph_elementwise(graph, output)
                    .map_err(ScheduleError::UOp)?,
            }
        } else {
            UOp::sink(vec![])
        };
    for node in kernel.topological().map_err(ScheduleError::UOp)? {
        if let crate::Operation::Index(crate::IndexValue::View { buffer, view, .. }) =
            node.operation()
            && let Some(desc) = inputs.iter_mut().find(|desc| desc.id == *buffer)
        {
            desc.view = Some(view.clone());
        }
    }
    let input_bindings = input_bindings(&kernel, &inputs, &out)?;
    let quantized_input_bindings = quantized_input_bindings(&kernel)?;
    let mut item = ScheduleItem {
        id: 0,
        node: output,
        dependencies: vec![],
        consumers: vec![],
        inputs,
        input_bindings,
        quantized_input_bindings,
        external_materializations: vec![],
        outputs: ScheduledOutputs::single(out),
        kernel,
        boundary,
        cache_key: 0,
    };
    item.cache_key = item_cache_key(&item)?;
    Ok(Schedule {
        items: vec![item],
        value_bindings: vec![],
        state_bindings: vec![],
    })
}
