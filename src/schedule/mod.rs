//! A deterministic, non-mutating producer-aware schedule DAG. Pure
//! elementwise/view regions fuse into their consumers while materialization
//! roots retain stable buffer and UOp identities for realization.
use crate::{DType, Graph, NodeId, Op, Shape, UOp, UOpError};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

/// Graph operands whose scheduled payload ABI names the exact dense NodeId.
/// These operations cannot reconstruct an intervening computed alias through
/// the ordinary scalar IndexView path.
fn op_direct_payload_operands(graph: &Graph, op: &Op) -> Result<Vec<NodeId>, ScheduleError> {
    let operands = match op {
        Op::Matmul { lhs, rhs } => vec![*lhs, *rhs],
        Op::PrefixScan { input, .. } | Op::Sort { input, .. } | Op::TensorGuard { input, .. } => {
            vec![*input]
        }
        Op::Threefry { counter, key } => vec![*counter, *key],
        Op::Conv2d {
            input,
            weight,
            bias,
            ..
        } => [Some(*input), Some(*weight), *bias]
            .into_iter()
            .flatten()
            .collect(),
        _ => Vec::new(),
    };
    operands
        .into_iter()
        .map(|node| {
            graph
                .contiguous_backward_owner(node)
                .map_err(ScheduleError::Graph)
        })
        .collect()
}
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

/// One requested logical value that is an affine read of one existing storage
/// owner. It deliberately is not a schedule item: `source` remains the sole
/// physical owner (an immutable graph source or one scheduled producer) and
/// `requested` names only the ordered logical result projected through
/// `desc.view`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RequestedPassthrough {
    pub requested: NodeId,
    pub source: NodeId,
    pub desc: BufferDesc,
}

impl RequestedPassthrough {
    pub(crate) fn validate(&self) -> Result<(), ScheduleError> {
        if self.requested == self.source
            || self.desc.id != self.source.index() as u64
            || !self.desc.read_only
        {
            return Err(ScheduleError::Binding(
                "requested passthrough ownership is invalid".into(),
            ));
        }
        validate_buffer_desc(&self.desc)?;
        let Some(view) = &self.desc.view else {
            return Err(ScheduleError::Binding(
                "requested passthrough affine view is absent".into(),
            ));
        };
        if view.source_shape != self.desc.shape {
            return Err(ScheduleError::Binding(
                "requested passthrough source shape is invalid".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_against_graph(&self, graph: &Graph) -> Result<(), ScheduleError> {
        self.validate()?;
        let rangeified = crate::rangeify::static_view(graph, self.requested)
            .or_else(|_| crate::rangeify::computed_view(graph, self.requested))
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        if rangeified.source != self.source || self.desc.view.as_ref() != Some(&rangeified.view) {
            return Err(ScheduleError::Binding(
                "requested passthrough diverges from its graph view".into(),
            ));
        }
        let expected = buffer(graph, self.source, true)?;
        let mut physical = self.desc.clone();
        physical.view = None;
        if physical != expected {
            return Err(ScheduleError::Binding(
                "requested passthrough source descriptor is invalid".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn project(
        &self,
        source: &crate::TensorData,
    ) -> Result<crate::TensorData, ScheduleError> {
        self.validate()?;
        if source.shape() != &self.desc.shape || source.dtype() != self.desc.dtype {
            return Err(ScheduleError::Binding(
                "requested passthrough source value is inconsistent".into(),
            ));
        }
        let view = self.desc.view.as_ref().expect("validated view");
        source
            .affine_read(view)
            .map_err(|error| ScheduleError::Binding(error.to_string()))
    }
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

/// Canonical consumer-local view metadata for one physical buffer ABI slot.
/// A single repeated view retains its historical descriptor identity. Dense
/// access or more than one distinct view makes the pointer slot view-free;
/// each `IndexValue::View` still owns and validates its exact address map.
pub(crate) fn common_buffer_views(nodes: &[UOp]) -> BTreeMap<u64, Option<crate::AffineView>> {
    let mut views = BTreeMap::<u64, Option<crate::AffineView>>::new();
    for node in nodes {
        let (buffer, view) = match node.operation() {
            crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. }) => (*buffer, None),
            crate::Operation::Index(crate::IndexValue::View { buffer, view, .. }) => {
                (*buffer, Some(view.clone()))
            }
            _ => continue,
        };
        views
            .entry(buffer)
            .and_modify(|common| {
                if *common != view {
                    *common = None;
                }
            })
            .or_insert(view);
    }
    views
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
    /// Sorted unique physical output IDs retained by the caller's requested
    /// graph values. This inventory survives graph-visible forward aliases so
    /// graph-independent allocation planning never treats an escaping owner
    /// as reusable scratch.
    pub requested_materializations: Vec<u64>,
    /// Zero-kernel requested aliases of one immutable or scheduled owner.
    pub requested_passthroughs: Vec<RequestedPassthrough>,
    /// Explicit edges from a materialized pure output to an effect STORE
    /// source. Ordinary pure schedules keep this empty.
    pub value_bindings: Vec<ScheduleValueBinding>,
    /// Explicit immutable persistent-state snapshots consumed by pure items.
    pub state_bindings: Vec<ScheduleStateBinding>,
}

/// Proves that an ordinary scalar kernel owns exactly the dense output ABI it
/// declares. Replay-local mixed schedules may deliberately rebind `node`, so
/// output integrity is defined by the Store graph and descriptors rather than
/// by a generic node/output identity projection.
pub(crate) fn validate_item_output_bindings(item: &ScheduleItem) -> Result<(), ScheduleError> {
    if item.boundary.is_some() || !matches!(item.kernel.operation(), crate::Operation::Sink) {
        return Ok(());
    }
    // Multi-output inventories are inspection-only envelopes and are rejected
    // by executable replay, not validated as ordinary scalar kernels.
    if !item.outputs.is_single() {
        return Ok(());
    }
    // A live/current executable schedule must always produce its declared
    // single output. Authenticated historical empty Sinks are upgraded to an
    // explicit unsupported boundary before reaching this validator.
    if item.kernel.sources().is_empty() {
        return Err(ScheduleError::Binding(
            "scheduled single-output Sink has no Store".into(),
        ));
    }
    let outputs = item
        .outputs
        .iter()
        .map(|output| (output.id, output))
        .collect::<BTreeMap<_, _>>();
    let mut stores = BTreeSet::new();
    for node in item.kernel.topological().map_err(ScheduleError::UOp)? {
        if !matches!(node.operation(), crate::Operation::Store) {
            continue;
        }
        let [index, value] = node.sources() else {
            return Err(ScheduleError::Binding(
                "scheduled Store does not have index and value sources".into(),
            ));
        };
        let crate::Operation::Index(crate::IndexValue::Buffer {
            buffer,
            elements,
            input_shape,
            output_shape,
            addressing: crate::IndexAddressing::Broadcast,
        }) = index.operation()
        else {
            return Err(ScheduleError::Binding(
                "scheduled Store target is not a dense output".into(),
            ));
        };
        let Some(output) = outputs.get(buffer) else {
            return Err(ScheduleError::Binding(
                "scheduled Store target is not a declared output".into(),
            ));
        };
        let expected_elements = output.shape.numel().map_err(|_| ScheduleError::Overflow)?;
        let expected_type = crate::UType::scalar(output.dtype);
        let exact_address = index.sources().first().is_some_and(|address| {
            matches!(
                address.operation(),
                crate::Operation::DefineGlobal(crate::AddressValue {
                    space: crate::AddressSpace::Global,
                    name,
                    element,
                }) if name == &format!("b{buffer}") && *element == expected_type
            ) && address.ty() == Some(expected_type)
        });
        if !stores.insert(*buffer)
            || *elements != expected_elements
            || input_shape != &output.shape
            || output_shape != &output.shape
            || output.read_only
            || output.view.is_some()
            || index.ty() != Some(expected_type)
            || value.ty() != Some(expected_type)
            || !exact_address
        {
            return Err(ScheduleError::Binding(
                "scheduled Store/output descriptor mismatch".into(),
            ));
        }
    }
    if stores.len() != outputs.len() {
        return Err(ScheduleError::Binding(
            "scheduled Stores and declared outputs are not bijective".into(),
        ));
    }
    Ok(())
}

pub(crate) fn physical_requested_materializations(
    items: &[ScheduleItem],
    passthroughs: &[RequestedPassthrough],
    requested: impl IntoIterator<Item = u64>,
) -> Vec<u64> {
    let produced = items
        .iter()
        .flat_map(|item| item.outputs.iter().map(|output| output.id))
        .collect::<BTreeSet<_>>();
    requested
        .into_iter()
        .filter(|buffer| produced.contains(buffer))
        .chain(
            passthroughs
                .iter()
                .map(|passthrough| passthrough.desc.id)
                .filter(|buffer| produced.contains(buffer)),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
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
        let mut output_descs = BTreeMap::new();
        for item in &self.items {
            for output in item.outputs.iter() {
                if output_producers.insert(output.id, item.id).is_some() {
                    return Err(ScheduleError::Binding(
                        "scheduled output has multiple producers".into(),
                    ));
                }
                output_descs.insert(output.id, output);
            }
        }
        let mut passthrough_ids = BTreeSet::new();
        for passthrough in &self.requested_passthroughs {
            passthrough.validate()?;
            let requested = passthrough.requested.index() as u64;
            if !passthrough_ids.insert(requested) || output_producers.contains_key(&requested) {
                return Err(ScheduleError::Binding(
                    "requested passthrough has conflicting ownership".into(),
                ));
            }
            if let Some(output) = output_descs.get(&passthrough.desc.id) {
                let mut physical = passthrough.desc.clone();
                physical.view = None;
                physical.read_only = false;
                if *output != &physical {
                    return Err(ScheduleError::Binding(
                        "requested passthrough producer descriptor is inconsistent".into(),
                    ));
                }
            }
        }
        if self
            .requested_materializations
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || self
                .requested_materializations
                .iter()
                .any(|buffer| !output_producers.contains_key(buffer))
        {
            return Err(ScheduleError::Binding(
                "requested materialization inventory is not canonical".into(),
            ));
        }
        let requested_materializations = self
            .requested_materializations
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if self.requested_passthroughs.iter().any(|passthrough| {
            output_producers.contains_key(&passthrough.desc.id)
                && !requested_materializations.contains(&passthrough.desc.id)
        }) {
            return Err(ScheduleError::Binding(
                "requested passthrough owner is not retained".into(),
            ));
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
            let mut external_inputs = BTreeSet::new();
            for external in &item.external_materializations {
                let buffer = external.index() as u64;
                if !external_inputs.insert(buffer)
                    || output_producers.contains_key(&buffer)
                    || !item
                        .input_bindings
                        .iter()
                        .any(|binding| binding.input_node == *external)
                {
                    return Err(ScheduleError::Binding(
                        "external materialization provenance is invalid".into(),
                    ));
                }
            }
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
            validate_item_output_bindings(item)?;
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
    /// future allocator. The schedule's canonical requested-materialization
    /// inventory, legacy caller IDs, external identities, and every output of
    /// a consumerless terminal item are kept out of this list, so a planner
    /// cannot accidentally reuse a value that escapes the schedule. Terminal
    /// protection remains a conservative defense for manually constructed
    /// schedules; the inventory protects requested owners that also have
    /// consumers without requiring this graph-independent boundary to recover
    /// graph-visible aliases.
    pub fn internal_temporaries(&self, requested: &[NodeId]) -> Vec<BufferDesc> {
        let mut requested = requested
            .iter()
            .map(|node| node.index() as u64)
            .collect::<BTreeSet<_>>();
        requested.extend(self.requested_materializations.iter().copied());
        requested.extend(
            self.requested_passthroughs
                .iter()
                .map(|passthrough| passthrough.source.index() as u64),
        );
        self.items
            .iter()
            .filter(|item| !item.consumers.is_empty())
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
        requested_materializations: vec![],
        requested_passthroughs: vec![],
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
    if let crate::Operation::Threefry(plan) = kernel.operation() {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        let operands = plan.buffer_operands().collect::<Vec<_>>();
        let mut out = Vec::new();
        for &(node, shape, mutable) in &operands[..operands.len() - 1] {
            debug_assert!(!mutable);
            let desc = inputs
                .iter()
                .find(|desc| desc.id == node.index() as u64)
                .cloned()
                .ok_or_else(|| ScheduleError::Binding("threefry input is absent".into()))?;
            if desc.shape != *shape
                || desc.dtype != crate::DType::U64
                || !desc.read_only
                || desc.view.is_some()
            {
                return Err(ScheduleError::Binding(
                    "threefry input descriptor mismatch".into(),
                ));
            }
            out.push(ScheduleInputBinding {
                input_node: node,
                desc,
                abi_index: out.len(),
            });
        }
        let (output_node, output_shape, output_mutable) = operands[operands.len() - 1];
        if !output_mutable
            || output.id != output_node.index() as u64
            || output.shape != *output_shape
            || output.dtype != crate::DType::U64
            || output.read_only
            || output.view.is_some()
        {
            return Err(ScheduleError::Binding(
                "threefry output descriptor mismatch".into(),
            ));
        }
        return Ok(out);
    }
    if let crate::Operation::PrefixScan(plan) = kernel.operation() {
        let desc = inputs
            .iter()
            .find(|desc| desc.id == plan.input.index() as u64)
            .cloned()
            .ok_or_else(|| ScheduleError::Binding("prefix scan input is absent".into()))?;
        let expected_dtype =
            crate::ir::prefix_scan_output_dtype(desc.dtype, plan.kind, plan.output);
        if desc.shape != plan.input_shape
            || desc.dtype != plan.input_dtype
            || !desc.read_only
            || desc.view.is_some()
            || output.shape != plan.output_shape
            || output.id != plan.destination.index() as u64
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

/// Uses the existing pure renderers as the shared capability oracle for the
/// bounded cross-backend redirection. A rejected ordinary kernel retains its
/// raw Contiguous copy, so transport-only dtypes and backend-specific scalar
/// gaps never lose an already-supported materialization route.
fn portable_ordinary_kernel(kernel: &UOp) -> bool {
    if kernel.topological().is_ok_and(|nodes| {
        nodes
            .iter()
            .any(crate::projected_index::ProjectedIndexPlan::is_predicated)
    }) {
        // Predicated loads carry one backend-neutral authenticated plan. Keep
        // that plan visible in the schedule even when a particular renderer's
        // dtype or address-width capability will reject it before resources;
        // otherwise scheduling would silently replace the semantic contract
        // with a Pad materialization based on the host's backend inventory.
        return true;
    }
    let ptx = crate::PtxRenderer::new(80).and_then(|renderer| renderer.render(kernel));
    let opencl = crate::runtime::opencl::OpenClRenderer::default().render(kernel);
    let metal = crate::runtime::metal::MetalRenderer::new(
        1,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: usize::MAX,
            unified_memory: false,
            family: "portable-schedule-admission".into(),
        },
    )
    .and_then(|renderer| renderer.render(kernel));
    let webgpu = crate::runtime::webgpu::WgslRenderer::new(
        1,
        crate::runtime::webgpu::WebGpuCapabilities {
            max_buffer_size: usize::MAX,
            max_storage_buffers_per_shader_stage: u32::MAX,
            max_compute_workgroup_size_x: 1,
            max_compute_workgroups_per_dimension: u32::MAX,
            timestamp_query: false,
            shader_f16: false,
        },
    )
    .and_then(|renderer| renderer.render(kernel));
    ptx.is_ok() && opencl.is_ok() && metal.is_ok() && webgpu.is_ok()
}

/// One rehearsed Contiguous ownership rewrite. The exact loaded graph nodes
/// come from the accepted Sink, not from the pre-rewrite movement inventory;
/// this keeps descriptor/dependency discovery aligned after the producer root
/// is suppressed.
struct ContiguousRedirection {
    producer: usize,
    load_nodes: BTreeSet<usize>,
    kernel: UOp,
}

struct ScalarAliasFusion {
    removed_roots: BTreeSet<usize>,
    load_nodes: BTreeSet<usize>,
    kernel: UOp,
}

/// One rehearsed reduction plus scalar epilogue. The final normalized Sink is
/// retained because view lowering may still load a computed source that the
/// graph-only ownership probe considered inlineable.
struct ReductionEpilogueFusion {
    load_nodes: BTreeSet<usize>,
    kernel: UOp,
}

/// A reduction epilogue rehearsed against the immutable pre-fusion roots.
/// Root removal is deferred until every rehearsal has contributed its exact
/// load inventory, so selection order cannot delete another epilogue's input.
struct ReductionEpilogueRehearsal {
    reduction: usize,
    candidates: BTreeSet<usize>,
    fusion: ReductionEpilogueFusion,
}

fn rehearse_reduction_epilogue(
    graph: &Graph,
    root: usize,
    reduction: usize,
    roots: &BTreeSet<usize>,
    external: &BTreeSet<usize>,
    candidates: &BTreeSet<usize>,
) -> Option<ReductionEpilogueFusion> {
    let materialized = roots
        .iter()
        .copied()
        .filter(|candidate| *candidate != root && !candidates.contains(candidate))
        .chain(external.iter().copied())
        .collect::<BTreeSet<_>>();
    let kernel = crate::kernel::lower_graph_reduction_epilogue_with_materialized(
        graph,
        NodeId::from_index(root),
        NodeId::from_index(reduction),
        &materialized,
    )
    .ok()?;
    let kernel = crate::uop::normalize_kernel(&kernel).ok()?;
    let topology = kernel.topological().ok()?;
    let mut load_nodes = BTreeSet::new();
    for value in topology {
        if !matches!(value.operation(), crate::Operation::Load) {
            continue;
        }
        let index = value.sources().first()?;
        let loaded = match index.operation() {
            crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
            | crate::Operation::Index(crate::IndexValue::View { buffer, .. }) => *buffer,
            _ => return None,
        };
        let loaded = usize::try_from(loaded).ok()?;
        if loaded == root {
            return None;
        }
        let source = NodeId::from_index(loaded);
        if !roots.contains(&loaded)
            && !external.contains(&loaded)
            && !matches!(graph.op(source), Ok(Op::Input { .. } | Op::Constant(_)))
        {
            return None;
        }
        load_nodes.insert(loaded);
    }
    (!load_nodes.contains(&reduction)).then_some(ReductionEpilogueFusion { load_nodes, kernel })
}

#[derive(Default)]
struct ScalarAliasCandidates {
    affine_maps: BTreeMap<usize, BTreeSet<crate::AffineView>>,
    direct_roots: BTreeSet<usize>,
    affine_view_roots: BTreeMap<usize, BTreeSet<usize>>,
    projected_view_roots: BTreeMap<usize, BTreeSet<usize>>,
}

struct ScalarAliasCollector<'a> {
    graph: &'a Graph,
    output: NodeId,
    iteration_shape: &'a Shape,
    roots: &'a BTreeSet<usize>,
    external: &'a BTreeSet<usize>,
    requested: &'a BTreeSet<usize>,
    candidates: ScalarAliasCandidates,
    seen: BTreeSet<NodeId>,
}

impl ScalarAliasCollector<'_> {
    fn record_affine_view_roots(
        &mut self,
        terminal: NodeId,
        source: NodeId,
    ) -> Result<(), ScheduleError> {
        let mut cursor = terminal;
        while cursor != source {
            if self.roots.contains(&cursor.index()) {
                self.candidates
                    .affine_view_roots
                    .entry(source.index())
                    .or_default()
                    .insert(cursor.index());
            }
            cursor = match self.graph.op(cursor).map_err(ScheduleError::Graph)? {
                Op::Shrink { input, .. }
                | Op::Reshape { input, .. }
                | Op::Permute { input, .. }
                | Op::Expand { input, .. }
                | Op::Stride { input, .. } => *input,
                _ => {
                    return Err(ScheduleError::Binding(
                        "invalid computed affine path".into(),
                    ));
                }
            };
        }
        Ok(())
    }

    fn record_projected_alias_roots(
        &mut self,
        terminal: NodeId,
        source: NodeId,
    ) -> Result<(), ScheduleError> {
        let mut cursor = terminal;
        while cursor != source {
            if self.roots.contains(&cursor.index()) {
                self.candidates
                    .projected_view_roots
                    .entry(source.index())
                    .or_default()
                    .insert(cursor.index());
            }
            cursor = match self.graph.op(cursor).map_err(ScheduleError::Graph)? {
                Op::Shrink { input, .. }
                | Op::Reshape { input, .. }
                | Op::Permute { input, .. }
                | Op::Expand { input, .. }
                | Op::Stride { input, .. } => *input,
                Op::Pad { input, .. }
                    if crate::rangeify::is_constant_zero_pad(self.graph, cursor) =>
                {
                    *input
                }
                _ => {
                    return Err(ScheduleError::Binding(
                        "invalid computed projected path".into(),
                    ));
                }
            };
        }
        Ok(())
    }

    fn visit(&mut self, node: NodeId) -> Result<(), ScheduleError> {
        let mut stack = vec![node];
        while let Some(node) = stack.pop() {
            let op = self.graph.op(node).map_err(ScheduleError::Graph)?;
            let is_view = matches!(
                op,
                Op::Shrink { .. }
                    | Op::Reshape { .. }
                    | Op::Permute { .. }
                    | Op::Expand { .. }
                    | Op::Stride { .. }
            ) || crate::rangeify::is_constant_zero_pad(self.graph, node);
            // A canonical-zero Pad is semantically a guarded source read, even
            // when an affine suffix (for example Reshape) can otherwise collapse
            // only as far as the Pad root. Prefer the exact predicated projection
            // before computed-view ownership so the Pad itself is not mistaken
            // for an affine producer that would remain materialized.
            if is_view
                && !self.requested.contains(&node.index())
                && !self.external.contains(&node.index())
                && let Ok(source) =
                    crate::rangeify::predicated_source(self.graph, node, self.iteration_shape)
                && (self.roots.contains(&source.index())
                    || self.external.contains(&source.index())
                    || matches!(
                        self.graph.op(source),
                        Ok(Op::Input { .. } | Op::Constant(_))
                    ))
            {
                self.record_projected_alias_roots(node, source)?;
                continue;
            }
            if is_view
                && !self.requested.contains(&node.index())
                && !self.external.contains(&node.index())
                && let Ok(planned) = crate::rangeify::computed_view(self.graph, node)
                && self.roots.contains(&planned.source.index())
                && let Ok(view) = planned.view.expand(self.iteration_shape.clone())
            {
                self.candidates
                    .affine_maps
                    .entry(planned.source.index())
                    .or_default()
                    .insert(view);
                // `computed_view` canonicalizes the whole movement chain to its
                // ultimate producer. Retain every scheduled root on that path so
                // an accepted scalar owner removes the complete physical chain,
                // including a shared intermediate view hidden below two equivalent
                // terminal maps.
                self.record_affine_view_roots(node, planned.source)?;
                continue;
            }
            if is_view
                && self.roots.contains(&node.index())
                && !self.requested.contains(&node.index())
                && !self.external.contains(&node.index())
                && let Ok(source) =
                    crate::rangeify::projected_source(self.graph, node, self.iteration_shape)
                && (self.roots.contains(&source.index())
                    || self.external.contains(&source.index())
                    || matches!(
                        self.graph.op(source),
                        Ok(Op::Input { .. } | Op::Constant(_))
                    ))
            {
                self.record_projected_alias_roots(node, source)?;
                continue;
            }
            if node != self.output && self.roots.contains(&node.index()) {
                self.candidates.direct_roots.insert(node.index());
                continue;
            }
            let children = op.value_inputs();
            if !self.seen.insert(node) {
                continue;
            }
            stack.extend(children.into_iter().rev());
        }
        Ok(())
    }
}

impl ScalarAliasCandidates {
    fn collect(
        graph: &Graph,
        output: NodeId,
        roots: &BTreeSet<usize>,
        external: &BTreeSet<usize>,
        requested: &BTreeSet<usize>,
    ) -> Result<Self, ScheduleError> {
        // Scalar elementwise lowering iterates the output shape. Reduction
        // lowering first lowers its producer over the complete reduction
        // input domain, so alias ownership must authenticate projections
        // against that same iteration shape before rehearsal.
        let iteration_shape = match graph.op(output).map_err(ScheduleError::Graph)? {
            Op::Reduce { input, .. } => graph.shape(*input).map_err(ScheduleError::Graph)?,
            _ => graph.shape(output).map_err(ScheduleError::Graph)?,
        };
        let mut collector = ScalarAliasCollector {
            graph,
            output,
            iteration_shape,
            roots,
            external,
            requested,
            candidates: Self::default(),
            seen: BTreeSet::new(),
        };
        collector.visit(output)?;
        Ok(collector.candidates)
    }
}

fn checked_scalar_sink(
    graph: &Graph,
    output: NodeId,
    kernel: UOp,
    materialized: &BTreeSet<usize>,
    removed: &BTreeSet<usize>,
) -> Result<Option<(UOp, BTreeSet<usize>)>, ScheduleError> {
    // This is an optimization rehearsal, not the authoritative lowering of
    // the graph. An affine projection can be individually valid while making
    // one trial Index incompatible with the scalar owner's iteration domain.
    // Preserve the explicit materialization boundary in that case; the
    // ordinary fallback is lowered and validated independently below.
    let Ok(kernel) = crate::uop::normalize_kernel(&kernel) else {
        return Ok(None);
    };
    if kernel.validate().is_err() {
        return Ok(None);
    }
    let Ok(topology) = kernel.topological() else {
        return Ok(None);
    };
    let stores = topology
        .iter()
        .filter(|value| matches!(value.operation(), crate::Operation::Store))
        .collect::<Vec<_>>();
    if stores.len() != 1
        || kernel.sources().len() != 2
        || !matches!(kernel.sources()[0].operation(), crate::Operation::Store)
        || !matches!(kernel.sources()[1].operation(), crate::Operation::EndRange)
    {
        return Ok(None);
    }
    let output_desc = buffer(graph, output, false)?;
    let Some(store_index) = stores[0].sources().first() else {
        return Ok(None);
    };
    let crate::Operation::Index(crate::IndexValue::Buffer {
        buffer: store_buffer,
        elements,
        input_shape,
        output_shape,
        addressing: crate::IndexAddressing::Broadcast,
    }) = store_index.operation()
    else {
        return Ok(None);
    };
    if *store_buffer != output.index() as u64
        || *elements != output_desc.shape.numel().map_err(ScheduleError::Graph)?
        || input_shape != &output_desc.shape
        || output_shape != &output_desc.shape
    {
        return Ok(None);
    }
    let mut load_nodes = BTreeSet::new();
    for value in &topology {
        if crate::kernel::scalar_uop_may_fault(value.operation()) {
            return Ok(None);
        }
        if !matches!(value.operation(), crate::Operation::Load) {
            continue;
        }
        let Some(index) = value.sources().first() else {
            return Ok(None);
        };
        let loaded = match index.operation() {
            crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
            | crate::Operation::Index(crate::IndexValue::View { buffer, .. }) => *buffer,
            _ => return Ok(None),
        };
        let loaded = usize::try_from(loaded).map_err(|_| ScheduleError::Overflow)?;
        if removed.contains(&loaded) {
            return Ok(None);
        }
        let loaded_node = NodeId::from_index(loaded);
        let leaf = matches!(
            graph.op(loaded_node).map_err(ScheduleError::Graph)?,
            Op::Input { .. } | Op::Constant(_)
        );
        if !leaf && !materialized.contains(&loaded) {
            return Ok(None);
        }
        load_nodes.insert(loaded);
    }
    if !portable_ordinary_kernel(&kernel) {
        return Ok(None);
    }
    Ok(Some((kernel, load_nodes)))
}

fn graph_node_uses(graph: &Graph, output: NodeId, target: NodeId) -> Result<usize, ScheduleError> {
    enum Frame {
        Node(NodeId),
        Edge(NodeId),
    }

    let mut seen = BTreeSet::new();
    let mut stack = vec![Frame::Node(output)];
    let mut uses = 0usize;
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Node(node) => {
                if node == target || !seen.insert(node) {
                    continue;
                }
                let children = graph.op(node).map_err(ScheduleError::Graph)?.value_inputs();
                stack.extend(children.into_iter().rev().map(Frame::Edge));
            }
            Frame::Edge(child) if child == target => {
                uses = uses.checked_add(1).ok_or(ScheduleError::Overflow)?;
            }
            Frame::Edge(child) => stack.push(Frame::Node(child)),
        }
    }
    Ok(uses)
}

fn scalar_aliases_are_exclusive(
    graph: &Graph,
    output: NodeId,
    aliases: &BTreeSet<usize>,
    external: &BTreeSet<usize>,
    requested: &BTreeSet<usize>,
    consumers: &[usize],
) -> Result<bool, ScheduleError> {
    for alias in aliases {
        if requested.contains(alias)
            || external.contains(alias)
            || graph_node_uses(graph, output, NodeId::from_index(*alias))?
                != consumers.get(*alias).copied().unwrap_or(0)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn exclusive_affine_alias_group(
    graph: &Graph,
    output: NodeId,
    source: usize,
    candidates: &ScalarAliasCandidates,
    external: &BTreeSet<usize>,
    requested: &BTreeSet<usize>,
    consumers: &[usize],
) -> Result<Option<BTreeSet<usize>>, ScheduleError> {
    let Some(maps) = candidates.affine_maps.get(&source) else {
        return Ok(None);
    };
    if maps.len() != 1
        || candidates.direct_roots.contains(&source)
        || requested.contains(&source)
        || external.contains(&source)
        || graph_node_uses(graph, output, NodeId::from_index(source))?
            != consumers.get(source).copied().unwrap_or(0)
    {
        return Ok(None);
    }
    let mut group = BTreeSet::from([source]);
    let views = candidates
        .affine_view_roots
        .get(&source)
        .into_iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    if !scalar_aliases_are_exclusive(graph, output, &views, external, requested, consumers)? {
        return Ok(None);
    }
    group.extend(views);
    Ok(Some(group))
}

/// Returns projected alias roots owned wholly by one scalar output. Unlike an
/// affine producer fusion, the dense source remains materialized: removing
/// only these aliases lets ordinary lowering attach the checked projected
/// address to the source Load. Rehearsal below remains the authority for the
/// resulting Store and complete input ABI.
fn exclusive_projected_alias_group(
    graph: &Graph,
    output: NodeId,
    source: usize,
    candidates: &ScalarAliasCandidates,
    external: &BTreeSet<usize>,
    requested: &BTreeSet<usize>,
    consumers: &[usize],
) -> Result<Option<BTreeSet<usize>>, ScheduleError> {
    let Some(views) = candidates.projected_view_roots.get(&source) else {
        return Ok(None);
    };
    if !scalar_aliases_are_exclusive(graph, output, views, external, requested, consumers)? {
        return Ok(None);
    }
    Ok(Some(views.clone()))
}

fn scalar_alias_materialized(
    output: NodeId,
    roots: &BTreeSet<usize>,
    external: &BTreeSet<usize>,
    removed: &BTreeSet<usize>,
) -> BTreeSet<usize> {
    roots
        .iter()
        .copied()
        .filter(|root| *root != output.index() && !removed.contains(root))
        .chain(external.iter().copied())
        .collect()
}

fn scalar_alias_group_is_protected(
    group: &BTreeSet<usize>,
    protected: &BTreeSet<usize>,
    hard_loads: &BTreeSet<usize>,
    movement_operand_owners: &BTreeMap<usize, BTreeSet<usize>>,
) -> bool {
    group.iter().any(|candidate| {
        protected.contains(candidate)
            && (hard_loads.contains(candidate)
                || !movement_operand_owners
                    .get(candidate)
                    .is_some_and(|owners| owners.is_subset(group)))
    })
}

fn rehearse_scalar_alias_fusion(
    graph: &Graph,
    output: NodeId,
    roots: &BTreeSet<usize>,
    external: &BTreeSet<usize>,
    removed: &BTreeSet<usize>,
) -> Result<Option<(UOp, BTreeSet<usize>)>, ScheduleError> {
    let materialized = scalar_alias_materialized(output, roots, external, removed);
    let lowered = match graph.op(output).map_err(ScheduleError::Graph)? {
        Op::Reduce { .. } => {
            crate::kernel::lower_graph_reduction_with_materialized(graph, output, &materialized)
        }
        _ => crate::kernel::lower_graph_elementwise_with_owned_aliases(
            graph,
            output,
            &materialized,
            removed,
        ),
    };
    let kernel = match lowered {
        Ok(kernel) => kernel,
        Err(_) => return Ok(None),
    };
    checked_scalar_sink(graph, output, kernel, &materialized, removed)
}

fn scalar_alias_output(op: &Op) -> bool {
    matches!(
        op,
        Op::Cast { .. }
            | Op::Detach { .. }
            | Op::ContiguousBackward { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Compare { .. }
            | Op::Logical { .. }
            | Op::Select { .. }
            | Op::Reduce { .. }
    )
}

struct ScalarAliasOwnership<'a> {
    roots: &'a BTreeSet<usize>,
    external: &'a BTreeSet<usize>,
    requested: &'a BTreeSet<usize>,
    consumers: &'a [usize],
    protected: &'a BTreeSet<usize>,
    hard_loads: &'a BTreeSet<usize>,
    movement_operand_owners: &'a BTreeMap<usize, BTreeSet<usize>>,
}

fn checked_scalar_alias_fusion(
    graph: &Graph,
    output: NodeId,
    ownership: &ScalarAliasOwnership<'_>,
) -> Result<Option<ScalarAliasFusion>, ScheduleError> {
    if !scalar_alias_output(graph.op(output).map_err(ScheduleError::Graph)?) {
        return Ok(None);
    }
    let candidates = ScalarAliasCandidates::collect(
        graph,
        output,
        ownership.roots,
        ownership.external,
        ownership.requested,
    )?;
    let mut accepted = BTreeSet::new();
    for source in candidates.affine_maps.keys().copied() {
        let Some(group) = exclusive_affine_alias_group(
            graph,
            output,
            source,
            &candidates,
            ownership.external,
            ownership.requested,
            ownership.consumers,
        )?
        else {
            continue;
        };
        if !group.is_disjoint(ownership.protected) {
            continue;
        }
        if rehearse_scalar_alias_fusion(graph, output, ownership.roots, ownership.external, &group)?
            .is_some()
        {
            accepted.extend(group);
        }
    }
    for source in candidates.projected_view_roots.keys().copied() {
        let Some(group) = exclusive_projected_alias_group(
            graph,
            output,
            source,
            &candidates,
            ownership.external,
            ownership.requested,
            ownership.consumers,
        )?
        else {
            continue;
        };
        if scalar_alias_group_is_protected(
            &group,
            ownership.protected,
            ownership.hard_loads,
            ownership.movement_operand_owners,
        ) {
            continue;
        }
        if rehearse_scalar_alias_fusion(graph, output, ownership.roots, ownership.external, &group)?
            .is_some()
        {
            accepted.extend(group);
        }
    }
    if accepted.is_empty() {
        return Ok(None);
    }
    let Some((kernel, load_nodes)) = rehearse_scalar_alias_fusion(
        graph,
        output,
        ownership.roots,
        ownership.external,
        &accepted,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(ScalarAliasFusion {
        removed_roots: accepted,
        load_nodes,
        kernel,
    }))
}

fn ordinary_scalar_fallback_loads(
    graph: &Graph,
    output: NodeId,
    roots: &BTreeSet<usize>,
    external: &BTreeSet<usize>,
) -> Result<BTreeSet<usize>, ScheduleError> {
    if !scalar_alias_output(graph.op(output).map_err(ScheduleError::Graph)?) {
        return Ok(BTreeSet::new());
    }
    let materialized = scalar_alias_materialized(output, roots, external, &BTreeSet::new());
    let kernel = match graph.op(output).map_err(ScheduleError::Graph)? {
        Op::Reduce { .. } => {
            crate::kernel::lower_graph_reduction_with_materialized(graph, output, &materialized)
        }
        _ => crate::kernel::lower_graph_elementwise_with_materialized(graph, output, &materialized),
    }
    .map_err(ScheduleError::UOp)?;
    let kernel = crate::uop::normalize_kernel(&kernel).map_err(ScheduleError::UOp)?;
    let topology = kernel.topological().map_err(ScheduleError::UOp)?;
    let mut loads = BTreeSet::new();
    for value in topology {
        if !matches!(value.operation(), crate::Operation::Load) {
            continue;
        }
        let index = value
            .sources()
            .first()
            .ok_or_else(|| ScheduleError::Binding("ordinary scalar Load index is absent".into()))?;
        let buffer = match index.operation() {
            crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
            | crate::Operation::Index(crate::IndexValue::View { buffer, .. }) => *buffer,
            _ => {
                return Err(ScheduleError::Binding(
                    "ordinary scalar Load index is not a buffer descriptor".into(),
                ));
            }
        };
        loads.insert(usize::try_from(buffer).map_err(|_| ScheduleError::Overflow)?);
    }
    Ok(loads)
}

/// Returns the fully rehearsed scalar kernel when one dense Contiguous output
/// may own its sole-use producer's computation. `None` is the conservative
/// explicit-copy fallback; malformed canonical plans still reject scheduling.
fn checked_contiguous_redirection(
    graph: &Graph,
    output: NodeId,
    roots: &BTreeSet<usize>,
    external: &BTreeSet<usize>,
    requested: &BTreeSet<usize>,
    consumers: &[usize],
) -> Result<Option<ContiguousRedirection>, ScheduleError> {
    let plan = crate::MovementKernelPlan::from_scheduled_graph(graph, output)
        .map_err(|error| ScheduleError::Binding(error.to_string()))?;
    let (producer, iteration_view) = match &plan.kind {
        crate::MovementKernelKind::Contiguous { input } => (input.node, None),
        crate::MovementKernelKind::AffineCopy { input, view } => (input.node, Some(view)),
        _ => return Ok(None),
    };
    if !roots.contains(&producer.index())
        || requested.contains(&producer.index())
        || external.contains(&producer.index())
        || consumers.get(producer.index()) != Some(&1)
        || matches!(
            graph.op(producer).map_err(ScheduleError::Graph)?,
            Op::ContiguousBackward { .. }
        )
        || crate::kernel::single_reduction_epilogue(graph, producer)
            .map_err(ScheduleError::UOp)?
            .is_some()
    {
        return Ok(None);
    }

    // Every movement node between the producer and Contiguous must belong
    // exclusively to this boundary. A requested, external, or shared view is
    // independently observable and retains the producer + AffineCopy route.
    let Op::Contiguous { input: movement } = graph.op(output).map_err(ScheduleError::Graph)? else {
        return Ok(None);
    };
    let mut cursor = *movement;
    while cursor != producer {
        if requested.contains(&cursor.index())
            || external.contains(&cursor.index())
            || consumers.get(cursor.index()) != Some(&1)
        {
            return Ok(None);
        }
        cursor = match graph.op(cursor).map_err(ScheduleError::Graph)? {
            Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. } => *input,
            _ => return Ok(None),
        };
    }

    let producer_desc = buffer(graph, producer, false)?;
    let output_desc = buffer(graph, output, false)?;
    if producer_desc.dtype != output_desc.dtype
        || producer_desc.alignment != output_desc.alignment
        || producer_desc.view.is_some()
        || output_desc.view.is_some()
        || (iteration_view.is_none()
            && (producer_desc.shape != output_desc.shape
                || producer_desc.bytes != output_desc.bytes))
    {
        return Ok(None);
    }

    let mut materialized = roots.clone();
    materialized.remove(&producer.index());
    materialized.extend(external.iter().copied());
    let lowered = match iteration_view {
        Some(view) => crate::kernel::lower_graph_elementwise_affine_into_with_materialized(
            graph,
            producer,
            output,
            view,
            &materialized,
        ),
        None => crate::kernel::lower_graph_elementwise_into_with_materialized(
            graph,
            producer,
            output,
            &materialized,
        ),
    };
    let Ok(kernel) = lowered else {
        return Ok(None);
    };
    // Redirection is optional. If its rehearsed scalar index domain is not
    // valid, retain the producer plus movement item; their normal lowering
    // remains the authoritative validation path.
    let Ok(kernel) = crate::uop::normalize_kernel(&kernel) else {
        return Ok(None);
    };
    if kernel.validate().is_err() {
        return Ok(None);
    }
    let Ok(topology) = kernel.topological() else {
        return Ok(None);
    };
    let stores = topology
        .iter()
        .filter(|value| matches!(value.operation(), crate::Operation::Store))
        .collect::<Vec<_>>();
    if stores.len() != 1
        || kernel.sources().len() != 2
        || !matches!(kernel.sources()[0].operation(), crate::Operation::Store)
        || !matches!(kernel.sources()[1].operation(), crate::Operation::EndRange)
    {
        return Ok(None);
    }
    let Some(store_index) = stores[0].sources().first() else {
        return Ok(None);
    };
    let crate::Operation::Index(crate::IndexValue::Buffer {
        buffer: store_buffer,
        elements,
        input_shape,
        output_shape,
        addressing: crate::IndexAddressing::Broadcast,
    }) = store_index.operation()
    else {
        return Ok(None);
    };
    if *store_buffer != output.index() as u64
        || *elements != output_desc.shape.numel().map_err(ScheduleError::Graph)?
        || input_shape != &output_desc.shape
        || output_shape != &output_desc.shape
    {
        return Ok(None);
    }
    let mut load_nodes = BTreeSet::new();
    for value in &topology {
        if crate::kernel::scalar_uop_may_fault(value.operation()) {
            return Ok(None);
        }
        if !matches!(value.operation(), crate::Operation::Load) {
            continue;
        }
        let Some(index) = value.sources().first() else {
            return Ok(None);
        };
        let loaded = match index.operation() {
            crate::Operation::Index(crate::IndexValue::Buffer {
                buffer,
                input_shape,
                ..
            }) if iteration_view.is_none()
                || input_shape.numel().map_err(ScheduleError::Graph)? == 1 =>
            {
                *buffer
            }
            crate::Operation::Index(crate::IndexValue::View { buffer, view, .. })
                if iteration_view.is_some() =>
            {
                let loaded = usize::try_from(*buffer).map_err(|_| ScheduleError::Overflow)?;
                let leaf_shape = graph
                    .shape(NodeId::from_index(loaded))
                    .map_err(ScheduleError::Graph)?
                    .clone();
                let Ok(expected) = crate::rangeify::computed_broadcast_view(
                    graph, *movement, producer, leaf_shape,
                ) else {
                    return Ok(None);
                };
                if &expected != view {
                    return Ok(None);
                }
                *buffer
            }
            _ => return Ok(None),
        };
        if loaded == producer.index() as u64 || loaded == output.index() as u64 {
            return Ok(None);
        }
        let loaded = usize::try_from(loaded).map_err(|_| ScheduleError::Overflow)?;
        let loaded_node = NodeId::from_index(loaded);
        let is_leaf = matches!(
            graph.op(loaded_node).map_err(ScheduleError::Graph)?,
            Op::Input { .. } | Op::Constant(_)
        );
        if !is_leaf && !materialized.contains(&loaded) {
            return Ok(None);
        }
        load_nodes.insert(loaded);
    }
    if !portable_ordinary_kernel(&kernel) {
        return Ok(None);
    }
    Ok(Some(ContiguousRedirection {
        producer: producer.index(),
        load_nodes,
        kernel,
    }))
}

fn supported(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. }
            | Op::Constant(_)
            | Op::Random { .. }
            | Op::Threefry { .. }
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
            | Op::ShapeIota { .. }
            | Op::Shrink { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
            | Op::Expand { .. }
            | Op::Stride { .. }
            | Op::Pad { .. }
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::Scatter { .. }
            | Op::ScatterPositions { .. }
            | Op::ScatterPositionsVjp { .. }
            | Op::Reduce { .. }
            | Op::PrefixScan { .. }
            | Op::TensorGuard { .. }
            | Op::Sort { .. }
            | Op::Matmul { .. }
            | Op::Conv2d { .. }
    )
}

struct LeafTraversal<'a> {
    graph: &'a Graph,
    roots: &'a BTreeSet<usize>,
    owner: usize,
    external: &'a BTreeSet<usize>,
    allow_projected: bool,
}

impl LeafTraversal<'_> {
    fn visit_one(
        &self,
        id: NodeId,
        out: &mut BTreeSet<usize>,
        boundary: &mut Option<ScheduleBoundary>,
        pending: &mut Vec<NodeId>,
    ) -> Result<(), ScheduleError> {
        if id.index() != self.owner && self.roots.contains(&id.index()) {
            out.insert(id.index());
            return Ok(());
        }
        if self.external.contains(&id.index()) {
            out.insert(id.index());
            return Ok(());
        }
        let op = self.graph.op(id).map_err(ScheduleError::Graph)?;
        if !supported(op) {
            *boundary = Some(ScheduleBoundary::Unsupported(
                "operation requires materialization",
            ));
            if id.index() != self.owner {
                out.insert(id.index());
            }
            return Ok(());
        }
        match op {
            Op::Input { .. } | Op::Constant(_) => {
                out.insert(id.index());
            }
            Op::Random { .. } | Op::ShapeIota { .. } => {}
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
            | Op::Pad { input, .. }
            | Op::ScatterPositions { input, .. } => pending.push(*input),
            Op::ScatterPositionsVjp { cotangent, .. } => pending.push(*cotangent),
            Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. } => {
                match crate::rangeify::static_view(self.graph, id) {
                    Ok(view) => {
                        out.insert(view.source.index());
                    }
                    Err(_) => match crate::rangeify::computed_view(self.graph, id) {
                        Ok(view) => {
                            out.insert(view.source.index());
                        }
                        Err(_) if self.allow_projected => {
                            match self.graph.shape(id).map_err(ScheduleError::Graph).and_then(
                                |shape| {
                                    crate::rangeify::projected_source(self.graph, id, shape)
                                        .map_err(|_| {
                                            ScheduleError::Binding(
                                                "view is outside static owned index projection"
                                                    .into(),
                                            )
                                        })
                                },
                            ) {
                                Ok(source) => {
                                    out.insert(source.index());
                                }
                                Err(_) => {
                                    *boundary = Some(ScheduleBoundary::Unsupported(
                                        "view is outside static owned index projection",
                                    ));
                                    out.insert(input.index());
                                }
                            }
                        }
                        Err(_) => {
                            *boundary = Some(ScheduleBoundary::Unsupported(
                                "projected indexing is outside symbolic capture",
                            ));
                            // Preserve a complete producer inventory even though
                            // symbolic capture rejects the projected address. The
                            // unsupported boundary must not manufacture a binding
                            // to an unscheduled intermediate movement node.
                            let source = self
                                .graph
                                .shape(id)
                                .ok()
                                .and_then(|shape| {
                                    crate::rangeify::projected_source(self.graph, id, shape).ok()
                                })
                                .unwrap_or(*input);
                            out.insert(source.index());
                        }
                    },
                }
            }
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Threefry {
                counter: lhs,
                key: rhs,
            }
            | Op::Matmul { lhs, rhs } => {
                pending.push(*rhs);
                pending.push(*lhs);
            }
            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            } => {
                if let Some(bias) = bias {
                    pending.push(*bias);
                }
                pending.push(*weight);
                pending.push(*input);
            }
            Op::Concat { inputs, .. } => {
                pending.extend(inputs.iter().rev().copied());
            }
            Op::Gather { input, index, .. } => {
                pending.push(*index);
                pending.push(*input);
            }
            Op::Scatter {
                base,
                index,
                updates,
                ..
            } => {
                pending.push(*updates);
                pending.push(*index);
                pending.push(*base);
            }
            Op::Logical { lhs, rhs, .. } => {
                if let Some(rhs) = rhs {
                    pending.push(*rhs);
                }
                pending.push(*lhs);
            }
            Op::Select {
                condition,
                on_true,
                on_false,
            } => {
                pending.push(*on_false);
                pending.push(*on_true);
                pending.push(*condition);
            }
            _ => unreachable!(),
        };
        Ok(())
    }

    fn visit(
        &self,
        id: NodeId,
        out: &mut BTreeSet<usize>,
        boundary: &mut Option<ScheduleBoundary>,
    ) -> Result<(), ScheduleError> {
        let mut pending = vec![id];
        while let Some(node) = pending.pop() {
            self.visit_one(node, out, boundary, &mut pending)?;
        }
        Ok(())
    }
}

/// Creates one conservative fused item for a pure elementwise output. Anything
/// else is a visible schedule boundary, never an implicit mislowering.
pub fn schedule(graph: &Graph, output: NodeId) -> Result<Schedule, ScheduleError> {
    schedule_many(graph, &[output])
}
/// Schedules requested graph outputs as a stable producer-aware DAG. Pure
/// elementwise/view chains are fused until an explicit materialization root.
pub fn schedule_many(graph: &Graph, outputs: &[NodeId]) -> Result<Schedule, ScheduleError> {
    schedule_many_with_external(graph, outputs, &BTreeSet::new(), SchedulePolicy::ORDINARY)
}

/// Symbolic families retain explicit computed-affine producer and movement
/// boundaries; a terminal requested view remains a separately authenticated
/// zero-kernel alias of that producer.
pub(crate) fn schedule_many_for_symbolic_capture(
    graph: &Graph,
    outputs: &[NodeId],
    external: &BTreeSet<usize>,
) -> Result<Schedule, ScheduleError> {
    schedule_many_with_external(graph, outputs, external, SchedulePolicy::SYMBOLIC)
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
        let node = graph
            .contiguous_backward_owner(*node)
            .map_err(ScheduleError::Graph)?;
        if !external.insert(node.index()) {
            return Err(ScheduleError::Binding(
                "duplicate external materialization".into(),
            ));
        }
        match graph.op(node).map_err(ScheduleError::Graph)? {
            Op::Input { .. } | Op::Constant(_) => {
                return Err(ScheduleError::Binding(
                    "external materialization must be computed".into(),
                ));
            }
            _ => {}
        }
    }
    if outputs.iter().try_fold(false, |found, output| {
        graph
            .contiguous_backward_owner(*output)
            .map(|owner| found || external.contains(&owner.index()))
            .map_err(ScheduleError::Graph)
    })? {
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
        let mut pending = vec![node];
        while let Some(node) = pending.pop() {
            let node = graph
                .contiguous_backward_owner(node)
                .map_err(ScheduleError::Graph)?;
            if !seen.insert(node.index()) {
                continue;
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
                | Op::Pad { input, .. }
                | Op::ScatterPositions { input, .. } => vec![*input],
                Op::ScatterPositionsVjp { cotangent, .. } => vec![*cotangent],
                Op::Binary { lhs, rhs, .. }
                | Op::Compare { lhs, rhs, .. }
                | Op::Threefry {
                    counter: lhs,
                    key: rhs,
                }
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
            pending.extend(children.into_iter().rev());
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
    schedule_many_with_external(graph, outputs, &external, SchedulePolicy::ORDINARY)
}

#[derive(Clone, Copy)]
struct SchedulePolicy {
    redirect_contiguous: bool,
    allow_projected: bool,
}

impl SchedulePolicy {
    const ORDINARY: Self = Self {
        redirect_contiguous: true,
        allow_projected: true,
    };
    const SYMBOLIC: Self = Self {
        redirect_contiguous: false,
        allow_projected: true,
    };
}

fn mark_needed(
    graph: &Graph,
    output: NodeId,
    needed: &mut BTreeSet<usize>,
    consumers: &mut [usize],
    external: &BTreeSet<usize>,
) -> Result<(), ScheduleError> {
    enum Frame {
        Node(NodeId),
        Edge(NodeId),
    }

    let mut stack = vec![Frame::Node(output)];
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Node(node) => {
                if !needed.insert(node.index()) || external.contains(&node.index()) {
                    continue;
                }
                let op = graph.op(node).map_err(ScheduleError::Graph)?;
                let children = if supported(op) {
                    op.value_inputs()
                } else {
                    Vec::new()
                };
                stack.extend(children.into_iter().rev().map(Frame::Edge));
            }
            Frame::Edge(child) => {
                let child = graph
                    .contiguous_backward_owner(child)
                    .map_err(ScheduleError::Graph)?;
                consumers[child.index()] += 1;
                stack.push(Frame::Node(child));
            }
        }
    }
    Ok(())
}

fn schedule_many_with_external(
    graph: &Graph,
    outputs: &[NodeId],
    external: &BTreeSet<usize>,
    policy: SchedulePolicy,
) -> Result<Schedule, ScheduleError> {
    if outputs.is_empty() {
        return Ok(Schedule {
            items: vec![],
            requested_materializations: vec![],
            requested_passthroughs: vec![],
            value_bindings: vec![],
            state_bindings: vec![],
        });
    }
    let outputs = outputs
        .iter()
        .map(|node| {
            graph
                .contiguous_backward_owner(*node)
                .map_err(ScheduleError::Graph)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let external = external
        .iter()
        .map(|index| {
            graph
                .contiguous_backward_owner(NodeId::from_index(*index))
                .map(NodeId::index)
                .map_err(ScheduleError::Graph)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut needed = BTreeSet::new();
    let mut consumers = vec![0usize; graph.node_count()];
    for output in &outputs {
        graph.op(*output).map_err(ScheduleError::Graph)?;
        mark_needed(graph, *output, &mut needed, &mut consumers, &external)?;
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
    let mut requested_passthroughs = Vec::new();
    let mut requested_passthrough_ids = BTreeSet::new();
    for &requested_node in &outputs {
        if requested_passthrough_ids.contains(&requested_node.index())
            || matches!(
                graph.op(requested_node).map_err(ScheduleError::Graph)?,
                Op::Input { .. } | Op::Constant(_)
            )
        {
            continue;
        }
        let Ok(rangeified) = crate::rangeify::static_view(graph, requested_node)
            .or_else(|_| crate::rangeify::computed_view(graph, requested_node))
        else {
            continue;
        };
        if rangeified.source == requested_node {
            continue;
        }
        let requested_shape = graph.shape(requested_node).map_err(ScheduleError::Graph)?;
        let requested_dtype = graph.dtype(requested_node).map_err(ScheduleError::Graph)?;
        let source_dtype = graph
            .dtype(rangeified.source)
            .map_err(ScheduleError::Graph)?;
        if rangeified.view.logical_shape != *requested_shape || requested_dtype != source_dtype {
            return Err(ScheduleError::Binding(
                "requested passthrough graph descriptor is invalid".into(),
            ));
        }
        let mut desc = buffer(graph, rangeified.source, true)?;
        desc.view = Some(rangeified.view);
        let passthrough = RequestedPassthrough {
            requested: requested_node,
            source: rangeified.source,
            desc,
        };
        passthrough.validate_against_graph(graph)?;
        requested_passthrough_ids.insert(requested_node.index());
        requested_passthroughs.push(passthrough);
    }
    // Direct-payload operations authenticate dense operand identities in
    // their typed plan. Keep every computed operand materialized even when a
    // requested alias could otherwise publish its producer directly.
    let direct_payload_operands = needed
        .iter()
        .map(|index| {
            let op = graph
                .op(NodeId::from_index(*index))
                .map_err(ScheduleError::Graph)?;
            op_direct_payload_operands(graph, op)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .map(NodeId::index)
        .filter(|index| {
            !matches!(
                graph.op(NodeId::from_index(*index)),
                Ok(Op::Input { .. } | Op::Constant(_))
            )
        })
        .collect::<BTreeSet<_>>();
    // Materializing movement kernels name their exact pointer ABI. In
    // particular, a contiguous boundary over an affine view consumes the
    // rangeified storage source rather than first materializing the view.
    let mut movement_operand_owners = BTreeMap::<usize, BTreeSet<usize>>::new();
    for index in &needed {
        let id = NodeId::from_index(*index);
        let plan = match crate::MovementKernelPlan::from_scheduled_graph(graph, id) {
            Ok(plan) => plan,
            Err(crate::MovementPlanError::NotMovement) => continue,
            Err(error) => return Err(ScheduleError::Binding(error.to_string())),
        };
        for input in plan.input_operands() {
            if !matches!(graph.op(input.node), Ok(Op::Input { .. } | Op::Constant(_))) {
                movement_operand_owners
                    .entry(input.node.index())
                    .or_default()
                    .insert(*index);
            }
        }
    }
    let movement_operands = movement_operand_owners
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    // A computed affine view normally materializes as its own dense movement
    // item, while a terminal requested alias keeps only its physical source.
    // Either way that source must remain a schedule root when the view is its
    // only consumer, so the copy or final projection has an owned input ABI.
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
            .then(|| {
                crate::rangeify::computed_view(graph, id)
                    .map(|view| view.source)
                    .or_else(|_| {
                        let shape = graph
                            .shape(id)
                            .map_err(|_| crate::rangeify::RangeifyError::Invalid)?;
                        crate::rangeify::projected_source(graph, id, shape)
                    })
                    .ok()
            })
            .flatten()
            .map(|source| source.index())
        })
        .collect::<BTreeSet<_>>();
    // A computed affine alias of a caller-owned materialization does not own
    // another physical buffer merely because one scalar root reads it more
    // than once. Keep the named external producer as the exact ABI boundary;
    // downstream kernels reconstruct the alias with one IndexView. Requested
    // aliases remain roots because their dense value is independently
    // observable.
    let external_view_aliases = needed
        .iter()
        .filter(|index| !requested.contains(index))
        // Specialized payloads name their operands directly rather than
        // consuming scalar IndexView nodes. Preserve those exact operand
        // roots even when their storage source is caller-owned; the payload
        // must depend on the intervening materialization.
        .filter(|index| {
            !direct_payload_operands.contains(index) && !movement_operands.contains(index)
        })
        .filter_map(|index| {
            let id = NodeId::from_index(*index);
            crate::rangeify::computed_view(graph, id)
                .map(|view| view.source)
                .or_else(|_| {
                    let shape = graph
                        .shape(id)
                        .map_err(|_| crate::rangeify::RangeifyError::Invalid)?;
                    crate::rangeify::projected_source(graph, id, shape)
                })
                .ok()
                .filter(|source| external.contains(&source.index()))
                .map(|_| *index)
        })
        .collect::<BTreeSet<_>>();
    // Direct-payload kernels own dense operand IDs and cannot consume a
    // source-backed affine alias through the scalar IndexView path. If the
    // same alias is requested, retain its existing materialization root
    // rather than publishing conflicting passthrough/output ownership.
    requested_passthroughs.retain(|passthrough| {
        let id = passthrough.requested.index();
        !direct_payload_operands.contains(&id) && !movement_operands.contains(&id)
    });
    requested_passthrough_ids = requested_passthroughs
        .iter()
        .map(|passthrough| passthrough.requested.index())
        .collect();
    let requested_passthrough_sources = requested_passthroughs
        .iter()
        .map(|passthrough| passthrough.source.index())
        .collect::<BTreeSet<_>>();
    let mut roots: BTreeSet<usize> = needed
        .iter()
        .copied()
        .filter(|index| {
            let id = NodeId::from_index(*index);
            !external.contains(index)
                && !external_view_aliases.contains(index)
                && !requested_passthrough_ids.contains(index)
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
                    || direct_payload_operands.contains(index)
                    || movement_operands.contains(index)
                    || computed_view_sources.contains(index)
                    || (consumers[*index] > 1
                        && !matches!(graph.op(id), Ok(Op::Input { .. } | Op::Constant(_))))
                    || matches!(
                        graph.op(id),
                        Ok(Op::Random { .. }
                            | Op::ShapeIota { .. }
                            | Op::Threefry { .. }
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
                            | Op::Scatter { .. }
                            | Op::ScatterPositions { .. }
                            | Op::ScatterPositionsVjp { .. })
                    )
                    || !matches!(graph.op(id), Ok(op) if supported(op)))
        })
        .collect();
    let fusion_candidates = roots
        .iter()
        .copied()
        .filter_map(|root| {
            let reduction =
                match crate::kernel::single_reduction_epilogue(graph, NodeId::from_index(root)) {
                    Ok(Some(reduction)) => reduction,
                    Ok(None) => return None,
                    Err(error) => return Some(Err(ScheduleError::UOp(error))),
                };
            (roots.contains(&reduction.index())
                && !requested.contains(&reduction.index())
                && !requested_passthrough_sources.contains(&reduction.index())
                && !external.contains(&reduction.index())
                && crate::kernel::reduction_epilogue_node_uses(
                    graph,
                    NodeId::from_index(root),
                    reduction,
                )
                .is_ok_and(|uses| uses == consumers[reduction.index()]))
            .then_some(Ok((root, reduction.index())))
        })
        .collect::<Result<Vec<_>, ScheduleError>>()?;
    // The nearest observable root owns fusion. An outer epilogue must load a
    // requested, caller-materialized, or shared nested result instead of
    // subsuming it and removing the reduction recurrence needed to produce
    // that result. A non-observable nested root may be absorbed only when this
    // candidate owns every one of its graph uses.
    let fusion_candidates = fusion_candidates
        .into_iter()
        .filter(|(root, reduction)| {
            roots
                .iter()
                .chain(&requested)
                .chain(&requested_passthrough_sources)
                .chain(&external)
                .filter(|nested| **nested != *root && **nested != *reduction)
                .all(|nested| {
                    crate::kernel::reduction_epilogue_node_uses(
                        graph,
                        NodeId::from_index(*root),
                        NodeId::from_index(*nested),
                    )
                    .is_ok_and(|uses| {
                        uses == 0
                            || (!requested.contains(nested)
                                && !requested_passthrough_sources.contains(nested)
                                && !external.contains(nested)
                                && uses == consumers[*nested])
                    })
                })
        })
        .collect::<Vec<_>>();
    let selected_epilogues = fusion_candidates
        .iter()
        .copied()
        .filter(|(root, reduction)| {
            !fusion_candidates.iter().any(|(outer, outer_reduction)| {
                outer != root
                    && outer_reduction == reduction
                    && crate::kernel::reduction_epilogue_node_uses(
                        graph,
                        NodeId::from_index(*outer),
                        NodeId::from_index(*root),
                    )
                    .is_ok_and(|uses| uses != 0 && uses == consumers[*root])
            })
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut rehearsed_epilogues = BTreeMap::<usize, ReductionEpilogueRehearsal>::new();
    for (root, reduction) in selected_epilogues {
        let mut candidates = BTreeSet::from([reduction]);
        for nested in roots.iter().copied() {
            if nested != root
                && !requested.contains(&nested)
                && !requested_passthrough_sources.contains(&nested)
                && !external.contains(&nested)
                && !direct_payload_operands.contains(&nested)
                && !movement_operands.contains(&nested)
                && crate::kernel::reduction_epilogue_node_uses(
                    graph,
                    NodeId::from_index(root),
                    NodeId::from_index(nested),
                )
                .is_ok_and(|uses| uses != 0 && uses == consumers[nested])
            {
                candidates.insert(nested);
            }
        }
        let Some(fusion) =
            rehearse_reduction_epilogue(graph, root, reduction, &roots, &external, &candidates)
        else {
            continue;
        };
        rehearsed_epilogues.insert(
            root,
            ReductionEpilogueRehearsal {
                reduction,
                candidates,
                fusion,
            },
        );
    }
    let reserved_roots = rehearsed_epilogues
        .values()
        .flat_map(|rehearsal| rehearsal.fusion.load_nodes.iter().copied())
        .chain(rehearsed_epilogues.keys().copied())
        .chain(requested_passthrough_sources.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut accepted_epilogues =
        BTreeMap::<usize, (BTreeSet<usize>, ReductionEpilogueFusion)>::new();
    for (root, rehearsal) in rehearsed_epilogues {
        if reserved_roots.contains(&rehearsal.reduction) {
            continue;
        }
        let candidates = rehearsal
            .candidates
            .difference(&reserved_roots)
            .copied()
            .collect::<BTreeSet<_>>();
        let Some(fusion) = rehearse_reduction_epilogue(
            graph,
            root,
            rehearsal.reduction,
            &roots,
            &external,
            &candidates,
        ) else {
            continue;
        };
        if !fusion.load_nodes.is_subset(&reserved_roots) {
            continue;
        }
        accepted_epilogues.insert(root, (candidates, fusion));
    }
    let mut fused_epilogues = BTreeMap::<usize, ReductionEpilogueFusion>::new();
    for (root, (removed_roots, fusion)) in accepted_epilogues {
        for removed in &removed_roots {
            roots.remove(removed);
        }
        fused_epilogues.insert(root, fusion);
    }

    // An ordinary scalar consumer may absorb branch-local computed aliases
    // when every graph use is owned by that consumer. Affine producer aliases
    // still require one exact map; projected aliases retain their dense source
    // and remove only the redundant view root. The normalized trial Sink
    // remains the ABI authority, so uncertain or multi-owner cases retain
    // their roots.
    let mut scalar_alias_loads = fused_epilogues
        .values()
        .flat_map(|fusion| fusion.load_nodes.iter().copied())
        .chain(direct_payload_operands.iter().copied())
        .chain(requested_passthrough_sources.iter().copied())
        .collect::<BTreeSet<_>>();
    let epilogue_reserved = scalar_alias_loads
        .iter()
        .copied()
        .chain(fused_epilogues.keys().copied())
        // Movement operands remain protected from unrelated reduction/scalar
        // ownership, but are not hard loads against their own checked
        // Contiguous redirection. Its sole-use proof is the exact exception.
        .chain(movement_operands.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut scalar_alias_reserved = epilogue_reserved;
    let scalar_alias_fusions = if policy.redirect_contiguous {
        // Proposal acceptance and ordinary fallback are both load-bearing.
        // Grow one monotone reservation frontier until every root is evaluated
        // against every other root's exact current load inventory; mutate
        // ownership only after that fixed point is stable.
        loop {
            let mut next_reserved = scalar_alias_reserved.clone();
            let mut next_loads = scalar_alias_loads.clone();
            let mut next_fusions = BTreeMap::new();
            for root in roots.iter().copied() {
                let node = NodeId::from_index(root);
                if let Some(fusion) = checked_scalar_alias_fusion(
                    graph,
                    node,
                    &ScalarAliasOwnership {
                        roots: &roots,
                        external: &external,
                        requested: &requested,
                        consumers: &consumers,
                        protected: &scalar_alias_reserved,
                        hard_loads: &scalar_alias_loads,
                        movement_operand_owners: &movement_operand_owners,
                    },
                )? {
                    next_reserved.insert(root);
                    next_reserved.extend(fusion.load_nodes.iter().copied());
                    next_loads.extend(fusion.load_nodes.iter().copied());
                    next_fusions.insert(root, fusion);
                }
            }
            let proposed_removals = next_fusions
                .values()
                .flat_map(|fusion| fusion.removed_roots.iter().copied())
                .collect::<BTreeSet<_>>();
            for root in roots.iter().copied() {
                if next_fusions.contains_key(&root) || proposed_removals.contains(&root) {
                    continue;
                }
                let loads = ordinary_scalar_fallback_loads(
                    graph,
                    NodeId::from_index(root),
                    &roots,
                    &external,
                )?;
                if !loads.is_empty() {
                    next_reserved.insert(root);
                    next_reserved.extend(loads.iter().copied());
                    next_loads.extend(loads.iter().copied());
                }
            }
            if next_reserved == scalar_alias_reserved {
                scalar_alias_loads = next_loads;
                break next_fusions;
            }
            scalar_alias_reserved = next_reserved;
            scalar_alias_loads = next_loads;
        }
    } else {
        BTreeMap::new()
    };
    for fusion in scalar_alias_fusions.values() {
        for removed in &fusion.removed_roots {
            roots.remove(removed);
        }
    }

    // A dense Contiguous boundary normally owns a raw-copy item. When its
    // ordinary pure producer has exactly this one graph use and is neither
    // requested nor caller-owned, the producer can instead write directly to
    // the boundary's fresh dense buffer. The Contiguous node remains the sole
    // observable schedule/output identity; every uncertain ownership or
    // operation-specific producer retains the explicit copy.
    let mut contiguous_redirections = BTreeMap::<usize, ContiguousRedirection>::new();
    if policy.redirect_contiguous {
        for contiguous in roots.iter().copied().collect::<Vec<_>>() {
            let node = NodeId::from_index(contiguous);
            if !matches!(
                graph.op(node).map_err(ScheduleError::Graph)?,
                Op::Contiguous { .. }
            ) {
                continue;
            }
            // Trial the exact value-root/destination projection before changing
            // root ownership. Specialized roots lower as a load of `producer` and
            // are rejected below; ordinary scalar DAGs lower to one checked Store.
            let Some(redirection) = checked_contiguous_redirection(
                graph, node, &roots, &external, &requested, &consumers,
            )?
            else {
                continue;
            };
            if scalar_alias_loads.contains(&redirection.producer) {
                continue;
            }
            contiguous_redirections.insert(contiguous, redirection);
        }
    }
    for redirection in contiguous_redirections.values() {
        roots.remove(&redirection.producer);
    }

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
    let mut items = Vec::with_capacity(roots.len());
    for &index in &roots {
        let node = NodeId::from_index(index);
        let redirection = contiguous_redirections.get(&index);
        let alias_fusion = scalar_alias_fusions.get(&index);
        let mut leaf_ids = match (redirection, alias_fusion) {
            (Some(value), _) => value.load_nodes.clone(),
            (None, Some(value)) => value.load_nodes.clone(),
            (None, None) => fused_epilogues
                .get(&index)
                .map(|value| value.load_nodes.clone())
                .unwrap_or_default(),
        };
        let mut boundary = None;
        if redirection.is_none() && alias_fusion.is_none() && !fused_epilogues.contains_key(&index)
        {
            match crate::MovementKernelPlan::from_scheduled_graph(graph, node) {
                Ok(plan) => {
                    // Movement kernels can bypass observable graph-view roots.
                    // Their validated physical operands—not graph traversal—
                    // are therefore the authoritative input inventory.
                    leaf_ids.extend(
                        plan.input_operands()
                            .into_iter()
                            .map(|input| input.node.index()),
                    );
                }
                Err(crate::MovementPlanError::NotMovement) => LeafTraversal {
                    graph,
                    roots: &roots,
                    owner: index,
                    external: &external,
                    allow_projected: policy.allow_projected,
                }
                .visit(node, &mut leaf_ids, &mut boundary)?,
                Err(error) => return Err(ScheduleError::Binding(error.to_string())),
            }
        }
        let materialized = leaf_ids
            .iter()
            .filter(|leaf| roots.contains(leaf))
            .copied()
            .collect::<BTreeSet<_>>();
        let materialized = materialized
            .union(&external)
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
        let kernel = if boundary.is_none() {
            if let Some(redirection) = contiguous_redirections.get(&index) {
                redirection.kernel.clone()
            } else if let Some(fusion) = scalar_alias_fusions.get(&index) {
                fusion.kernel.clone()
            } else if let Some(fusion) = fused_epilogues.get(&index) {
                fusion.kernel.clone()
            } else {
                match graph.op(node).map_err(ScheduleError::Graph)? {
                    Op::Random { .. } => crate::kernel::lower_graph_random(graph, node)
                        .map_err(ScheduleError::UOp)?,
                    Op::Threefry { .. } => crate::kernel::lower_graph_threefry(graph, node)
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
                    | Op::Scatter { .. }
                    | Op::ScatterPositions { .. }
                    | Op::ScatterPositionsVjp { .. } => {
                        crate::kernel::lower_graph_movement(graph, node)
                            .map_err(ScheduleError::UOp)?
                    }
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
            }
        } else {
            UOp::sink(vec![])
        };
        let kernel = if boundary.is_none() && matches!(kernel.operation(), crate::Operation::Sink) {
            crate::uop::normalize_kernel(&kernel).map_err(ScheduleError::UOp)?
        } else {
            kernel
        };
        let kernel_nodes = kernel.topological().map_err(ScheduleError::UOp)?;
        for (buffer, view) in common_buffer_views(&kernel_nodes) {
            if let Some(desc) = inputs.iter_mut().find(|desc| desc.id == buffer) {
                desc.view = view;
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
            let mut dependencies = BTreeSet::new();
            for binding in &input_bindings {
                let input = binding.input_node.index();
                if binding.desc.id != input as u64 {
                    return Err(ScheduleError::Binding(
                        "scheduled input node/descriptor identity mismatch".into(),
                    ));
                }
                if let Some(producer) = node_to_item.get(&input) {
                    dependencies.insert(*producer);
                    continue;
                }
                match graph.op(binding.input_node).map_err(ScheduleError::Graph)? {
                    Op::Input { .. } | Op::Constant(_) => {}
                    _ if external.contains(&input)
                        && external_materializations.contains(&binding.input_node) => {}
                    _ => {
                        return Err(ScheduleError::Binding(format!(
                            "computed input producer {input} is absent"
                        )));
                    }
                }
            }
            dependencies
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
    let requested_materializations = physical_requested_materializations(
        &items,
        &requested_passthroughs,
        outputs.iter().map(|node| node.index() as u64),
    );
    let schedule = Schedule {
        items,
        requested_materializations,
        requested_passthroughs,
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
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Threefry {
                counter: lhs,
                key: rhs,
            } => {
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
                Op::Threefry { .. } => crate::kernel::lower_graph_threefry(graph, output)
                    .map_err(ScheduleError::UOp)?,
                _ => crate::kernel::lower_graph_elementwise(graph, output)
                    .map_err(ScheduleError::UOp)?,
            }
        } else {
            UOp::sink(vec![])
        };
    let kernel = if boundary.is_none() && matches!(kernel.operation(), crate::Operation::Sink) {
        crate::uop::normalize_kernel(&kernel).map_err(ScheduleError::UOp)?
    } else {
        kernel
    };
    let kernel_nodes = kernel.topological().map_err(ScheduleError::UOp)?;
    for (buffer, view) in common_buffer_views(&kernel_nodes) {
        if let Some(desc) = inputs.iter_mut().find(|desc| desc.id == buffer) {
            desc.view = view;
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
        requested_materializations: vec![output.index() as u64],
        requested_passthroughs: vec![],
        value_bindings: vec![],
        state_bindings: vec![],
    })
}

#[cfg(test)]
mod ownership_traversal_tests {
    use super::*;

    #[test]
    fn deep_ownership_walks_preserve_edges_and_do_not_use_the_call_stack() {
        std::thread::Builder::new()
            .name("deep-schedule-ownership".into())
            .stack_size(512 * 1024)
            .spawn(|| {
                let mut graph = Graph::new();
                let input = graph.input("input", [1, 1]);
                let mut chain = vec![input];
                for _ in 0..8_192 {
                    chain.push(graph.neg(*chain.last().unwrap()).unwrap());
                }
                let deep = *chain.last().unwrap();

                let duplicated = graph.add(deep, deep).unwrap();
                let mut needed = BTreeSet::new();
                let mut consumers = vec![0usize; graph.node_count()];
                mark_needed(
                    &graph,
                    duplicated,
                    &mut needed,
                    &mut consumers,
                    &BTreeSet::new(),
                )
                .unwrap();
                assert_eq!(consumers[deep.index()], 2);
                assert_eq!(graph_node_uses(&graph, duplicated, deep).unwrap(), 2);
                assert_eq!(graph_node_uses(&graph, deep, input).unwrap(), 1);

                let external = chain[chain.len() / 2];
                needed.clear();
                consumers.fill(0);
                mark_needed(
                    &graph,
                    deep,
                    &mut needed,
                    &mut consumers,
                    &BTreeSet::from([external.index()]),
                )
                .unwrap();
                assert!(needed.contains(&external.index()));
                assert!(!needed.contains(&chain[chain.len() / 2 - 1].index()));
                assert_eq!(consumers[external.index()], 1);

                let candidates = ScalarAliasCandidates::collect(
                    &graph,
                    deep,
                    &BTreeSet::new(),
                    &BTreeSet::new(),
                    &BTreeSet::new(),
                )
                .unwrap();
                assert!(candidates.affine_maps.is_empty());
                assert!(candidates.direct_roots.is_empty());
                assert!(candidates.affine_view_roots.is_empty());
                assert!(candidates.projected_view_roots.is_empty());

                let alias_input = graph.input("alias_input", [1]);
                let alias_source = graph.neg(alias_input).unwrap();
                let alias = graph.reshape(alias_source, [1, 1]).unwrap();
                let rhs = graph.input("rhs", [1, 1]);
                let alias_owner = graph.add(alias, rhs).unwrap();
                let alias_candidates = ScalarAliasCandidates::collect(
                    &graph,
                    alias_owner,
                    &BTreeSet::from([alias_source.index()]),
                    &BTreeSet::new(),
                    &BTreeSet::new(),
                )
                .unwrap();
                assert!(
                    alias_candidates
                        .affine_maps
                        .contains_key(&alias_source.index())
                );
                assert_eq!(
                    graph_node_uses(&graph, alias_owner, alias_source).unwrap(),
                    1
                );

                let mut leaf_ids = BTreeSet::new();
                let mut boundary = None;
                LeafTraversal {
                    graph: &graph,
                    roots: &BTreeSet::new(),
                    owner: deep.index(),
                    external: &BTreeSet::new(),
                    allow_projected: true,
                }
                .visit(deep, &mut leaf_ids, &mut boundary)
                .unwrap();
                assert_eq!(leaf_ids, BTreeSet::from([input.index()]));
                assert!(boundary.is_none());

                let reduction_input = graph.input("reduction_input", [1, 2]);
                let reduction = graph
                    .reduce(reduction_input, crate::ReduceKind::Sum, Some(vec![1]), true)
                    .unwrap();
                let mut epilogue = reduction;
                for _ in 0..4_096 {
                    epilogue = graph.neg(epilogue).unwrap();
                }
                assert_eq!(
                    crate::kernel::single_reduction_epilogue(&graph, epilogue).unwrap(),
                    Some(reduction)
                );
                assert_eq!(
                    crate::kernel::reduction_epilogue_node_uses(&graph, epilogue, reduction)
                        .unwrap(),
                    1
                );

                // Every duplicated value is an explicit schedule root, so
                // scalar lowering stays one operation deep while the public
                // external-materialization reachability check must still
                // traverse the complete ancestry without the call stack.
                let midpoint = graph.neg(input).unwrap();
                let mut output = midpoint;
                for _ in 0..4_096 {
                    output = graph.add(output, output).unwrap();
                }
                let scheduled =
                    schedule_with_external_materializations(&graph, &[output], &[midpoint])
                        .unwrap();
                let first = scheduled
                    .items
                    .first()
                    .expect("duplicated chain schedule item");
                assert_eq!(first.external_materializations, vec![midpoint]);
                assert_eq!(scheduled.items.last().unwrap().node, output);
                assert!(scheduled.items.iter().all(|item| item.boundary.is_none()));
                scheduled.validate().unwrap();
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
