//! A deterministic, non-mutating producer-aware schedule DAG. Pure
//! elementwise/view regions fuse into their consumers while materialization
//! roots retain stable buffer and UOp identities for realization.
use crate::{DType, Graph, NodeId, Op, Shape, UOp, UOpError};
use std::{
    collections::{BTreeSet, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};
pub mod artifact;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BufferDesc {
    pub id: u64,
    pub shape: Shape,
    pub dtype: DType,
    pub bytes: usize,
    pub alignment: usize,
    pub read_only: bool,
    pub view: Option<crate::ViewMap>,
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
    pub output: BufferDesc,
    pub kernel: UOp,
    pub boundary: Option<ScheduleBoundary>,
    pub cache_key: u64,
}
#[derive(Clone, Debug)]
pub struct Schedule {
    pub items: Vec<ScheduleItem>,
}
impl Schedule {
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
            .filter(|item| !requested.contains(&item.output.id))
            .map(|item| item.output.clone())
            .collect()
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
        if self.input_bindings.len() != self.inputs.len() {
            return Err(ScheduleError::Binding(
                "binding/inventory count mismatch".into(),
            ));
        }
        let mut nodes = BTreeSet::new();
        let mut buffers = BTreeSet::new();
        let mut indices = BTreeSet::new();
        for binding in &self.input_bindings {
            if binding.desc.id == self.output.id {
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
            if binding.input_node.index() as u64 == self.output.id
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
pub(crate) fn item_cache_key(item: &ScheduleItem) -> u64 {
    let mut hasher = DefaultHasher::new();
    item.id.hash(&mut hasher);
    item.node.hash(&mut hasher);
    item.dependencies.hash(&mut hasher);
    item.inputs.hash(&mut hasher);
    item.output.hash(&mut hasher);
    item.boundary.hash(&mut hasher);
    item.kernel.hash(&mut hasher);
    item.external_materializations.hash(&mut hasher);
    item.input_bindings.hash(&mut hasher);
    item.quantized_input_bindings.hash(&mut hasher);
    hasher.finish()
}
pub(crate) fn specialized_item_cache_key(
    item: &ScheduleItem,
    source_identity: u64,
    bindings: &[(u64, i64)],
) -> u64 {
    let mut hasher = DefaultHasher::new();
    item_cache_key(item).hash(&mut hasher);
    source_identity.hash(&mut hasher);
    bindings.hash(&mut hasher);
    hasher.finish()
}
fn input_bindings(
    kernel: &UOp,
    inputs: &[BufferDesc],
    output: &BufferDesc,
) -> Result<Vec<ScheduleInputBinding>, ScheduleError> {
    if matches!(kernel.kind(), crate::UOpKind::Movement)
        && let Some(plan) = kernel.arg().quantized_row_gather_plan()
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
    if matches!(kernel.kind(), crate::UOpKind::Matmul)
        && let Some(plan) = kernel.arg().quantized_matmul_plan()
    {
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
    if let (crate::UOpKind::Movement, crate::UArg::Movement(plan)) = (kernel.kind(), kernel.arg()) {
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
    if matches!(kernel.kind(), crate::UOpKind::Matmul)
        && let Some(plan) = kernel.arg().matmul_plan()
    {
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
            if &desc.shape != shape || desc.dtype != dtype || !desc.read_only || desc.view.is_some()
            {
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
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for node in kernel.topological().map_err(ScheduleError::UOp)? {
        if !matches!(node.kind(), crate::UOpKind::Load) {
            continue;
        }
        let Some(index) = node.sources().first() else {
            return Err(ScheduleError::Binding("load lacks index".into()));
        };
        let buffer = match index.arg() {
            crate::UArg::BufferIndex { buffer, .. }
            | crate::UArg::ViewBufferIndex { buffer, .. } => *buffer,
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
    if let Some(plan) = kernel.arg().quantized_matmul_plan() {
        plan.validate()
            .map_err(|error| ScheduleError::Binding(error.to_string()))?;
        return Ok(vec![QuantizedScheduleInputBinding {
            input_node: plan.weight,
            desc: plan.weight_desc.clone(),
            abi_index: 1,
        }]);
    }
    if let Some(plan) = kernel.arg().quantized_row_gather_plan() {
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
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::Scatter { .. }
            | Op::Reduce { .. }
            | Op::Matmul { .. }
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
            | Op::Unary { input, .. }
            | Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. }
            | Op::Reduce { input, .. } => vec![*input],
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Matmul { lhs, rhs } => vec![*lhs, *rhs],
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
        return Ok(Schedule { items: vec![] });
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
            | Op::Unary { input, .. }
            | Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. }
            | Op::Reduce { input, .. } => child(*input)?,
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Matmul { lhs, rhs } => {
                child(*lhs)?;
                child(*rhs)?;
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
    let roots: BTreeSet<usize> = needed
        .iter()
        .copied()
        .filter(|index| {
            let id = NodeId::from_index(*index);
            !external.contains(index)
                && (requested.contains(index)
                    || matmul_operands.contains(index)
                    || (consumers[*index] > 1
                        && !matches!(graph.op(id), Ok(Op::Input { .. } | Op::Constant(_))))
                    || matches!(
                        graph.op(id),
                        Ok(Op::Random { .. }
                            | Op::Reduce { .. }
                            | Op::Matmul { .. }
                            | Op::Concat { .. }
                            | Op::Gather { .. }
                            | Op::Scatter { .. })
                    )
                    || !matches!(graph.op(id), Ok(op) if supported(op)))
        })
        .collect();
    let node_to_item: std::collections::BTreeMap<usize, u64> = roots
        .iter()
        .enumerate()
        .map(|(item, node)| (*node, item as u64))
        .collect();
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
            Op::Cast { input, .. } | Op::Unary { input, .. } | Op::Reduce { input, .. } => {
                leaves(g, *input, roots, here, out, boundary, external)?
            }
            Op::Shrink { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Stride { input, .. } => match crate::rangeify::static_view(g, id) {
                Ok(view) => {
                    out.insert(view.source.index());
                }
                Err(_) => {
                    *boundary = Some(ScheduleBoundary::Unsupported(
                        "view of a computed value requires materialization",
                    ));
                    out.insert(input.index());
                }
            },
            Op::Binary { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::Matmul { lhs, rhs } => {
                leaves(g, *lhs, roots, here, out, boundary, external)?;
                leaves(g, *rhs, roots, here, out, boundary, external)?;
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
        let kernel = if boundary.is_none() {
            match graph.op(node).map_err(ScheduleError::Graph)? {
                Op::Random { .. } => {
                    crate::kernel::lower_graph_random(graph, node).map_err(ScheduleError::UOp)?
                }
                Op::Matmul { .. } => {
                    crate::kernel::lower_graph_matmul(graph, node).map_err(ScheduleError::UOp)?
                }
                Op::Concat { .. } | Op::Gather { .. } | Op::Scatter { .. } => {
                    crate::kernel::lower_graph_movement(graph, node).map_err(ScheduleError::UOp)?
                }
                Op::Reduce { .. } => crate::kernel::lower_graph_reduction_with_materialized(
                    graph,
                    node,
                    &materialized,
                )
                .map_err(ScheduleError::UOp)?,
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
            if let crate::UArg::ViewBufferIndex { buffer, view, .. } = value.arg()
                && let Some(desc) = inputs.iter_mut().find(|desc| desc.id == *buffer)
            {
                desc.view = Some(view.clone());
            }
        }
        let id = *node_to_item.get(&index).ok_or(ScheduleError::Overflow)?;
        let dependencies: Vec<u64> = inputs
            .iter()
            .filter_map(|desc| node_to_item.get(&(desc.id as usize)).copied())
            .collect();
        let external_materializations = input_bindings(&kernel, &inputs, &output)?
            .iter()
            .filter(|binding| external.contains(&binding.input_node.index()))
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>();
        let input_bindings = input_bindings(&kernel, &inputs, &output)?;
        let quantized_input_bindings = quantized_input_bindings(&kernel)?;
        let mut item = ScheduleItem {
            id,
            node,
            dependencies,
            consumers: vec![],
            inputs,
            input_bindings,
            quantized_input_bindings,
            external_materializations,
            output,
            kernel,
            boundary,
            cache_key: 0,
        };
        item.cache_key = item_cache_key(&item);
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
    Ok(Schedule { items })
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
            Op::Cast { input, .. } | Op::Unary { input, .. } => walk(g, *input, leaves, boundary)?,
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
            Op::Reduce { input, .. } => walk(g, *input, leaves, boundary)?,
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
                _ => crate::kernel::lower_graph_elementwise(graph, output)
                    .map_err(ScheduleError::UOp)?,
            }
        } else {
            UOp::sink(vec![])
        };
    for node in kernel.topological().map_err(ScheduleError::UOp)? {
        if let crate::UArg::ViewBufferIndex { buffer, view, .. } = node.arg()
            && let Some(desc) = inputs.iter_mut().find(|desc| desc.id == *buffer)
        {
            desc.view = Some(view.clone());
        }
    }
    let mut h = DefaultHasher::new();
    inputs.hash(&mut h);
    out.hash(&mut h);
    boundary.hash(&mut h);
    kernel.hash(&mut h);
    let input_bindings = input_bindings(&kernel, &inputs, &out)?;
    let quantized_input_bindings = quantized_input_bindings(&kernel)?;
    input_bindings.hash(&mut h);
    quantized_input_bindings.hash(&mut h);
    let cache_key = h.finish();
    Ok(Schedule {
        items: vec![ScheduleItem {
            id: 0,
            node: output,
            dependencies: vec![],
            consumers: vec![],
            inputs,
            input_bindings,
            quantized_input_bindings,
            external_materializations: vec![],
            output: out,
            kernel,
            boundary,
            cache_key,
        }],
    })
}
