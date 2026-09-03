//! Deterministic MSL lowering for static exact scalar and serial-reduction UOps.
use super::{
    MetalCapabilities, MetalError, guard::emit_transactional, transaction::MetalTransactionAbi,
};
use crate::{
    AffineView, DType, IndexValue, LiteralValue, MovementValue, Operation, ScheduleInputBinding,
    Shape, UOp,
    runtime::scalar_lane::{
        ScalarLaneDialect, dialect_seal, emit_scalar_lane, project_scalar_lane,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

pub const METAL_RENDERER_VERSION: &str = "rustgrad-metal-static-v8";
pub const METAL_RAW_COPY_RENDERER_VERSION: &str = "rustgrad-metal-raw-copy-v1";
pub const METAL_PORTABLE_BITCAST_RENDERER_VERSION: &str = "rustgrad-metal-portable-bitcast-v1";
pub const METAL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION: &str =
    "rustgrad-metal-portable-dense-materialization-v1";
pub const METAL_INDEXED_MOVEMENT_RENDERER_VERSION: &str = "rustgrad-metal-indexed-movement-v1";
pub const METAL_APPEND_STATE_RENDERER_VERSION: &str = "rustgrad-metal-append-state-v1";
pub const METAL_STATIC_POSITION_RENDERER_VERSION: &str = "rustgrad-metal-static-position-v1";
pub const METAL_PORTABLE_F32_MATMUL_RENDERER_VERSION: &str =
    "rustgrad-metal-portable-f32-matmul-v1";
pub const METAL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION: &str =
    "rustgrad-metal-portable-prefix-scan-v1";
pub const METAL_PORTABLE_SORT_RENDERER_VERSION: &str = "rustgrad-metal-portable-sort-v1";
pub const METAL_PORTABLE_THREEFRY_RENDERER_VERSION: &str = "rustgrad-metal-portable-threefry-v1";
pub const METAL_ABI_VERSION: u32 = 2;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// One ordered pointer entry in the Metal kernel ABI.
pub struct MetalBufferAbi {
    /// Stable scheduled buffer identity.
    pub id: u64,
    /// Exact storage dtype expected by the pointer.
    pub dtype: DType,
    /// Physical source-storage shape.
    pub source_shape: Shape,
    /// Physical source-storage element count.
    pub elements: usize,
    /// Whether this entry is an output pointer. Mutable entries are ordered
    /// by their scheduled output ordinal.
    pub mutable: bool,
    /// Optional source-backed affine logical mapping.
    pub view: Option<AffineView>,
}

/// Handle-free authenticated ABI for one complete-row append into an
/// exclusively owned recurrent state allocation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MetalAppendStateAbi {
    pub state_input: u64,
    pub state_output: u64,
    pub index: u64,
    pub updates: u64,
    pub axis: usize,
    pub axis_extent: usize,
    pub row_elements: usize,
}

#[derive(Clone, Debug)]
/// Immutable MSL source plus the complete checked launch contract.
pub struct RenderedMetal {
    /// Deterministically emitted Metal Shading Language source.
    pub source: String,
    /// UOp expression IDs to one-based generated source lines.
    pub source_map: BTreeMap<usize, usize>,
    /// Ordered input pointers followed by the ordered output pointers.
    pub buffers: Vec<MetalBufferAbi>,
    /// Exact launch work-item count supplied as the final scalar ABI value.
    /// This equals the logical output extent for ordinary kernels; PrefixScan
    /// and coupled Sort launch per independent lane, while Bitcast launches
    /// per raw byte.
    pub extent: usize,
    /// Generated MSL entry-point name.
    pub entry: String,
    /// Content-addressed renderer and capability identity.
    pub cache_key: String,
    /// Exact device capabilities used while rendering.
    pub capabilities: MetalCapabilities,
    /// Guard/status metadata when the output must be committed transactionally.
    pub transaction: Option<MetalTransactionAbi>,
    /// Renderer-private data-dependent movement status contract. Keeping it
    /// separate preserves every historical integer-transaction cache key.
    pub(super) indexed_movement: Option<super::MetalIndexedMovementAbi>,
    /// Authenticated sparse append-state contract. Presence means the output
    /// aliases the state input only inside an append-state session plan.
    pub(super) append_state: Option<MetalAppendStateAbi>,
    pub(super) schedule_inputs: Vec<MetalBufferAbi>,
    pub(super) semantic_program: Arc<super::dispatch::KernelSemanticProgram>,
}

impl RenderedMetal {
    /// Returns the checked data-dependent movement status contract, when this
    /// artifact uses one. Its presence requires transactional launch.
    pub fn indexed_movement(&self) -> Option<&super::MetalIndexedMovementAbi> {
        self.indexed_movement.as_ref()
    }

    /// Returns the sparse append-state ABI when this item was rendered under
    /// that explicit resource-free policy.
    pub fn append_state(&self) -> Option<&MetalAppendStateAbi> {
        self.append_state.as_ref()
    }

    /// Validates schedule-owned first-use ordering against the Metal pointer ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), MetalError> {
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Matmul(value) = root.operation()
        {
            crate::matmul::PortableF32Matmul::new(value)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::PrefixScan(value) = root.operation()
        {
            let portable = crate::prefix_scan_native::PortablePrefixScan::new(value)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
            crate::runtime::static_schedule::validate_portable_prefix_scan_bindings(
                &portable, bindings,
            )
            .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Sort(value) = root.operation()
        {
            crate::portable_sort::PortableSortPair::new(value)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Threefry(value) = root.operation()
        {
            crate::portable_threefry::PortableThreefry::new(value)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Movement(MovementValue::Plan(plan)) = root.operation()
            && matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. })
        {
            crate::movement_plan::PortableBitcast::new(plan)
                .and_then(|portable| portable.validate_schedule_bindings(bindings))
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Movement(MovementValue::Plan(plan)) = root.operation()
            && matches!(
                &plan.kind,
                crate::MovementKernelKind::Pad { .. } | crate::MovementKernelKind::Concat { .. }
            )
        {
            crate::movement_plan::PortableDenseMaterialization::new(plan)
                .and_then(|portable| portable.validate_schedule_bindings(bindings))
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Movement(MovementValue::Plan(plan)) = root.operation()
            && matches!(
                &plan.kind,
                crate::MovementKernelKind::Gather { .. }
                    | crate::MovementKernelKind::Scatter { .. }
            )
        {
            crate::movement_plan::PortableIndexedMovement::new(plan)
                .and_then(|portable| portable.validate_schedule_bindings(bindings))
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        }
        if bindings.len() != self.schedule_inputs.len() {
            return Err(MetalError::InvalidBinding(
                "schedule/Metal input count mismatch".into(),
            ));
        }
        for (position, (binding, expected)) in
            bindings.iter().zip(&self.schedule_inputs).enumerate()
        {
            if binding.abi_index != position
                || binding.desc.id != expected.id
                || binding.desc.dtype != expected.dtype
                || binding.desc.shape != expected.source_shape
                || binding.desc.view != expected.view
                || binding.desc.bytes
                    != expected
                        .elements
                        .checked_mul(expected.dtype.itemsize())
                        .ok_or(MetalError::Overflow)?
            {
                return Err(MetalError::InvalidBinding(format!(
                    "schedule binding {position} mismatches Metal ABI"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Pure static MSL renderer configured for one device capability identity.
pub struct MetalRenderer {
    /// Preferred launch threadgroup width; pipeline limits are checked later.
    pub local_size: usize,
    /// Selected device capabilities included in source/cache identity.
    pub capabilities: MetalCapabilities,
}

impl MetalRenderer {
    /// Creates a renderer, rejecting a zero threadgroup width.
    pub fn new(local_size: usize, capabilities: MetalCapabilities) -> Result<Self, MetalError> {
        if local_size == 0 {
            return Err(MetalError::InvalidArgument("zero local size"));
        }
        Ok(Self {
            local_size,
            capabilities,
        })
    }

    /// Lowers a validated scheduled UOp into the exact static subset.
    pub fn render(&self, root: &UOp) -> Result<RenderedMetal, MetalError> {
        if let Operation::Matmul(value) = root.operation() {
            return render_portable_f32_matmul(self, root, value);
        }
        if let Operation::PrefixScan(value) = root.operation() {
            return render_portable_prefix_scan(self, root, value);
        }
        if let Operation::Sort(value) = root.operation() {
            return render_portable_sort(self, root, value);
        }
        if let Operation::Threefry(value) = root.operation() {
            return render_portable_threefry(self, root, value);
        }
        if let Operation::Movement(value) = root.operation() {
            return match value {
                MovementValue::Plan(plan)
                    if matches!(
                        &plan.kind,
                        crate::MovementKernelKind::ScatterPositions { .. }
                    ) =>
                {
                    render_static_positions(self, root, plan)
                }
                MovementValue::Plan(plan)
                    if matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. }) =>
                {
                    render_portable_bitcast(self, root, plan)
                }
                MovementValue::Plan(plan)
                    if matches!(
                        &plan.kind,
                        crate::MovementKernelKind::Pad { .. }
                            | crate::MovementKernelKind::Concat { .. }
                    ) =>
                {
                    render_portable_dense_materialization(self, root, plan)
                }
                MovementValue::Plan(plan)
                    if matches!(
                        &plan.kind,
                        crate::MovementKernelKind::Gather { .. }
                            | crate::MovementKernelKind::Scatter { .. }
                    ) =>
                {
                    render_indexed_movement(self, root, plan)
                }
                MovementValue::Plan(plan) => render_raw_copy(self, root, plan),
                MovementValue::QuantizedRowGather(_) => Err(MetalError::Unsupported(
                    "quantized movement is outside Metal contiguous-copy lowering".into(),
                )),
            };
        }
        if let Operation::Random(plan) = root.operation() {
            return super::random::render(self, plan);
        }
        if matches!(root.operation(), Operation::TensorGuard(_)) {
            return Err(MetalError::Unsupported(
                "guards are outside Metal lowering".into(),
            ));
        }
        root.validate()
            .map_err(|error| MetalError::Unsupported(error.to_string()))?;
        let nodes = root
            .topological()
            .map_err(|error| MetalError::Unsupported(error.to_string()))?;
        if nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::Barrier | Operation::If | Operation::EndIf
            )
        }) {
            return Err(MetalError::Unsupported(
                "effects and control flow are outside the exact Metal subset".into(),
            ));
        }
        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.operation(), Operation::Store))
            .ok_or_else(|| MetalError::Unsupported("sink has no store".into()))?;
        let output_index = store
            .sources()
            .first()
            .ok_or_else(|| MetalError::Unsupported("store has no index".into()))?;
        let Operation::Index(IndexValue::Buffer {
            buffer: output_id,
            elements: extent,
            input_shape: output_shape,
            output_shape: store_shape,
            addressing: crate::IndexAddressing::Broadcast,
        }) = output_index.operation()
        else {
            return Err(MetalError::Unsupported(
                "output requires a contiguous BufferIndex".into(),
            ));
        };
        if output_shape != store_shape {
            return Err(MetalError::Unsupported(
                "non-contiguous output addressing".into(),
            ));
        }
        let output_dtype = output_index
            .ty()
            .ok_or_else(|| MetalError::Unsupported("untyped output index".into()))?
            .scalar;
        supported_storage(output_dtype)?;

        let common_views = crate::schedule::common_buffer_views(&nodes);
        let mut inventory = BTreeMap::<u64, MetalBufferAbi>::new();
        for node in &nodes {
            let (buffer, source_shape, elements) = match node.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements,
                    input_shape,
                    ..
                }) => (*buffer, input_shape.clone(), *elements),
                Operation::Index(IndexValue::View { buffer, view, .. }) => {
                    let access = MetalViewAccess::new(view)?;
                    let elements = access
                        .source_shape
                        .numel()
                        .map_err(|_| MetalError::Overflow)?;
                    (*buffer, access.source_shape, elements)
                }
                _ => continue,
            };
            let dtype = node
                .ty()
                .ok_or_else(|| MetalError::Unsupported("untyped buffer index".into()))?
                .scalar;
            supported_storage(dtype)?;
            let abi = MetalBufferAbi {
                id: buffer,
                dtype,
                source_shape,
                elements,
                mutable: buffer == *output_id,
                view: common_views.get(&buffer).cloned().flatten(),
            };
            if let Some(previous) = inventory.insert(buffer, abi.clone())
                && previous != abi
            {
                return Err(MetalError::InvalidBinding(format!(
                    "buffer {buffer} has conflicting ABI metadata"
                )));
            }
        }

        let mut seen = BTreeSet::new();
        let mut schedule_inputs = Vec::new();
        for node in &nodes {
            if !matches!(node.operation(), Operation::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| MetalError::InvalidBinding("load lacks index".into()))?;
            let buffer = match index.operation() {
                Operation::Index(IndexValue::Buffer { buffer, .. })
                | Operation::Index(IndexValue::View { buffer, .. }) => *buffer,
                _ => {
                    return Err(MetalError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            if seen.insert(buffer) {
                schedule_inputs.push(
                    inventory
                        .get(&buffer)
                        .ok_or_else(|| MetalError::InvalidBinding("load ABI missing".into()))?
                        .clone(),
                );
            }
        }
        let mut buffers = schedule_inputs.clone();
        if seen.insert(*output_id) {
            buffers.push(
                inventory
                    .get(output_id)
                    .ok_or_else(|| MetalError::InvalidBinding("output ABI missing".into()))?
                    .clone(),
            );
        }
        if buffers.last().is_none_or(|buffer| buffer.id != *output_id) {
            return Err(MetalError::InvalidBinding(
                "output aliases an input buffer".into(),
            ));
        }
        let output_position = buffers.len() - 1;
        let ids = buffers
            .iter()
            .enumerate()
            .map(|(position, buffer)| (buffer.id, position))
            .collect::<BTreeMap<_, _>>();
        let value = store
            .sources()
            .get(1)
            .ok_or_else(|| MetalError::Unsupported("store has no value".into()))?;
        let reduction = crate::reduction_native::NativeReductionKernel::from_store(store)
            .map_err(|reason| MetalError::Unsupported(reason.into()))?;
        let reduction_roots = nodes
            .iter()
            .filter(|node| matches!(node.operation(), Operation::ReduceFinalize))
            .fold(Vec::<&UOp>::new(), |mut roots, node| {
                if !roots.iter().any(|root| node.shares_node_with(root)) {
                    roots.push(node);
                }
                roots
            })
            .len();
        if reduction_roots != usize::from(reduction.is_some()) {
            return Err(MetalError::Unsupported(
                "reduction must be the sole stored value".into(),
            ));
        }
        if let Some(reduction) = &reduction {
            let plan = &reduction.plan;
            for dtype in [plan.source_dtype, plan.accumulator_dtype, plan.output_dtype] {
                supported_storage(dtype)?;
            }
        }
        let transaction = MetalTransactionAbi::analyze(
            value,
            output_position,
            buffers[output_position].source_shape.clone(),
        )?;
        if transaction.is_some()
            && nodes
                .iter()
                .any(crate::projected_index::ProjectedIndexPlan::is_projected)
        {
            return Err(MetalError::Unsupported(
                "guarded projected indexing is outside the exact Metal subset".into(),
            ));
        }
        let entry = format!("rg_metal_e{}_b{}", extent, buffers.len());
        let mut lines = vec![
            "#include <metal_stdlib>".into(),
            "using namespace metal;".into(),
            format!("// {METAL_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
            format!("kernel void {entry}("),
        ];
        for (position, buffer) in buffers.iter().enumerate() {
            let qualifier = if buffer.mutable { "" } else { "const " };
            lines.push(format!(
                "    device {qualifier}{}* b{position} [[buffer({position})]],",
                metal_storage_type(buffer.dtype)
            ));
        }
        lines.push(format!(
            "    constant ulong& extent [[buffer({})]],",
            buffers.len()
        ));
        if transaction.is_some() {
            lines.push(format!(
                "    device atomic_uint* rg_status [[buffer({})]],",
                buffers.len() + 1
            ));
        }
        lines.push("    uint gid [[thread_position_in_grid]]) {".into());
        lines.push("  if ((ulong)gid >= extent) return;".into());
        let mut source_map = BTreeMap::new();
        let expression = if let Some(reduction) = &reduction {
            if transaction.is_some() {
                return Err(MetalError::Unsupported(
                    "guarded reduction producers are outside the exact Metal subset".into(),
                ));
            }
            emit_metal_reduction(
                reduction,
                output_position,
                &ids,
                &mut source_map,
                &mut lines,
            )?;
            None
        } else if let Some(transaction) = &transaction {
            Some(emit_transactional(
                value,
                transaction,
                &ids,
                &mut source_map,
                &mut lines,
            )?)
        } else {
            Some(emit_expr(
                value,
                &ids,
                &mut source_map,
                &mut lines,
                "(ulong)gid",
            )?)
        };
        if let Some(expression) = expression {
            let stored = if output_dtype == DType::Bool {
                format!("(uchar)(({expression}) != 0)")
            } else {
                expression
            };
            lines.push(if transaction.is_some() {
                format!("  if (rg_ok) b{output_position}[gid] = {stored};")
            } else {
                format!("  b{output_position}[gid] = {stored};")
            });
        }
        lines.push("}".into());
        let source = lines.join("\n") + "\n";
        let cache_key = stable_key(&(
            METAL_RENDERER_VERSION,
            METAL_ABI_VERSION,
            self.local_size,
            &self.capabilities,
            &source,
            &buffers,
            &schedule_inputs,
            &transaction,
        ));
        Ok(RenderedMetal {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            capabilities: self.capabilities.clone(),
            transaction,
            indexed_movement: None,
            append_state: None,
            schedule_inputs,
            semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
                root.clone(),
            ))),
        })
    }

    pub(crate) fn render_append_state(
        &self,
        root: &UOp,
        link: &crate::runtime::static_schedule::StaticAppendStateLink,
    ) -> Result<RenderedMetal, MetalError> {
        root.validate()
            .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        let Operation::Movement(MovementValue::Plan(plan)) = root.operation() else {
            return Err(MetalError::InvalidBinding(
                "append state owner is not a movement plan".into(),
            ));
        };
        let portable = crate::movement_plan::PortableIndexedMovement::new(plan)
            .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        let crate::MovementKernelKind::Scatter {
            base,
            index,
            updates,
            axis,
            add,
        } = &plan.kind
        else {
            return Err(MetalError::InvalidBinding(
                "append state owner is not Scatter".into(),
            ));
        };
        if *add
            || base.node.index() as u64 != link.input
            || plan.output.index() as u64 != link.output
            || index.node.index() as u64 != link.index
            || updates.node.index() as u64 != link.updates
            || *axis != link.axis
            || portable.axis_extent() != link.axis_extent
            || portable.index_elements() != link.row_elements
            || index.shape != updates.shape
            || index.shape.rank() != base.shape.rank()
            || index.shape.dims()[*axis] != 1
            || index
                .shape
                .dims()
                .iter()
                .zip(base.shape.dims())
                .enumerate()
                .any(|(position, (index, base))| position != *axis && index != base)
        {
            return Err(MetalError::InvalidBinding(
                "append state movement geometry mismatch".into(),
            ));
        }
        let mut buffers = portable
            .inputs()
            .iter()
            .map(|input| {
                Ok(MetalBufferAbi {
                    id: input.node.index() as u64,
                    dtype: input.dtype,
                    source_shape: input.shape.clone(),
                    elements: input.shape.numel().map_err(|_| MetalError::Overflow)?,
                    mutable: false,
                    view: None,
                })
            })
            .collect::<Result<Vec<_>, MetalError>>()?;
        let schedule_inputs = buffers.clone();
        let output_position = buffers.len();
        buffers.push(MetalBufferAbi {
            id: link.output,
            dtype: DType::F32,
            source_shape: base.shape.clone(),
            elements: portable.output_elements(),
            mutable: true,
            view: None,
        });
        let abi = MetalAppendStateAbi {
            state_input: link.input,
            state_output: link.output,
            index: link.index,
            updates: link.updates,
            axis: link.axis,
            axis_extent: link.axis_extent,
            row_elements: link.row_elements,
        };
        let index_position = portable.index_abi();
        let update_position = portable
            .update_abi()
            .ok_or_else(|| MetalError::InvalidBinding("append updates ABI is absent".into()))?;
        let mut lines = vec![
            format!("// {METAL_APPEND_STATE_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
            "#include <metal_stdlib>".into(),
            "using namespace metal;".into(),
            "kernel void rg_metal_append_state_f32_i32(".into(),
        ];
        for (position, input) in portable.inputs().iter().enumerate() {
            lines.push(format!(
                "    device const {}* b{position} [[buffer({position})]],",
                metal_storage_type(input.dtype)
            ));
        }
        lines.extend([
            format!("    device float* b{output_position} [[buffer({output_position})]],"),
            format!("    constant ulong& extent [[buffer({})]],", buffers.len()),
            "    uint rg_gid [[thread_position_in_grid]]) {".into(),
            "  const ulong gid = (ulong)rg_gid;".into(),
            "  if (gid >= extent) return;".into(),
            format!("  const int rg_selected = b{index_position}[gid];"),
        ]);
        let destination = portable
            .axes()
            .iter()
            .map(|axis| {
                let coordinate = if axis.axis == portable.axis() {
                    "(ulong)rg_selected".into()
                } else {
                    indexed_coordinate("gid", axis.index_divisor, axis.index_dimension)
                };
                format!("({coordinate} * (ulong){}ul)", axis.data_stride)
            })
            .collect::<Vec<_>>()
            .join(" + ");
        lines.push(format!(
            "  b{output_position}[{}] = b{update_position}[gid];",
            if destination.is_empty() {
                "0ul"
            } else {
                &destination
            }
        ));
        lines.push("}".into());
        let source = lines.join("\n") + "\n";
        let cache_key = stable_key(&(
            METAL_APPEND_STATE_RENDERER_VERSION,
            METAL_ABI_VERSION,
            self.local_size,
            &self.capabilities,
            plan,
            &abi,
            &source,
            &buffers,
        ));
        Ok(RenderedMetal {
            source,
            source_map: BTreeMap::new(),
            buffers,
            extent: link.row_elements,
            entry: "rg_metal_append_state_f32_i32".into(),
            cache_key,
            capabilities: self.capabilities.clone(),
            transaction: None,
            indexed_movement: None,
            append_state: Some(abi),
            schedule_inputs,
            semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
                root.clone(),
            ))),
        })
    }
}

fn render_portable_threefry(
    renderer: &MetalRenderer,
    root: &UOp,
    value: &crate::ThreefryValue,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::portable_threefry::PortableThreefry::new(value).map_err(|error| match error {
            crate::portable_threefry::PortableThreefryError::Unsupported(reason) => {
                MetalError::Unsupported(reason.into())
            }
            crate::portable_threefry::PortableThreefryError::Overflow => MetalError::Overflow,
            other => MetalError::InvalidBinding(other.to_string()),
        })?;
    let mut buffers = portable
        .inputs()
        .iter()
        .map(|input| MetalBufferAbi {
            id: input.node.index() as u64,
            dtype: DType::U64,
            source_shape: input.shape.clone(),
            elements: input.elements,
            mutable: false,
            view: None,
        })
        .collect::<Vec<_>>();
    let schedule_inputs = buffers.clone();
    buffers.push(MetalBufferAbi {
        id: value.output.index() as u64,
        dtype: DType::U64,
        source_shape: value.output_shape.clone(),
        elements: portable.elements(),
        mutable: true,
        view: None,
    });
    for buffer in &buffers {
        let bytes = buffer
            .elements
            .checked_mul(DType::U64.itemsize())
            .ok_or(MetalError::Overflow)?;
        if bytes > renderer.capabilities.max_buffer_length {
            return Err(MetalError::Unsupported(
                "portable Threefry binding exceeds device buffer limit".into(),
            ));
        }
    }
    let entry = format!("rg_metal_threefry_e{}", portable.elements());
    let mut lines = vec![
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        format!("// {METAL_PORTABLE_THREEFRY_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        format!("kernel void {entry}("),
    ];
    for (index, buffer) in buffers.iter().enumerate() {
        let qualifier = if buffer.mutable { "" } else { "const " };
        lines.push(format!(
            "    device {qualifier}ulong* b{index} [[buffer({index})]],"
        ));
    }
    lines.extend([
        format!("    constant ulong& extent [[buffer({})]],", buffers.len()),
        "    uint gid32 [[thread_position_in_grid]]) {".into(),
        "  const ulong gid = (ulong)gid32;".into(),
        "  if (gid >= extent) return;".into(),
    ]);
    lines.extend(crate::portable_threefry::emit_portable_threefry_body(
        &portable,
        &crate::portable_threefry::CLikePortableThreefryDialect,
    ));
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_PORTABLE_THREEFRY_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.elements(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_sort(
    renderer: &MetalRenderer,
    root: &UOp,
    value: &crate::SortValue,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::portable_sort::PortableSortPair::new(value).map_err(|error| match error {
            crate::portable_sort::PortableSortError::Unsupported(reason) => {
                MetalError::Unsupported(reason.into())
            }
            crate::portable_sort::PortableSortError::Overflow => MetalError::Overflow,
            other => MetalError::InvalidBinding(other.to_string()),
        })?;
    let elements = portable.elements();
    let input = MetalBufferAbi {
        id: value.input.index() as u64,
        dtype: value.dtype,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: false,
        view: None,
    };
    let values = MetalBufferAbi {
        id: value.values.index() as u64,
        dtype: value.dtype,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: true,
        view: None,
    };
    let indices = MetalBufferAbi {
        id: value.indices.index() as u64,
        dtype: DType::I32,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: true,
        view: None,
    };
    let buffers = vec![input.clone(), values, indices];
    for buffer in &buffers {
        let bytes = buffer
            .elements
            .checked_mul(buffer.dtype.itemsize())
            .ok_or(MetalError::Overflow)?;
        if bytes > renderer.capabilities.max_buffer_length {
            return Err(MetalError::Unsupported(
                "portable sort binding exceeds device buffer limit".into(),
            ));
        }
    }
    let scalar_type = metal_storage_type(value.dtype);
    let padding = match (value.dtype, value.descending) {
        (DType::Bool, true) => "(uchar)0".into(),
        (DType::Bool, false) => "(uchar)1".into(),
        (DType::I32, true) => "as_type<int>(0x80000000u)".into(),
        (DType::I32, false) => "as_type<int>(0x7fffffffu)".into(),
        (DType::U32, true) => "0u".into(),
        (DType::U32, false) => "0xffffffffu".into(),
        (DType::F32, true) => "as_type<float>(0xff800000u)".into(),
        (DType::F32, false) => "as_type<float>(0x7f800000u)".into(),
        _ => unreachable!("portable sort validated storage"),
    };
    let entry = format!(
        "rg_metal_sort_{:?}_a{}_n{}",
        value.dtype,
        value.axis,
        portable.elements()
    )
    .to_ascii_lowercase();
    let mut lines = vec![
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        "#pragma clang fp contract(off)".into(),
        format!("// {METAL_PORTABLE_SORT_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        format!("kernel void {entry}("),
        format!("    device const {scalar_type}* b0 [[buffer(0)]],"),
        format!("    device {scalar_type}* b1 [[buffer(1)]],"),
        "    device int* b2 [[buffer(2)]],".into(),
        "    constant ulong& extent [[buffer(3)]],".into(),
        "    uint gid32 [[thread_position_in_grid]]) {".into(),
        "  const ulong gid = (ulong)gid32;".into(),
        "  if (gid >= extent) return;".into(),
    ];
    lines.extend(
        crate::portable_sort::emit_portable_sort_body(
            &portable,
            &crate::portable_sort::CLikePortableSortDialect {
                scalar_type,
                padding,
            },
        )
        .map_err(|error| MetalError::Unsupported(error.to_string()))?,
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_PORTABLE_SORT_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.launch_extent(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs: vec![input],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_prefix_scan(
    renderer: &MetalRenderer,
    root: &UOp,
    value: &crate::PrefixScanValue,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::prefix_scan_native::PortablePrefixScan::new(value).map_err(|error| match error {
            crate::prefix_scan_native::PortablePrefixScanError::Unsupported(reason) => {
                MetalError::Unsupported(reason.into())
            }
            crate::prefix_scan_native::PortablePrefixScanError::Overflow => MetalError::Overflow,
            other => MetalError::InvalidBinding(other.to_string()),
        })?;
    let plan = portable.plan();
    let input = MetalBufferAbi {
        id: plan.input,
        dtype: plan.input_dtype,
        source_shape: value.input_shape.clone(),
        elements: plan.elements,
        mutable: false,
        view: None,
    };
    let output = MetalBufferAbi {
        id: plan.output,
        dtype: plan.output_dtype,
        source_shape: value.output_shape.clone(),
        elements: plan.elements,
        mutable: true,
        view: None,
    };
    let buffers = vec![input.clone(), output];
    for buffer in &buffers {
        let bytes = buffer
            .elements
            .checked_mul(buffer.dtype.itemsize())
            .ok_or(MetalError::Overflow)?;
        if bytes > renderer.capabilities.max_buffer_length {
            return Err(MetalError::Unsupported(
                "portable scan binding exceeds device buffer limit".into(),
            ));
        }
    }
    let entry = format!(
        "rg_metal_scan_{:?}_{:?}_a{}_n{}",
        plan.kind, plan.result, plan.axis, plan.elements
    )
    .to_ascii_lowercase();
    let input_type = metal_storage_type(plan.input_dtype);
    let output_type = metal_storage_type(plan.output_dtype);
    let mut lines = vec![
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        "#pragma clang fp contract(off)".into(),
        format!("// {METAL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        format!("kernel void {entry}("),
        format!("    device const {input_type}* b0 [[buffer(0)]],"),
        format!("    device {output_type}* b1 [[buffer(1)]],"),
        "    constant ulong& extent [[buffer(2)]],".into(),
        "    uint gid32 [[thread_position_in_grid]]) {".into(),
        "  const ulong gid = (ulong)gid32;".into(),
        "  if (gid >= extent) return;".into(),
    ];
    lines.extend(
        crate::prefix_scan_native::emit_portable_prefix_scan_body(
            &portable,
            &MetalPrefixScanDialect,
        )
        .map_err(|error| MetalError::Unsupported(error.to_string()))?,
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.launch_extent(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs: vec![input],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

struct MetalPrefixScanDialect;

impl crate::prefix_scan_native::PortablePrefixScanDialect for MetalPrefixScanDialect {
    fn scalar_body(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<Vec<String>, crate::prefix_scan_native::PortablePrefixScanError> {
        Ok(vec![match plan.result {
            crate::PrefixScanOutput::Indices => "  b1[0] = (int)0;".into(),
            crate::PrefixScanOutput::Values if plan.input_dtype == plan.output_dtype => {
                if plan.input_dtype == DType::F32 {
                    "  b1[0] = as_type<float>(as_type<uint>(b0[0]));".into()
                } else {
                    "  b1[0] = b0[0];".into()
                }
            }
            crate::PrefixScanOutput::Values => "  b1[0] = (int)b0[0];".into(),
        }])
    }

    fn domain(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> Vec<String> {
        vec![
            format!(
                "  const uint rg_row = (uint)(gid / (ulong){}ul);",
                plan.inner
            ),
            format!(
                "  const uint rg_inner = (uint)(gid % (ulong){}ul);",
                plan.inner
            ),
        ]
    }

    fn identity(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<&'static str, crate::prefix_scan_native::PortablePrefixScanError> {
        Ok(match (plan.kind, plan.work_dtype) {
            (crate::PrefixScanKind::Sum, DType::F32) => "0.0f",
            (crate::PrefixScanKind::Product, DType::F32) => "1.0f",
            (crate::PrefixScanKind::Max, DType::F32) => "as_type<float>(0xff800000u)",
            (crate::PrefixScanKind::Min, DType::F32) => "as_type<float>(0x7f800000u)",
            (crate::PrefixScanKind::Product | crate::PrefixScanKind::Min, DType::Bool) => {
                "(uchar)1"
            }
            (crate::PrefixScanKind::Product, DType::I32) => "(int)1",
            (crate::PrefixScanKind::Product, DType::U32) => "1u",
            (crate::PrefixScanKind::Max, DType::I32) => "as_type<int>(0x80000000u)",
            (crate::PrefixScanKind::Min, DType::I32) => "as_type<int>(0x7fffffffu)",
            (crate::PrefixScanKind::Min, DType::U32) => "0xffffffffu",
            (_, DType::Bool | DType::I32 | DType::U32) => "0",
            _ => {
                return Err(
                    crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                        "Metal portable scan identity dtype",
                    ),
                );
            }
        })
    }

    fn accumulator(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
        identity: &str,
    ) -> String {
        format!(
            "  {} rg_acc = {identity};",
            metal_storage_type(plan.work_dtype)
        )
    }

    fn index(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!("  int rg_index = (int){};", plan.index_sentinel)
    }

    fn loop_open(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!(
            "  for (uint rg_axis = 0u; rg_axis < {}u; ++rg_axis) {{",
            plan.axis_len
        )
    }

    fn offset(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!(
            "    const ulong rg_offset = ((ulong)rg_row * (ulong){}ul + (ulong)rg_axis) * (ulong){}ul + (ulong)rg_inner;",
            plan.axis_len, plan.inner
        )
    }

    fn load(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<String, crate::prefix_scan_native::PortablePrefixScanError> {
        let work = metal_storage_type(plan.work_dtype);
        Ok(format!("    const {work} rg_next = ({work})b0[rg_offset];"))
    }

    fn strict(
        &self,
        _plan: &crate::prefix_scan_native::NativePrefixScanPlan,
        operator: &str,
    ) -> String {
        format!("    const bool rg_strict = rg_next {operator} rg_acc;")
    }

    fn equal_before(&self) -> String {
        "    const bool rg_equal_before = rg_next == rg_acc;".into()
    }

    fn update_extrema(&self) -> String {
        "    if (rg_strict) rg_acc = rg_next;".into()
    }

    fn update_first_index(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!(
            "    if (rg_strict || (rg_index == (int){} && rg_equal_before)) rg_index = (int)rg_axis;",
            plan.index_sentinel
        )
    }

    fn arithmetic(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
        operator: &str,
    ) -> Result<String, crate::prefix_scan_native::PortablePrefixScanError> {
        let expression = match plan.work_dtype {
            DType::Bool => "(uchar)((rg_acc != 0) && (rg_next != 0))".into(),
            DType::I32 => {
                format!("as_type<int>(as_type<uint>(rg_acc) {operator} as_type<uint>(rg_next))")
            }
            DType::U32 | DType::F32 => format!("rg_acc {operator} rg_next"),
            _ => {
                return Err(
                    crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                        "Metal portable scan arithmetic dtype",
                    ),
                );
            }
        };
        Ok(format!("    rg_acc = {expression};"))
    }

    fn store(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> Vec<String> {
        let stored = if plan.result == crate::PrefixScanOutput::Indices {
            "rg_index".into()
        } else if plan.output_dtype == DType::Bool {
            "(uchar)(rg_acc != 0)".into()
        } else {
            format!("({})rg_acc", metal_storage_type(plan.output_dtype))
        };
        vec![format!("    b1[rg_offset] = {stored};")]
    }

    fn loop_close(&self) -> String {
        "  }".into()
    }
}

fn render_portable_f32_matmul(
    renderer: &MetalRenderer,
    root: &UOp,
    value: &crate::MatmulValue,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let portable = crate::matmul::PortableF32Matmul::new(value).map_err(|error| match error {
        crate::matmul::PortableF32MatmulError::Unsupported(reason) => {
            MetalError::Unsupported(reason.into())
        }
        crate::matmul::PortableF32MatmulError::Overflow => MetalError::Overflow,
        other => MetalError::InvalidBinding(other.to_string()),
    })?;
    if portable.extent() > u32::MAX as usize {
        return Err(MetalError::Unsupported(
            "portable matmul extent exceeds uint thread indexing".into(),
        ));
    }
    let plan = portable.plan();
    let mut buffers = Vec::with_capacity(3);
    let mut schedule_inputs = Vec::with_capacity(2);
    for input in portable.inputs() {
        let elements = input.shape.numel().map_err(|_| MetalError::Overflow)?;
        let abi = MetalBufferAbi {
            id: input.node.index() as u64,
            dtype: DType::F32,
            source_shape: input.shape.clone(),
            elements,
            mutable: false,
            view: None,
        };
        schedule_inputs.push(abi.clone());
        buffers.push(abi);
    }
    buffers.push(MetalBufferAbi {
        id: plan.output.index() as u64,
        dtype: DType::F32,
        source_shape: plan.output_shape.clone(),
        elements: portable.extent(),
        mutable: true,
        view: None,
    });
    for buffer in &buffers {
        let logical = buffer
            .elements
            .checked_mul(DType::F32.itemsize())
            .ok_or(MetalError::Overflow)?;
        let physical = if portable.extent() != 0 && logical == 0 {
            DType::F32.itemsize()
        } else {
            logical
        };
        if physical > renderer.capabilities.max_buffer_length {
            return Err(MetalError::Unsupported(
                "portable matmul binding exceeds device buffer limit".into(),
            ));
        }
    }
    let positions = buffers
        .iter()
        .enumerate()
        .map(|(position, abi)| (abi.id, position))
        .collect::<BTreeMap<_, _>>();
    let lhs_position = positions[&(plan.lhs.index() as u64)];
    let rhs_position = positions[&(plan.rhs.index() as u64)];
    let output_position = buffers.len() - 1;
    let entry = format!("rg_metal_matmul_f32_{}", plan.cache_key);
    let mut lines = vec![
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        "#pragma clang fp contract(off)".into(),
        format!("// {METAL_PORTABLE_F32_MATMUL_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        format!("kernel void {entry}("),
    ];
    for (position, _) in buffers[..output_position].iter().enumerate() {
        lines.push(format!(
            "    device const float* b{position} [[buffer({position})]],"
        ));
    }
    lines.extend([
        format!("    device float* b{output_position} [[buffer({output_position})]],"),
        format!("    constant ulong& extent [[buffer({})]],", buffers.len()),
        "    uint gid32 [[thread_position_in_grid]]) {".into(),
        "  const ulong gid = (ulong)gid32;".into(),
        "  if (gid >= extent) return;".into(),
        "  ulong rg_q = gid;".into(),
        "  ulong rg_col = 0ul;".into(),
        "  ulong rg_row = 0ul;".into(),
    ]);
    if !plan.rhs_vector && plan.n != 0 {
        lines.push(format!(
            "  rg_col = rg_q % (ulong){}ul; rg_q /= (ulong){}ul;",
            plan.n, plan.n
        ));
    }
    if !plan.lhs_vector && plan.m != 0 {
        lines.push(format!(
            "  rg_row = rg_q % (ulong){}ul; rg_q /= (ulong){}ul;",
            plan.m, plan.m
        ));
    }
    lines.push("  const ulong rg_batch = rg_q;".into());
    for (name, axes) in [
        ("rg_lbatch", portable.lhs_batch_axes()),
        ("rg_rbatch", portable.rhs_batch_axes()),
    ] {
        lines.push(format!("  ulong {name} = 0ul;"));
        if portable.extent() != 0 {
            for axis in axes {
                lines.push(format!(
                    "  {name} += ((rg_batch / (ulong){}ul) % (ulong){}ul) * (ulong){}ul;",
                    axis.divisor, axis.dimension, axis.input_stride
                ));
            }
        }
    }
    let lhs_offset = if plan.lhs_vector {
        "rg_k".into()
    } else {
        format!(
            "((rg_lbatch * (ulong){}ul + rg_row) * (ulong){}ul + rg_k)",
            plan.m, plan.k
        )
    };
    let rhs_offset = if plan.rhs_vector {
        "rg_k".into()
    } else {
        format!(
            "((rg_rbatch * (ulong){}ul + rg_k) * (ulong){}ul + rg_col)",
            plan.k, plan.n
        )
    };
    lines.extend([
        "  float rg_acc = 0.0f;".into(),
        format!("  for (ulong rg_k = 0ul; rg_k < (ulong){}ul; ++rg_k) {{", plan.k),
        format!(
            "    const float rg_product = b{lhs_position}[{lhs_offset}] * b{rhs_position}[{rhs_offset}];"
        ),
        "    rg_acc = rg_acc + rg_product;".into(),
        "  }".into(),
        format!("  b{output_position}[gid] = rg_acc;"),
        "}".into(),
    ]);
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_PORTABLE_F32_MATMUL_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.extent(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_bitcast(
    renderer: &MetalRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let portable = crate::movement_plan::PortableBitcast::new(plan)
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    if portable.bytes() > u32::MAX as usize {
        return Err(MetalError::Unsupported(
            "portable Bitcast byte extent exceeds u32 thread indexing".into(),
        ));
    }
    let input = portable.input();
    let input_abi = MetalBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: portable.input_elements(),
        mutable: false,
        view: None,
    };
    let buffers = vec![
        input_abi.clone(),
        MetalBufferAbi {
            id: plan.output.index() as u64,
            dtype: plan.dtype,
            source_shape: plan.output_shape.clone(),
            elements: portable.output_elements(),
            mutable: true,
            view: None,
        },
    ];
    let entry = "rg_metal_portable_bitcast".to_owned();
    let stored = if portable.normalizes_bool() {
        "b0[gid] != (uchar)0"
    } else {
        "b0[gid]"
    };
    let source = [
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        format!("// {METAL_PORTABLE_BITCAST_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        format!("kernel void {entry}("),
        "    device const uchar* b0 [[buffer(0)]],".into(),
        "    device uchar* b1 [[buffer(1)]],".into(),
        "    constant ulong& extent [[buffer(2)]],".into(),
        "    uint gid [[thread_position_in_grid]]) {".into(),
        "  if ((ulong)gid >= extent) return;".into(),
        format!("  b1[gid] = (uchar)({stored});"),
        "}".into(),
    ]
    .join("\n")
        + "\n";
    let cache_key = stable_key(&(
        METAL_PORTABLE_BITCAST_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.bytes(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_dense_materialization(
    renderer: &MetalRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let portable = crate::movement_plan::PortableDenseMaterialization::new(plan)
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    if portable.elements() > u32::MAX as usize {
        return Err(MetalError::Unsupported(
            "portable dense materialization exceeds uint thread indexing".into(),
        ));
    }
    let mut buffers = portable
        .inputs()
        .iter()
        .map(|input| {
            Ok(MetalBufferAbi {
                id: input.node.index() as u64,
                dtype: input.dtype,
                source_shape: input.shape.clone(),
                elements: input.shape.numel().map_err(|_| MetalError::Overflow)?,
                mutable: false,
                view: None,
            })
        })
        .collect::<Result<Vec<_>, MetalError>>()?;
    let schedule_inputs = buffers.clone();
    buffers.push(MetalBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: portable.elements(),
        mutable: true,
        view: None,
    });
    let entry = "rg_metal_portable_dense_materialization".to_owned();
    let mut lines = vec![
        format!(
            "// {METAL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"
        ),
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        format!("kernel void {entry}("),
    ];
    for (index, _) in portable.inputs().iter().enumerate() {
        lines.push(format!(
            "    device const uchar* b{index} [[buffer({index})]],"
        ));
    }
    let output = portable.inputs().len();
    lines.extend([
        format!("    device uchar* b{output} [[buffer({output})]],"),
        format!("    constant ulong& extent [[buffer({})]],", output + 1),
        "    uint rg_gid [[thread_position_in_grid]]) {".into(),
        "  const ulong gid = (ulong)rg_gid;".into(),
        "  if (gid >= extent) return;".into(),
    ]);
    lines.extend(
        crate::portable_movement::emit_portable_dense_materialization_body(
            &portable,
            &crate::portable_movement::CLikePortableDenseDialect {
                input_address: "device",
                output_address: "device",
            },
        ),
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.elements(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_indexed_movement(
    renderer: &MetalRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let portable = crate::movement_plan::PortableIndexedMovement::new(plan).map_err(|error| {
        if matches!(error, crate::MovementPlanError::UnsupportedDType) {
            MetalError::Unsupported(
                "Metal indexed movement requires F32 values and I32 indices".into(),
            )
        } else {
            MetalError::InvalidBinding(error.to_string())
        }
    })?;
    if portable.output_elements() > u32::MAX as usize {
        return Err(MetalError::Unsupported(
            "Metal indexed movement output exceeds uint thread indexing".into(),
        ));
    }
    let mut buffers = portable
        .inputs()
        .iter()
        .map(|input| {
            Ok(MetalBufferAbi {
                id: input.node.index() as u64,
                dtype: input.dtype,
                source_shape: input.shape.clone(),
                elements: input.shape.numel().map_err(|_| MetalError::Overflow)?,
                mutable: false,
                view: None,
            })
        })
        .collect::<Result<Vec<_>, MetalError>>()?;
    let schedule_inputs = buffers.clone();
    let output_position = buffers.len();
    buffers.push(MetalBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: portable.output_elements(),
        mutable: true,
        view: None,
    });
    let transaction = super::MetalIndexedMovementAbi::new(
        output_position,
        portable.index_abi(),
        portable.axis(),
        portable.axis_extent(),
        portable.index_elements(),
    )?;
    let entry = match portable.scatter_add() {
        None => "rg_metal_gather_f32_i32",
        Some(false) => "rg_metal_scatter_f32_i32",
        Some(true) => "rg_metal_scatter_add_f32_i32",
    }
    .to_owned();
    let mut lines = vec![
        format!("// {METAL_INDEXED_MOVEMENT_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        format!("kernel void {entry}("),
    ];
    for (position, input) in portable.inputs().iter().enumerate() {
        lines.push(format!(
            "    device const {}* b{position} [[buffer({position})]],",
            metal_storage_type(input.dtype)
        ));
    }
    lines.extend([
        format!("    device float* b{output_position} [[buffer({output_position})]],"),
        format!("    constant ulong& extent [[buffer({})]],", buffers.len()),
        format!(
            "    device atomic_uint* rg_status [[buffer({})]],",
            buffers.len() + 1
        ),
        "    uint rg_gid [[thread_position_in_grid]]) {".into(),
        "  const ulong gid = (ulong)rg_gid;".into(),
        "  if (gid >= extent) return;".into(),
    ]);
    if portable.output_elements() != 0 {
        match portable.scatter_add() {
            None => emit_metal_gather_body(&portable, output_position, &mut lines),
            Some(add) => emit_metal_scatter_body(&portable, output_position, add, &mut lines),
        }
    }
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_INDEXED_MOVEMENT_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.plan(),
        &source,
        &buffers,
        &transaction,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.output_elements(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: Some(transaction),
        append_state: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn indexed_coordinate(linear: &str, divisor: usize, dimension: usize) -> String {
    format!("(({linear} / (ulong){divisor}ul) % (ulong){dimension}ul)")
}

fn emit_metal_gather_body(
    portable: &crate::movement_plan::PortableIndexedMovement<'_>,
    output_position: usize,
    lines: &mut Vec<String>,
) {
    let index = portable.index_abi();
    lines.push(format!("  const int rg_selected = b{index}[gid];"));
    lines.push(format!(
        "  if (rg_selected < 0 || (ulong)rg_selected >= (ulong){}ul) {{",
        portable.axis_extent()
    ));
    lines.push("    atomic_fetch_min_explicit(rg_status, rg_gid, memory_order_relaxed);".into());
    lines.push("    return;".into());
    lines.push("  }".into());
    let source = portable
        .axes()
        .iter()
        .map(|axis| {
            let coordinate = if axis.axis == portable.axis() {
                "(ulong)rg_selected".into()
            } else {
                indexed_coordinate("gid", axis.index_divisor, axis.index_dimension)
            };
            format!("({coordinate} * (ulong){}ul)", axis.data_stride)
        })
        .collect::<Vec<_>>()
        .join(" + ");
    lines.push(format!(
        "  b{output_position}[gid] = b0[{}];",
        if source.is_empty() { "0ul" } else { &source }
    ));
}

fn emit_metal_scatter_body(
    portable: &crate::movement_plan::PortableIndexedMovement<'_>,
    output_position: usize,
    add: bool,
    lines: &mut Vec<String>,
) {
    lines.push("  float rg_value = b0[gid];".into());
    if portable.index_elements() != 0 {
        let index = portable.index_abi();
        let update = portable
            .update_abi()
            .expect("checked Scatter has an update operand");
        let indexed_axis = portable
            .axes()
            .iter()
            .find(|axis| axis.axis == portable.axis())
            .expect("checked Scatter axis");
        let destination_coordinates = portable
            .axes()
            .iter()
            .map(|axis| {
                (
                    axis.axis,
                    indexed_coordinate("gid", axis.data_divisor, axis.data_dimension),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let covered = portable
            .axes()
            .iter()
            .filter(|axis| axis.axis != portable.axis())
            .map(|axis| {
                format!(
                    "({} < (ulong){}ul)",
                    destination_coordinates[&axis.axis], axis.index_dimension
                )
            })
            .collect::<Vec<_>>()
            .join(" && ");
        let index_base = portable
            .axes()
            .iter()
            .filter(|axis| axis.axis != portable.axis())
            .map(|axis| {
                format!(
                    "({} * (ulong){}ul)",
                    destination_coordinates[&axis.axis], axis.index_divisor
                )
            })
            .collect::<Vec<_>>()
            .join(" + ");
        lines.push(format!(
            "  if ({}) {{",
            if covered.is_empty() { "true" } else { &covered }
        ));
        lines.push(format!(
            "    const ulong rg_index_base = {};",
            if index_base.is_empty() {
                "0ul"
            } else {
                &index_base
            }
        ));
        lines.push(format!(
            "    for (ulong rg_axis_index = 0ul; rg_axis_index < (ulong){}ul; ++rg_axis_index) {{",
            indexed_axis.index_dimension
        ));
        lines.push(format!(
            "      const ulong rg_index = rg_index_base + rg_axis_index * (ulong){}ul;",
            indexed_axis.index_divisor
        ));
        lines.push(format!("      const int rg_selected = b{index}[rg_index];"));
        lines.push(format!(
            "      if (rg_selected < 0 || (ulong)rg_selected >= (ulong){}ul) {{",
            portable.axis_extent()
        ));
        lines.push(
            "        atomic_fetch_min_explicit(rg_status, (uint)rg_index, memory_order_relaxed);"
                .into(),
        );
        lines.push("        continue;".into());
        lines.push("      }".into());
        let update_offset = portable
            .axes()
            .iter()
            .map(|axis| {
                let coordinate = if axis.axis == portable.axis() {
                    "rg_axis_index"
                } else {
                    &destination_coordinates[&axis.axis]
                };
                format!(
                    "({coordinate} * (ulong){}ul)",
                    axis.update_stride.expect("checked update stride")
                )
            })
            .collect::<Vec<_>>()
            .join(" + ");
        let operator = if add { "+=" } else { "=" };
        lines.push(format!(
            "      if ({} == (ulong)rg_selected) rg_value {operator} b{update}[{}];",
            destination_coordinates[&portable.axis()],
            if update_offset.is_empty() {
                "0ul"
            } else {
                &update_offset
            }
        ));
        lines.push("    }".into());
        lines.push("  }".into());
    }
    lines.push(format!("  b{output_position}[gid] = rg_value;"));
}

fn render_raw_copy(
    renderer: &MetalRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let copy = plan
        .raw_copy()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
        .ok_or_else(|| {
            MetalError::Unsupported(
                "only raw AffineCopy and Contiguous have Metal movement lowering".into(),
            )
        })?;
    let input = copy.input();
    let extent = copy.elements();
    if extent > u32::MAX as usize {
        return Err(MetalError::Unsupported(
            "raw-copy Metal extent exceeds u32 thread indexing".into(),
        ));
    }
    let width = copy.width();
    let raw_type = match width {
        1 => "uchar",
        2 => "ushort",
        4 => "uint",
        8 => "ulong",
        _ => {
            return Err(MetalError::Unsupported(format!(
                "raw-copy Metal storage width {width}"
            )));
        }
    };
    debug_assert_eq!(copy.bytes(), extent * width);
    let input_abi = MetalBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: copy.input_elements(),
        mutable: false,
        view: None,
    };
    let output_abi = MetalBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    let buffers = vec![input_abi.clone(), output_abi];
    let entry = format!("rg_metal_raw_copy_w{width}");
    let mut lines = vec![
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        format!("// {METAL_RAW_COPY_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        format!("kernel void {entry}("),
        format!("    device const {raw_type}* b0 [[buffer(0)]],"),
        format!("    device {raw_type}* b1 [[buffer(1)]],"),
        "    constant ulong& extent [[buffer(2)]],".into(),
        "    uint gid [[thread_position_in_grid]]) {".into(),
        "  if ((ulong)gid >= extent) return;".into(),
    ];
    let source_index = if let Some(address) = copy
        .address()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
    {
        lines.push(format!("  ulong rg_source = (ulong){}ul;", address.offset));
        for axis in address.axes {
            let output_axis = axis.output_axis;
            lines.push(format!(
                "  ulong rg_axis_{output_axis} = ((ulong)gid / (ulong){}ul) % (ulong){}ul;",
                axis.divisor, axis.dimension
            ));
            if axis.reversed {
                lines.push(format!(
                    "  rg_axis_{output_axis} = (ulong){}ul - rg_axis_{output_axis};",
                    axis.dimension - 1
                ));
            }
            lines.push(format!(
                "  rg_source += rg_axis_{output_axis} * (ulong){}ul;",
                axis.stride
            ));
        }
        "rg_source"
    } else {
        "gid"
    };
    lines.push(format!("  b1[gid] = b0[{source_index}];"));
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_RAW_COPY_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        copy.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent,
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_static_positions(
    renderer: &MetalRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedMetal, MetalError> {
    root.validate()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
    let placement = plan
        .static_position_write()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
        .ok_or_else(|| MetalError::InvalidBinding("missing static-position projection".into()))?;
    let input = placement.input();
    let extent = placement.elements();
    if extent > u32::MAX as usize {
        return Err(MetalError::Unsupported(
            "static-position Metal extent exceeds u32 thread indexing".into(),
        ));
    }
    let width = placement.width();
    let raw_type = match width {
        1 => "uchar",
        2 => "ushort",
        4 => "uint",
        8 => "ulong",
        _ => {
            return Err(MetalError::Unsupported(format!(
                "static-position Metal storage width {width}"
            )));
        }
    };
    debug_assert_eq!(placement.bytes(), extent * width);
    let input_abi = MetalBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: placement.input_elements(),
        mutable: false,
        view: None,
    };
    let output_abi = MetalBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    let buffers = vec![input_abi.clone(), output_abi];
    let entry = format!("rg_metal_static_position_w{width}");
    let mut lines = vec![
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        format!("// {METAL_STATIC_POSITION_RENDERER_VERSION} ABI {METAL_ABI_VERSION}"),
        format!("kernel void {entry}("),
        format!("    device const {raw_type}* b0 [[buffer(0)]],"),
        format!("    device {raw_type}* b1 [[buffer(1)]],"),
        "    constant ulong& extent [[buffer(2)]],".into(),
        "    uint gid32 [[thread_position_in_grid]]) {".into(),
        "  ulong gid = (ulong)gid32;".into(),
        "  if (gid >= extent) return;".into(),
        "  bool rg_mapped = false;".into(),
        "  ulong rg_source = 0ul;".into(),
    ];
    if placement.has_source() {
        lines.push("  rg_mapped = true;".into());
        for axis in placement.axes() {
            let name = axis.output_axis;
            lines.push(format!(
                "  ulong rg_coordinate_{name} = (gid / (ulong){}ul) % (ulong){}ul;",
                axis.output_divisor, axis.output_dimension
            ));
            lines.push(format!(
                "  ulong rg_delta_{name} = rg_coordinate_{name} >= (ulong){}ul ? rg_coordinate_{name} - (ulong){}ul : 0ul;",
                axis.first, axis.first
            ));
            lines.push(format!(
                "  ulong rg_quotient_{name} = rg_delta_{name} / (ulong){}ul;",
                axis.spacing
            ));
            lines.push(format!(
                "  if (rg_coordinate_{name} < (ulong){}ul || rg_delta_{name} % (ulong){}ul != 0ul || rg_quotient_{name} >= (ulong){}ul) rg_mapped = false;",
                axis.first, axis.spacing, axis.source_dimension
            ));
            lines.push(format!(
                "  ulong rg_source_axis_{name} = rg_quotient_{name} < (ulong){}ul ? {} : 0ul;",
                axis.source_dimension,
                if axis.reversed {
                    format!(
                        "(ulong){}ul - rg_quotient_{name}",
                        axis.source_dimension - 1
                    )
                } else {
                    format!("rg_quotient_{name}")
                }
            ));
            lines.push(format!(
                "  rg_source += rg_source_axis_{name} * (ulong){}ul;",
                axis.source_stride
            ));
        }
    }
    lines.push(format!("  {raw_type} rg_value = ({raw_type})0;"));
    lines.push("  if (rg_mapped) rg_value = b0[rg_source];".into());
    lines.push("  b1[gid] = rg_value;".into());
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        METAL_STATIC_POSITION_RENDERER_VERSION,
        METAL_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        placement.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedMetal {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent,
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn emit_metal_reduction(
    reduction: &crate::reduction_native::NativeReductionKernel<'_>,
    output_position: usize,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
) -> Result<(), MetalError> {
    let plan = &reduction.plan;
    let producer = reduction.producer;
    let accumulator_type = metal_storage_type(plan.accumulator_dtype);
    let identity = metal_reduction_literal(plan.accumulator_dtype, plan.identity())?;
    lines.push(format!("  {accumulator_type} rg_acc = {identity};"));
    let reduction_len = plan.reduction_len();
    if reduction_len != 0 {
        lines.push(format!(
            "  for (ulong rg_r = 0ul; rg_r < {reduction_len}ul; ++rg_r) {{"
        ));
        let source_index =
            crate::reduction_native::index_expression(&plan.geometry, "(ulong)gid", "rg_r", "ul");
        lines.push(format!("    const ulong rg_src = {source_index};"));
        let candidate = emit_expr(producer, ids, source_map, lines, "rg_src")?;
        let candidate = MetalScalarDialect
            .cast(plan.source_dtype, plan.accumulator_dtype, &candidate)
            .map_err(MetalError::Unsupported)?;
        if plan.is_singleton_identity() {
            lines.push(format!("    rg_acc = ({accumulator_type})({candidate});"));
        } else {
            match plan.kind {
                crate::ReduceKind::Sum | crate::ReduceKind::Mean => lines.push(format!(
                    "    rg_acc = {};",
                    if plan.accumulator_dtype == DType::Bool {
                        format!("(uchar)(rg_acc || ({candidate}))")
                    } else {
                        metal_reduction_arithmetic(
                            plan.accumulator_dtype,
                            "rg_acc",
                            &candidate,
                            false,
                        )
                    }
                )),
                crate::ReduceKind::Product => lines.push(format!(
                    "    rg_acc = {};",
                    if plan.accumulator_dtype == DType::Bool {
                        format!("(uchar)(rg_acc && ({candidate}))")
                    } else {
                        metal_reduction_arithmetic(
                            plan.accumulator_dtype,
                            "rg_acc",
                            &candidate,
                            true,
                        )
                    }
                )),
                crate::ReduceKind::Max | crate::ReduceKind::Min => {
                    if plan.accumulator_dtype == DType::Bool {
                        lines.push(format!(
                            "    rg_acc = (uchar)(rg_acc {} ({candidate}));",
                            if plan.kind == crate::ReduceKind::Max {
                                "||"
                            } else {
                                "&&"
                            }
                        ));
                    } else {
                        let comparison = if plan.kind == crate::ReduceKind::Max {
                            ">"
                        } else {
                            "<"
                        };
                        lines.push(format!(
                            "    if (({candidate}) {comparison} rg_acc) rg_acc = ({candidate});"
                        ));
                    }
                }
                crate::ReduceKind::Any => {
                    lines.push(format!("    rg_acc = (uchar)(rg_acc || ({candidate}));"));
                }
                crate::ReduceKind::All => {
                    lines.push(format!("    rg_acc = (uchar)(rg_acc && ({candidate}));"));
                }
            }
        }
        lines.push("  }".into());
    }
    if plan.kind == crate::ReduceKind::Mean {
        if reduction_len == 0 {
            lines.push("  rg_acc = as_type<float>(0x7fc00000u);".into());
        } else {
            let divisor = metal_reduction_literal(
                plan.accumulator_dtype,
                plan.mean_divisor()
                    .expect("nonempty validated Mean divisor"),
            )?;
            lines.push(format!("  rg_acc = (float)(rg_acc / {divisor});"));
        }
    }
    let finalized = MetalScalarDialect
        .cast(plan.accumulator_dtype, plan.output_dtype, "rg_acc")
        .map_err(MetalError::Unsupported)?;
    let committed = MetalScalarDialect
        .cast(plan.output_dtype, plan.output_dtype, &finalized)
        .map_err(MetalError::Unsupported)?;
    let stored = if reduction.has_epilogue() {
        emit_expr_with_substitution(
            reduction.epilogue_root,
            ids,
            source_map,
            lines,
            "(ulong)gid",
            Some((reduction.finalize, committed.as_str())),
        )?
    } else {
        committed
    };
    lines.push(format!("  b{output_position}[gid] = {stored};"));
    Ok(())
}

fn metal_reduction_literal(dtype: DType, value: crate::Scalar) -> Result<String, MetalError> {
    Ok(match value {
        crate::Scalar::Bool(value) => format!("(uchar){}u", u8::from(value)),
        crate::Scalar::I(value) if dtype == DType::I32 => {
            format!("as_type<int>(0x{:08x}u)", value as u32)
        }
        crate::Scalar::U(value) => format!("(uint){value}u"),
        crate::Scalar::F(value) => {
            format!("as_type<float>(0x{:08x}u)", (value as f32).to_bits())
        }
        _ => {
            return Err(MetalError::Unsupported(
                "Metal reduction identity is outside the exact storage subset".into(),
            ));
        }
    })
}

fn metal_reduction_arithmetic(dtype: DType, lhs: &str, rhs: &str, product: bool) -> String {
    let operator = if product { "*" } else { "+" };
    if dtype == DType::I32 {
        format!("as_type<int>(as_type<uint>({lhs}) {operator} as_type<uint>({rhs}))")
    } else {
        format!("(({lhs}) {operator} ({rhs}))")
    }
}

fn supported_storage(dtype: DType) -> Result<(), MetalError> {
    match dtype {
        DType::F32 | DType::Bool | DType::I32 | DType::U32 => Ok(()),
        _ => Err(MetalError::Unsupported(format!(
            "dtype {dtype:?} is outside the exact Metal static subset"
        ))),
    }
}

pub(super) fn metal_storage_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::Bool => "uchar",
        DType::I32 => "int",
        DType::U32 => "uint",
        _ => unreachable!("validated Metal storage"),
    }
}

pub(super) struct MetalScalarDialect;

impl dialect_seal::Sealed for MetalScalarDialect {}

impl ScalarLaneDialect for MetalScalarDialect {
    fn name(&self) -> &'static str {
        "Metal"
    }

    fn supports_value(&self, dtype: DType) -> bool {
        supported_storage(dtype).is_ok()
    }

    fn cast(&self, source: DType, target: DType, value: &str) -> Result<String, String> {
        Ok(match (source, target) {
            (DType::F32, DType::F32)
            | (DType::Bool, DType::Bool)
            | (DType::I32, DType::I32)
            | (DType::U32, DType::U32) => value.into(),
            (DType::Bool, DType::F32) => format!("(float)({value} != 0)"),
            (DType::F32, DType::Bool) => format!("(uchar)({value} != 0.0f)"),
            (DType::Bool, DType::I32) => format!("(int)({value})"),
            (DType::Bool, DType::U32) => format!("(uint)({value})"),
            (DType::I32 | DType::U32, DType::Bool) => {
                format!("(uchar)(({value}) != 0)")
            }
            (DType::I32, DType::U32) => format!("as_type<uint>({value})"),
            (DType::U32, DType::I32) => format!("as_type<int>({value})"),
            (DType::I32 | DType::U32, DType::F32) => format!("(float)({value})"),
            _ => return Err("cast is outside the exact Metal subset".into()),
        })
    }

    fn finish_float(&self, dtype: DType, value: String) -> Result<String, String> {
        if dtype == DType::F32 {
            Ok(value)
        } else {
            Err("Metal scalar float expression requires F32".into())
        }
    }

    fn signed_infix(
        &self,
        dtype: DType,
        operator: &'static str,
        lhs: &str,
        rhs: &str,
    ) -> Result<String, String> {
        if dtype == DType::I32 {
            Ok(format!(
                "as_type<int>(as_type<uint>({lhs}) {operator} as_type<uint>({rhs}))"
            ))
        } else {
            Err("Metal signed wrapping requires I32".into())
        }
    }

    fn signed_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        if dtype == DType::I32 {
            Ok(format!("as_type<int>((uint)0u - as_type<uint>({value}))"))
        } else {
            Err("Metal signed negation requires I32".into())
        }
    }

    fn unsigned_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        if dtype == DType::U32 {
            Ok(format!("((uint)0u - ({value}))"))
        } else {
            Err("Metal unsigned negation requires U32".into())
        }
    }

    fn signed_abs(&self, dtype: DType, value: &str) -> Result<String, String> {
        if dtype == DType::I32 {
            Ok(format!(
                "select(as_type<int>((uint)0u - as_type<uint>({value})), ({value}), ({value}) >= 0)"
            ))
        } else {
            Err("Metal signed absolute value requires I32".into())
        }
    }

    fn float_abs(&self, value: &str) -> String {
        format!("fabs({value})")
    }

    fn bool_value(&self, expression: String) -> String {
        format!("(uchar)({expression})")
    }

    fn select(&self, condition: &str, on_true: &str, on_false: &str) -> String {
        format!("(({condition}) ? ({on_true}) : ({on_false}))")
    }

    fn call_intrinsic(&self, canonical_name: &'static str, value: &str) -> String {
        if canonical_name == "sin" {
            format!("precise::sin({value})")
        } else {
            format!("{canonical_name}({value})")
        }
    }

    fn float_one(&self, dtype: DType) -> Result<&'static str, String> {
        if dtype == DType::F32 {
            Ok("1.0f")
        } else {
            Err("Metal reciprocal requires F32".into())
        }
    }
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
) -> Result<String, MetalError> {
    emit_expr_with_substitution(node, ids, source_map, lines, linear, None)
}

fn emit_expr_with_substitution(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
    substitution: Option<(&UOp, &str)>,
) -> Result<String, MetalError> {
    if let Some((target, value)) = substitution
        && node.shares_node_with(target)
    {
        return Ok(value.into());
    }
    let map_id = source_map.len();
    source_map.insert(map_id, lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| MetalError::Unsupported(format!("untyped {:?}", node.operation())))?
        .scalar;
    supported_storage(dtype)?;
    let child = |position: usize,
                 source_map: &mut BTreeMap<usize, usize>,
                 lines: &mut Vec<String>| {
        node.sources()
            .get(position)
            .ok_or_else(|| MetalError::Unsupported("missing expression operand".into()))
            .and_then(|source| {
                emit_expr_with_substitution(source, ids, source_map, lines, linear, substitution)
            })
    };
    match node.operation() {
        Operation::Const(value) => match value {
            LiteralValue::Scalar {
                dtype: DType::F32,
                bits,
            } => Ok(format!("as_type<float>((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::Bool,
                bits,
            } if *bits <= 1 => Ok(format!("(uchar){bits}u")),
            LiteralValue::Scalar {
                dtype: DType::I32,
                bits,
            } => Ok(format!("as_type<int>((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::U32,
                bits,
            } => Ok(format!("(uint)0x{:08x}u", *bits as u32)),
            _ => Err(MetalError::Unsupported(
                "invalid Metal scalar literal".into(),
            )),
        },
        Operation::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| MetalError::Unsupported("load has no index".into()))?;
            let (buffer, input_shape, output_shape, view) = match index.operation() {
                Operation::Index(IndexValue::Buffer { buffer, .. })
                    if crate::projected_index::ProjectedIndexPlan::is_projected(index) =>
                {
                    let plan = crate::projected_index::ProjectedIndexPlan::from_index(index)
                        .map_err(|_| MetalError::Unsupported("invalid projected index".into()))?;
                    let access = crate::projected_index::render_infix_projected_access(
                        &plan,
                        format!("((long)({linear}))"),
                        |value| {
                            Ok(if value == i64::MIN {
                                "((-9223372036854775807l) - 1l)".into()
                            } else {
                                format!("((long){value}l)")
                            })
                        },
                        |value| if value { "1" } else { "0" }.into(),
                    )
                    .map_err(|_| MetalError::Unsupported("invalid projected index".into()))?;
                    let position = ids.get(buffer).ok_or_else(|| {
                        MetalError::InvalidBinding("load buffer absent from ABI".into())
                    })?;
                    let raw = format!("b{position}[{}]", access.offset);
                    let value = if dtype == DType::Bool {
                        format!("(({raw}) != 0)")
                    } else {
                        raw
                    };
                    return Ok(access
                        .predicate
                        .map(|predicate| {
                            let zero = match dtype {
                                DType::Bool => "false",
                                DType::F32 => "0.0f",
                                DType::I32 => "((int)0)",
                                DType::U32 => "((uint)0u)",
                                _ => unreachable!("validated Metal storage"),
                            };
                            format!("(({predicate}) ? ({value}) : ({zero}))")
                        })
                        .unwrap_or(value));
                }
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    input_shape,
                    output_shape,
                    ..
                }) => (*buffer, input_shape, output_shape, None),
                Operation::Index(IndexValue::View {
                    buffer,
                    input_shape,
                    output_shape,
                    view,
                    ..
                }) => (*buffer, input_shape, output_shape, Some(view)),
                _ => {
                    return Err(MetalError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            let position = ids
                .get(&buffer)
                .ok_or_else(|| MetalError::InvalidBinding("load buffer absent from ABI".into()))?;
            let logical = broadcast_offset(input_shape, output_shape, linear)?;
            let offset = match view {
                Some(view) => MetalViewAccess::new(view)?.expression(&logical),
                None => logical,
            };
            Ok(format!("b{position}[{offset}]"))
        }
        other => {
            let mut sources = Vec::with_capacity(node.sources().len());
            for slot in 0..node.sources().len() {
                sources.push(child(slot, source_map, lines)?);
            }
            let instruction = project_scalar_lane(node, &sources)
                .map_err(MetalError::Unsupported)?
                .ok_or_else(|| MetalError::Unsupported(format!("{other:?}")))?;
            emit_scalar_lane(&MetalScalarDialect, &instruction).map_err(MetalError::Unsupported)
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct MetalViewAccess {
    source_shape: Shape,
    logical_shape: Shape,
    strides: Vec<i64>,
    offset: i64,
}

impl MetalViewAccess {
    pub(super) fn new(view: &AffineView) -> Result<Self, MetalError> {
        if view.logical_shape.rank() != view.strides.len() {
            return Err(MetalError::Unsupported("view rank/stride mismatch".into()));
        }
        view.validate_read()
            .map_err(|_| MetalError::Unsupported("invalid signed affine read map".into()))?;
        Ok(Self {
            source_shape: view.source_shape.clone(),
            logical_shape: view.logical_shape.clone(),
            strides: view.strides.clone(),
            offset: view.offset,
        })
    }

    pub(super) fn expression(&self, logical: &str) -> String {
        if self.offset >= 0 && self.strides.iter().all(|stride| *stride >= 0) {
            return self.unsigned_expression(logical);
        }
        self.signed_expression(logical)
    }

    fn unsigned_expression(&self, logical: &str) -> String {
        if self.logical_shape.numel().ok() == Some(1) {
            return format!("{}ul", self.offset);
        }
        let logical_strides = self.logical_shape.contiguous_strides();
        let mut terms = Vec::new();
        if self.offset != 0 {
            terms.push(format!("{}ul", self.offset));
        }
        for ((dim, stride), logical_stride) in self
            .logical_shape
            .dims()
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .zip(logical_strides)
        {
            if dim > 1 && stride != 0 {
                terms.push(format!(
                    "((({logical}) / {logical_stride}ul) % {dim}ul) * {stride}ul"
                ));
            }
        }
        if terms.is_empty() {
            "0ul".into()
        } else {
            format!("({})", terms.join(" + "))
        }
    }

    fn signed_expression(&self, logical: &str) -> String {
        let logical_strides = self.logical_shape.contiguous_strides();
        let mut terms = vec![format!("{}l", self.offset)];
        for ((dim, stride), logical_stride) in self
            .logical_shape
            .dims()
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .zip(logical_strides)
        {
            if dim > 1 && stride != 0 {
                terms.push(format!(
                    "((((long)({logical}) / {logical_stride}l) % {dim}l) * {stride}l)"
                ));
            }
        }
        // `new` proves every lane is in the physical source range before the
        // generated `ulong` index is formed.
        format!("((ulong)({}))", terms.join(" + "))
    }
}

pub(super) fn broadcast_offset(
    input: &Shape,
    output: &Shape,
    linear: &str,
) -> Result<String, MetalError> {
    if input.rank() > output.rank() {
        return Err(MetalError::Unsupported(
            "input rank exceeds output rank".into(),
        ));
    }
    if input.rank() == 0 {
        return Ok("0ul".into());
    }
    let input_strides = input.contiguous_strides();
    let output_strides = output.contiguous_strides();
    let pad = output.rank() - input.rank();
    let mut terms = Vec::new();
    for axis in 0..input.rank() {
        let dim = input.dims()[axis];
        let output_dim = output.dims()[pad + axis];
        if dim != 1 && dim != output_dim {
            return Err(MetalError::Unsupported("invalid broadcast metadata".into()));
        }
        if dim != 1 {
            terms.push(format!(
                "(({linear} / {}ul) % {}ul) * {}ul",
                output_strides[pad + axis],
                dim,
                input_strides[axis]
            ));
        }
    }
    Ok(if terms.is_empty() {
        "0ul".into()
    } else {
        terms.join(" + ")
    })
}

fn stable_key(value: &impl Hash) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod affine_view_tests {
    use super::*;
    #[test]
    fn signed_affine_view_lowers_without_unsigned_reinterpretation() {
        let view = AffineView {
            source_shape: Shape::from([4]),
            logical_shape: Shape::from([4]),
            strides: vec![-1],
            offset: 3,
        };
        let access = MetalViewAccess::new(&view).unwrap();
        assert!(access.expression("gid").contains("(long)(gid)"));
    }
}
