//! Pure deterministic OpenCL C lowering for a deliberately small scalar/reduction UOp subset.
use super::{
    OpenClCapabilities, OpenClError,
    guard::{emit_transactional, emit_transactional_reduction},
    narrow,
    reduction::{
        input_offset_expression, required_capabilities as reduction_capabilities, validate_dtype,
    },
    transaction::{GuardedIntegerOp, OpenClGuardDomain, OpenClTransactionAbi},
    view::OpenClViewAccess,
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

pub const OPENCL_RENDERER_VERSION: &str = "rustgrad-opencl-static-v9";
pub const OPENCL_RAW_COPY_RENDERER_VERSION: &str = "rustgrad-opencl-raw-copy-v1";
pub const OPENCL_PORTABLE_BITCAST_RENDERER_VERSION: &str = "rustgrad-opencl-portable-bitcast-v1";
pub const OPENCL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION: &str =
    "rustgrad-opencl-portable-dense-materialization-v1";
pub const OPENCL_STATIC_POSITION_RENDERER_VERSION: &str = "rustgrad-opencl-static-position-v1";
pub const OPENCL_PORTABLE_F32_MATMUL_RENDERER_VERSION: &str =
    "rustgrad-opencl-portable-f32-matmul-v1";
pub const OPENCL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION: &str =
    "rustgrad-opencl-portable-prefix-scan-v1";
pub const OPENCL_PORTABLE_SORT_RENDERER_VERSION: &str = "rustgrad-opencl-portable-sort-v1";
pub const OPENCL_PORTABLE_THREEFRY_RENDERER_VERSION: &str = "rustgrad-opencl-portable-threefry-v1";
pub const OPENCL_ABI_VERSION: u32 = 4;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenClBufferAbi {
    pub id: u64,
    pub dtype: DType,
    pub source_shape: Shape,
    pub elements: usize,
    pub mutable: bool,
    pub view: Option<AffineView>,
}

#[derive(Clone, Debug)]
pub struct RenderedOpenCl {
    pub source: String,
    pub source_map: BTreeMap<usize, usize>,
    pub buffers: Vec<OpenClBufferAbi>,
    pub extent: usize,
    pub entry: String,
    pub cache_key: String,
    pub required_capabilities: OpenClCapabilities,
    pub transaction: Option<OpenClTransactionAbi>,
    pub(crate) schedule_inputs: Vec<OpenClBufferAbi>,
    pub(crate) semantic_program: Arc<super::dispatch::KernelSemanticProgram>,
}

impl RenderedOpenCl {
    /// Validates the schedule-owned first-use order against the pointer ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), OpenClError> {
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Matmul(value) = root.operation()
        {
            crate::matmul::PortableF32Matmul::new(value)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::PrefixScan(value) = root.operation()
        {
            let portable = crate::prefix_scan_native::PortablePrefixScan::new(value)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
            crate::runtime::static_schedule::validate_portable_prefix_scan_bindings(
                &portable, bindings,
            )
            .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Sort(value) = root.operation()
        {
            crate::portable_sort::PortableSortPair::new(value)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Threefry(value) = root.operation()
        {
            crate::portable_threefry::PortableThreefry::new(value)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Movement(MovementValue::Plan(plan)) = root.operation()
            && matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. })
        {
            crate::movement_plan::PortableBitcast::new(plan)
                .and_then(|portable| portable.validate_schedule_bindings(bindings))
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
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
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
        }
        if bindings.len() != self.schedule_inputs.len() {
            return Err(OpenClError::InvalidBinding(
                "schedule/OpenCL input count mismatch".into(),
            ));
        }
        for (index, (binding, expected)) in bindings.iter().zip(&self.schedule_inputs).enumerate() {
            if binding.abi_index != index
                || binding.desc.id != expected.id
                || binding.desc.dtype != expected.dtype
                || binding.desc.shape != expected.source_shape
                || binding.desc.view != expected.view
                || binding.desc.bytes
                    != expected
                        .elements
                        .checked_mul(expected.dtype.itemsize())
                        .ok_or(OpenClError::Overflow)?
            {
                return Err(OpenClError::InvalidBinding(format!(
                    "schedule binding {index} mismatches OpenCL ABI"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpenClRenderer {
    pub local_size: usize,
    pub capabilities: OpenClCapabilities,
}

impl Default for OpenClRenderer {
    fn default() -> Self {
        Self {
            local_size: 64,
            capabilities: OpenClCapabilities::CORE_32,
        }
    }
}

impl OpenClRenderer {
    pub fn new(local_size: usize) -> Result<Self, OpenClError> {
        if local_size == 0 {
            return Err(OpenClError::InvalidArgument("zero local size"));
        }
        Ok(Self {
            local_size,
            capabilities: OpenClCapabilities::CORE_32,
        })
    }

    pub fn with_capabilities(
        local_size: usize,
        capabilities: OpenClCapabilities,
    ) -> Result<Self, OpenClError> {
        if local_size == 0 {
            return Err(OpenClError::InvalidArgument("zero local size"));
        }
        Ok(Self {
            local_size,
            capabilities,
        })
    }

    pub fn render(&self, root: &UOp) -> Result<RenderedOpenCl, OpenClError> {
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
                MovementValue::Plan(plan) => render_raw_copy(self, root, plan),
                MovementValue::QuantizedRowGather(_) => Err(OpenClError::Unsupported(
                    "quantized movement is outside OpenCL contiguous-copy lowering".into(),
                )),
            };
        }
        if let Operation::Random(plan) = root.operation() {
            return super::random::render(self, plan);
        }
        if matches!(root.operation(), Operation::TensorGuard(_)) {
            return Err(OpenClError::Unsupported(
                "guards are outside OpenCL lowering".into(),
            ));
        }
        root.validate()
            .map_err(|error| OpenClError::Unsupported(error.to_string()))?;
        let nodes = root
            .topological()
            .map_err(|error| OpenClError::Unsupported(error.to_string()))?;
        let uses_f16 = nodes
            .iter()
            .any(|node| node.ty().is_some_and(|ty| ty.scalar == DType::F16));
        let uses_bf16 = nodes
            .iter()
            .any(|node| node.ty().is_some_and(|ty| ty.scalar == DType::BF16));
        if nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::Barrier | Operation::If | Operation::EndIf
            )
        }) {
            return Err(OpenClError::Unsupported(
                "effects and barriers are outside the OpenCL static subset".into(),
            ));
        }
        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.operation(), Operation::Store))
            .ok_or_else(|| OpenClError::Unsupported("sink has no store".into()))?;
        let output_index = store
            .sources()
            .first()
            .ok_or_else(|| OpenClError::Unsupported("store has no index".into()))?;
        let Operation::Index(IndexValue::Buffer {
            buffer: output_id,
            elements: extent,
            input_shape: output_shape,
            output_shape: store_shape,
            addressing: crate::IndexAddressing::Broadcast,
        }) = output_index.operation()
        else {
            return Err(OpenClError::Unsupported(
                "output requires a contiguous BufferIndex".into(),
            ));
        };
        if output_shape != store_shape {
            return Err(OpenClError::Unsupported(
                "non-contiguous output addressing".into(),
            ));
        }
        let output_dtype = output_index
            .ty()
            .ok_or_else(|| OpenClError::Unsupported("untyped output index".into()))?
            .scalar;
        let store_value = store
            .sources()
            .get(1)
            .ok_or_else(|| OpenClError::Unsupported("store has no value".into()))?;
        let preserves_raw_predicated_narrow = narrow::is_narrow(output_dtype)
            && crate::projected_index::ProjectedIndexPlan::from_direct_predicated_load(store_value)
                .map_err(|_| OpenClError::Unsupported("invalid predicated narrow load".into()))?
                .is_some();
        supported_kernel_storage(
            output_dtype,
            self.capabilities,
            preserves_raw_predicated_narrow,
        )?;

        let mut inventory = BTreeMap::<u64, OpenClBufferAbi>::new();
        for node in &nodes {
            let (buffer, source_shape, elements, view) = match node.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements,
                    input_shape,
                    ..
                }) => (*buffer, input_shape.clone(), *elements, None),
                Operation::Index(IndexValue::View { buffer, view, .. }) => {
                    let access = OpenClViewAccess::new(
                        view,
                        node.ty()
                            .ok_or_else(|| OpenClError::Unsupported("untyped view index".into()))?
                            .scalar,
                    )?;
                    let elements = access
                        .source_shape
                        .numel()
                        .map_err(|_| OpenClError::Overflow)?;
                    (*buffer, access.source_shape, elements, Some(view.clone()))
                }
                _ => continue,
            };
            let dtype = node
                .ty()
                .ok_or_else(|| OpenClError::Unsupported("untyped buffer index".into()))?
                .scalar;
            supported_kernel_storage(dtype, self.capabilities, preserves_raw_predicated_narrow)?;
            let abi = OpenClBufferAbi {
                id: buffer,
                dtype,
                source_shape,
                elements,
                mutable: buffer == *output_id,
                view,
            };
            if let Some(old) = inventory.insert(buffer, abi.clone())
                && old != abi
            {
                return Err(OpenClError::InvalidBinding(format!(
                    "buffer {buffer} has conflicting ABI metadata"
                )));
            }
        }

        let reduction = crate::reduction_native::NativeReductionKernel::from_store(store)
            .map_err(|reason| OpenClError::Unsupported(reason.into()))?;
        let mut schedule_inputs = Vec::new();
        let mut seen = BTreeSet::new();
        for node in &nodes {
            if !matches!(node.operation(), Operation::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::InvalidBinding("load lacks index".into()))?;
            let buffer = match index.operation() {
                Operation::Index(IndexValue::Buffer { buffer, .. })
                | Operation::Index(IndexValue::View { buffer, .. }) => *buffer,
                _ => {
                    return Err(OpenClError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            if seen.insert(buffer) {
                schedule_inputs.push(
                    inventory
                        .get(&buffer)
                        .ok_or_else(|| OpenClError::InvalidBinding("load ABI missing".into()))?
                        .clone(),
                );
            }
        }
        let zero_epilogue_buffers = reduction
            .as_ref()
            .filter(|reduction| reduction.plan.reduction_len() == 0)
            .map(|reduction| {
                fn collect(node: &UOp, finalize: &UOp, buffers: &mut BTreeSet<u64>) {
                    if node.shares_node_with(finalize) {
                        return;
                    }
                    if let Operation::Index(
                        IndexValue::Buffer { buffer, .. } | IndexValue::View { buffer, .. },
                    ) = node.operation()
                    {
                        buffers.insert(*buffer);
                    }
                    for source in node.sources() {
                        collect(source, finalize, buffers);
                    }
                }
                let mut buffers = BTreeSet::new();
                collect(reduction.epilogue_root, reduction.finalize, &mut buffers);
                buffers
            });
        let mut buffers = schedule_inputs
            .iter()
            .filter(|buffer| {
                zero_epilogue_buffers
                    .as_ref()
                    .is_none_or(|epilogue| epilogue.contains(&buffer.id))
            })
            .cloned()
            .collect::<Vec<_>>();
        if seen.insert(*output_id) {
            buffers.push(
                inventory
                    .get(output_id)
                    .ok_or_else(|| OpenClError::InvalidBinding("output ABI missing".into()))?
                    .clone(),
            );
        }
        let output_position = buffers
            .iter()
            .position(|buffer| buffer.id == *output_id)
            .ok_or_else(|| OpenClError::InvalidBinding("output absent from ABI".into()))?;
        if output_position + 1 != buffers.len() {
            return Err(OpenClError::InvalidBinding(
                "output aliases an input buffer".into(),
            ));
        }
        let transaction = if reduction
            .as_ref()
            .is_some_and(|reduction| reduction.plan.reduction_len() == 0)
        {
            None
        } else if let Some(reduction) = &reduction {
            OpenClTransactionAbi::analyze(
                reduction.producer,
                output_position,
                OpenClGuardDomain::ReductionSource {
                    shape: reduction.plan.geometry.input.clone(),
                },
            )?
        } else {
            OpenClTransactionAbi::analyze(
                store_value,
                output_position,
                OpenClGuardDomain::Elementwise {
                    shape: buffers[output_position].source_shape.clone(),
                },
            )?
        };
        if transaction.is_some()
            && nodes
                .iter()
                .any(crate::projected_index::ProjectedIndexPlan::is_projected)
        {
            return Err(OpenClError::Unsupported(
                "guarded projected indexing is outside the exact OpenCL subset".into(),
            ));
        }

        let entry = format!("rg_opencl_e{}_b{}", extent, buffers.len());
        let mut required_capabilities = required_capabilities(
            &buffers,
            (uses_f16 || uses_bf16) && !preserves_raw_predicated_narrow,
        );
        required_capabilities.int64 |= nodes
            .iter()
            .any(crate::projected_index::ProjectedIndexPlan::is_projected);
        if let Some(reduction) = &reduction {
            validate_dtype(
                &reduction.plan,
                reduction.plan.output_dtype,
                self.capabilities,
            )?;
            supported_storage(output_dtype, self.capabilities)?;
            let needed = reduction_capabilities(&reduction.plan, output_dtype);
            required_capabilities.int64 |= needed.int64;
            required_capabilities.fp64 |= needed.fp64;
        }
        let mut lines = Vec::new();
        if required_capabilities.fp64 {
            lines.push("#pragma OPENCL EXTENSION cl_khr_fp64 : enable".into());
        }
        if uses_f16 {
            lines.push(narrow::F16_SOURCE.into());
        }
        if uses_bf16 {
            lines.push(narrow::BF16_SOURCE.into());
        }
        lines.extend([
            format!("// {OPENCL_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
            format!("__kernel void {entry}("),
        ]);
        for (index, buffer) in buffers.iter().enumerate() {
            let qualifier = if buffer.mutable { "" } else { "const " };
            lines.push(format!(
                "    __global {qualifier}{}* b{index},",
                cl_type(buffer.dtype)
            ));
        }
        lines.push("    ulong extent) {".into());
        if transaction.is_some() {
            let last = lines.pop().expect("signature terminator");
            debug_assert_eq!(last, "    ulong extent) {");
            lines.push("    ulong extent,".into());
            lines.push("    volatile __global uint* rg_status) {".into());
        }
        lines.push("  const ulong gid = (ulong)get_global_id(0);".into());
        lines.push("  if (gid >= extent) return;".into());
        let ids = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| (buffer.id, index))
            .collect::<BTreeMap<_, _>>();
        let mut source_map = BTreeMap::new();
        if let Some(reduction) = &reduction {
            emit_reduction(
                reduction,
                output_dtype,
                output_position,
                &ids,
                &mut source_map,
                &mut lines,
                self.capabilities,
                transaction.as_ref(),
            )?;
        } else {
            let raw_predicated_narrow = if transaction.is_none() && narrow::is_narrow(output_dtype)
            {
                emit_raw_predicated_narrow_load(store_value, &ids, "gid")?
            } else {
                None
            };
            let preserves_raw_narrow = raw_predicated_narrow.is_some();
            let value = if let Some(value) = raw_predicated_narrow {
                value
            } else if let Some(transaction) = &transaction {
                emit_transactional(
                    store_value,
                    transaction,
                    &ids,
                    &mut source_map,
                    &mut lines,
                    self.capabilities,
                )?
            } else {
                emit_expr(
                    store_value,
                    &ids,
                    &mut source_map,
                    &mut lines,
                    "gid",
                    self.capabilities,
                )?
            };
            let store = format!(
                "b{output_position}[gid] = {};",
                if preserves_raw_narrow {
                    value
                } else {
                    encode_store(output_dtype, value)
                }
            );
            lines.push(if transaction.is_some() {
                format!("  if (rg_ok) {store}")
            } else {
                format!("  {store}")
            });
        }
        lines.push("}".into());
        let source = lines.join("\n") + "\n";
        let cache_key = stable_key(&(
            OPENCL_RENDERER_VERSION,
            OPENCL_ABI_VERSION,
            self.local_size,
            self.capabilities,
            &source,
            &buffers,
            &schedule_inputs,
            &reduction
                .as_ref()
                .map(|kernel| (&kernel.plan, kernel.has_epilogue())),
            &transaction,
        ));
        Ok(RenderedOpenCl {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            required_capabilities,
            transaction,
            schedule_inputs,
            semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
                root.clone(),
            ))),
        })
    }
}

fn render_portable_threefry(
    renderer: &OpenClRenderer,
    root: &UOp,
    value: &crate::ThreefryValue,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::portable_threefry::PortableThreefry::new(value).map_err(|error| match error {
            crate::portable_threefry::PortableThreefryError::Unsupported(reason) => {
                OpenClError::Unsupported(reason.into())
            }
            crate::portable_threefry::PortableThreefryError::Overflow => OpenClError::Overflow,
            other => OpenClError::InvalidBinding(other.to_string()),
        })?;
    let required_capabilities = OpenClCapabilities {
        int64: true,
        fp64: false,
    };
    if !renderer.capabilities.supports(required_capabilities) {
        return Err(OpenClError::Unsupported(
            "live Threefry requires OpenCL 64-bit integer storage".into(),
        ));
    }
    let mut buffers = portable
        .inputs()
        .iter()
        .map(|input| OpenClBufferAbi {
            id: input.node.index() as u64,
            dtype: DType::U64,
            source_shape: input.shape.clone(),
            elements: input.elements,
            mutable: false,
            view: None,
        })
        .collect::<Vec<_>>();
    let schedule_inputs = buffers.clone();
    buffers.push(OpenClBufferAbi {
        id: value.output.index() as u64,
        dtype: DType::U64,
        source_shape: value.output_shape.clone(),
        elements: portable.elements(),
        mutable: true,
        view: None,
    });
    let entry = format!("rg_opencl_threefry_e{}", portable.elements());
    let mut lines = vec![
        format!("// {OPENCL_PORTABLE_THREEFRY_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
        format!("__kernel void {entry}("),
    ];
    for (index, buffer) in buffers.iter().enumerate() {
        let qualifier = if buffer.mutable { "" } else { "const " };
        lines.push(format!("    __global {qualifier}ulong* b{index},"));
    }
    lines.extend([
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
        "  if (gid >= extent) return;".into(),
    ]);
    lines.extend(crate::portable_threefry::emit_portable_threefry_body(
        &portable,
        &crate::portable_threefry::CLikePortableThreefryDialect,
    ));
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        OPENCL_PORTABLE_THREEFRY_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.elements(),
        entry,
        cache_key,
        required_capabilities,
        transaction: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_sort(
    renderer: &OpenClRenderer,
    root: &UOp,
    value: &crate::SortValue,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::portable_sort::PortableSortPair::new(value).map_err(|error| match error {
            crate::portable_sort::PortableSortError::Unsupported(reason) => {
                OpenClError::Unsupported(reason.into())
            }
            crate::portable_sort::PortableSortError::Overflow => OpenClError::Overflow,
            other => OpenClError::InvalidBinding(other.to_string()),
        })?;
    let elements = portable.elements();
    let input = OpenClBufferAbi {
        id: value.input.index() as u64,
        dtype: value.dtype,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: false,
        view: None,
    };
    let values = OpenClBufferAbi {
        id: value.values.index() as u64,
        dtype: value.dtype,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: true,
        view: None,
    };
    let indices = OpenClBufferAbi {
        id: value.indices.index() as u64,
        dtype: DType::I32,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: true,
        view: None,
    };
    let buffers = vec![input.clone(), values, indices];
    let scalar_type = cl_type(value.dtype);
    let padding = match (value.dtype, value.descending) {
        (DType::Bool, true) => "(uchar)0".into(),
        (DType::Bool, false) => "(uchar)1".into(),
        (DType::I32, true) => "as_int(0x80000000u)".into(),
        (DType::I32, false) => "as_int(0x7fffffffu)".into(),
        (DType::U32, true) => "0u".into(),
        (DType::U32, false) => "0xffffffffu".into(),
        (DType::F32, true) => "as_float(0xff800000u)".into(),
        (DType::F32, false) => "as_float(0x7f800000u)".into(),
        _ => unreachable!("portable sort validated storage"),
    };
    let entry = format!(
        "rg_opencl_sort_{:?}_a{}_n{}",
        value.dtype,
        value.axis,
        portable.elements()
    )
    .to_ascii_lowercase();
    let mut lines = vec![
        "#pragma OPENCL FP_CONTRACT OFF".into(),
        format!("// {OPENCL_PORTABLE_SORT_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
        format!("__kernel void {entry}("),
        format!("    __global const {scalar_type}* b0,"),
        format!("    __global {scalar_type}* b1,"),
        "    __global int* b2,".into(),
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
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
        .map_err(|error| OpenClError::Unsupported(error.to_string()))?,
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        OPENCL_PORTABLE_SORT_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.launch_extent(),
        entry,
        cache_key,
        required_capabilities: OpenClCapabilities::CORE_32,
        transaction: None,
        schedule_inputs: vec![input],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_prefix_scan(
    renderer: &OpenClRenderer,
    root: &UOp,
    value: &crate::PrefixScanValue,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::prefix_scan_native::PortablePrefixScan::new(value).map_err(|error| match error {
            crate::prefix_scan_native::PortablePrefixScanError::Unsupported(reason) => {
                OpenClError::Unsupported(reason.into())
            }
            crate::prefix_scan_native::PortablePrefixScanError::Overflow => OpenClError::Overflow,
            other => OpenClError::InvalidBinding(other.to_string()),
        })?;
    let plan = portable.plan();
    let input = OpenClBufferAbi {
        id: plan.input,
        dtype: plan.input_dtype,
        source_shape: value.input_shape.clone(),
        elements: plan.elements,
        mutable: false,
        view: None,
    };
    let output = OpenClBufferAbi {
        id: plan.output,
        dtype: plan.output_dtype,
        source_shape: value.output_shape.clone(),
        elements: plan.elements,
        mutable: true,
        view: None,
    };
    let buffers = vec![input.clone(), output];
    let entry = format!(
        "rg_opencl_scan_{:?}_{:?}_a{}_n{}",
        plan.kind, plan.result, plan.axis, plan.elements
    )
    .to_ascii_lowercase();
    let input_type = cl_type(plan.input_dtype);
    let output_type = cl_type(plan.output_dtype);
    let mut lines = vec![
        "#pragma OPENCL FP_CONTRACT OFF".into(),
        format!("// {OPENCL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
        format!("__kernel void {entry}("),
        format!("    __global const {input_type}* b0,"),
        format!("    __global {output_type}* b1,"),
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
        format!(
            "  if (gid >= extent || gid >= (ulong){}ul) return;",
            plan.work_items()
        ),
    ];
    lines.extend(
        crate::prefix_scan_native::emit_portable_prefix_scan_body(
            &portable,
            &OpenClPrefixScanDialect,
        )
        .map_err(|error| OpenClError::Unsupported(error.to_string()))?,
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        OPENCL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.launch_extent(),
        entry,
        cache_key,
        required_capabilities: OpenClCapabilities::CORE_32,
        transaction: None,
        schedule_inputs: vec![input],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

struct OpenClPrefixScanDialect;

impl crate::prefix_scan_native::PortablePrefixScanDialect for OpenClPrefixScanDialect {
    fn scalar_body(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<Vec<String>, crate::prefix_scan_native::PortablePrefixScanError> {
        Ok(vec![match plan.result {
            crate::PrefixScanOutput::Indices => "  b1[0] = (int)0;".into(),
            crate::PrefixScanOutput::Values if plan.input_dtype == plan.output_dtype => {
                if plan.input_dtype == DType::F32 {
                    "  b1[0] = as_float(as_uint(b0[0]));".into()
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
            (crate::PrefixScanKind::Max, DType::F32) => "as_float(0xff800000u)",
            (crate::PrefixScanKind::Min, DType::F32) => "as_float(0x7f800000u)",
            (crate::PrefixScanKind::Product | crate::PrefixScanKind::Min, DType::Bool) => {
                "(uchar)1"
            }
            (crate::PrefixScanKind::Product, DType::I32) => "(int)1",
            (crate::PrefixScanKind::Product, DType::U32) => "1u",
            (crate::PrefixScanKind::Max, DType::I32) => "as_int(0x80000000u)",
            (crate::PrefixScanKind::Min, DType::I32) => "as_int(0x7fffffffu)",
            (crate::PrefixScanKind::Min, DType::U32) => "0xffffffffu",
            (_, DType::Bool | DType::I32 | DType::U32) => "0",
            _ => {
                return Err(
                    crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                        "OpenCL portable scan identity dtype",
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
        format!("  {} rg_acc = {identity};", cl_type(plan.work_dtype))
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
        let work = cl_type(plan.work_dtype);
        Ok(format!("    const {work} rg_next = ({work})b0[rg_offset];"))
    }

    fn strict(
        &self,
        _plan: &crate::prefix_scan_native::NativePrefixScanPlan,
        operator: &str,
    ) -> String {
        format!("    const int rg_strict = rg_next {operator} rg_acc;")
    }

    fn equal_before(&self) -> String {
        "    const int rg_equal_before = rg_next == rg_acc;".into()
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
                format!("as_int(as_uint(rg_acc) {operator} as_uint(rg_next))")
            }
            DType::U32 | DType::F32 => format!("rg_acc {operator} rg_next"),
            _ => {
                return Err(
                    crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                        "OpenCL portable scan arithmetic dtype",
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
            format!("({})rg_acc", cl_type(plan.output_dtype))
        };
        vec![format!("    b1[rg_offset] = {stored};")]
    }

    fn loop_close(&self) -> String {
        "  }".into()
    }
}

fn render_portable_f32_matmul(
    renderer: &OpenClRenderer,
    root: &UOp,
    value: &crate::MatmulValue,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let portable = crate::matmul::PortableF32Matmul::new(value).map_err(|error| match error {
        crate::matmul::PortableF32MatmulError::Unsupported(reason) => {
            OpenClError::Unsupported(reason.into())
        }
        crate::matmul::PortableF32MatmulError::Overflow => OpenClError::Overflow,
        other => OpenClError::InvalidBinding(other.to_string()),
    })?;
    let plan = portable.plan();
    let mut buffers = Vec::with_capacity(3);
    let mut schedule_inputs = Vec::with_capacity(2);
    for input in portable.inputs() {
        let elements = input.shape.numel().map_err(|_| OpenClError::Overflow)?;
        let abi = OpenClBufferAbi {
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
    let output = OpenClBufferAbi {
        id: plan.output.index() as u64,
        dtype: DType::F32,
        source_shape: plan.output_shape.clone(),
        elements: portable.extent(),
        mutable: true,
        view: None,
    };
    buffers.push(output);
    let positions = buffers
        .iter()
        .enumerate()
        .map(|(position, abi)| (abi.id, position))
        .collect::<BTreeMap<_, _>>();
    let lhs_position = positions[&(plan.lhs.index() as u64)];
    let rhs_position = positions[&(plan.rhs.index() as u64)];
    let output_position = buffers.len() - 1;
    let entry = format!("rg_opencl_matmul_f32_{}", plan.cache_key);
    let mut lines = vec![
        format!("// {OPENCL_PORTABLE_F32_MATMUL_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
        "#pragma OPENCL FP_CONTRACT OFF".into(),
        format!("__kernel void {entry}("),
    ];
    for (position, _) in buffers[..output_position].iter().enumerate() {
        lines.push(format!("    __global const float* b{position},"));
    }
    lines.extend([
        format!("    __global float* b{output_position},"),
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
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
    lines.extend([
        "  float rg_acc = 0.0f;".into(),
        format!(
            "  for (ulong rg_k = 0ul; rg_k < (ulong){}ul; ++rg_k) {{",
            plan.k
        ),
    ]);
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
        OPENCL_PORTABLE_F32_MATMUL_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.extent(),
        entry,
        cache_key,
        required_capabilities: OpenClCapabilities::CORE_32,
        transaction: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_bitcast(
    renderer: &OpenClRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let portable = crate::movement_plan::PortableBitcast::new(plan)
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let input = portable.input();
    let input_abi = OpenClBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: portable.input_elements(),
        mutable: false,
        view: None,
    };
    let buffers = vec![
        input_abi.clone(),
        OpenClBufferAbi {
            id: plan.output.index() as u64,
            dtype: plan.dtype,
            source_shape: plan.output_shape.clone(),
            elements: portable.output_elements(),
            mutable: true,
            view: None,
        },
    ];
    let entry = "rg_opencl_portable_bitcast".to_owned();
    let stored = if portable.normalizes_bool() {
        "b0[gid] != (uchar)0"
    } else {
        "b0[gid]"
    };
    let source = [
        format!("// {OPENCL_PORTABLE_BITCAST_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
        format!("__kernel void {entry}("),
        "    __global const uchar* b0,".into(),
        "    __global uchar* b1,".into(),
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
        "  if (gid >= extent) return;".into(),
        format!("  b1[gid] = (uchar)({stored});"),
        "}".into(),
    ]
    .join("\n")
        + "\n";
    let cache_key = stable_key(&(
        OPENCL_PORTABLE_BITCAST_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        portable.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.bytes(),
        entry,
        cache_key,
        required_capabilities: OpenClCapabilities::CORE_32,
        transaction: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_portable_dense_materialization(
    renderer: &OpenClRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let portable = crate::movement_plan::PortableDenseMaterialization::new(plan)
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let mut buffers = portable
        .inputs()
        .iter()
        .map(|input| {
            Ok(OpenClBufferAbi {
                id: input.node.index() as u64,
                dtype: input.dtype,
                source_shape: input.shape.clone(),
                elements: input.shape.numel().map_err(|_| OpenClError::Overflow)?,
                mutable: false,
                view: None,
            })
        })
        .collect::<Result<Vec<_>, OpenClError>>()?;
    let schedule_inputs = buffers.clone();
    buffers.push(OpenClBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: portable.elements(),
        mutable: true,
        view: None,
    });
    let entry = "rg_opencl_portable_dense_materialization".to_owned();
    let mut lines = vec![format!(
        "// {OPENCL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"
    )];
    lines.push(format!("__kernel void {entry}("));
    for (index, _) in portable.inputs().iter().enumerate() {
        lines.push(format!("    __global const uchar* b{index},"));
    }
    lines.push(format!("    __global uchar* b{},", portable.inputs().len()));
    lines.extend([
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
        "  if (gid >= extent) return;".into(),
    ]);
    lines.extend(
        crate::portable_movement::emit_portable_dense_materialization_body(
            &portable,
            &crate::portable_movement::CLikePortableDenseDialect {
                input_address: "__global",
                output_address: "__global",
            },
        ),
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        OPENCL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        portable.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.elements(),
        entry,
        cache_key,
        required_capabilities: OpenClCapabilities::CORE_32,
        transaction: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_raw_copy(
    renderer: &OpenClRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let copy = plan
        .raw_copy()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?
        .ok_or_else(|| {
            OpenClError::Unsupported(
                "only raw AffineCopy and Contiguous have OpenCL movement lowering".into(),
            )
        })?;
    let input = copy.input();
    let extent = copy.elements();
    let width = copy.width();
    let raw_type = match width {
        1 => "uchar",
        2 => "ushort",
        4 => "uint",
        8 => "ulong",
        _ => {
            return Err(OpenClError::Unsupported(format!(
                "raw-copy OpenCL storage width {width}"
            )));
        }
    };
    let required_capabilities = OpenClCapabilities {
        int64: width == 8,
        fp64: false,
    };
    if !renderer.capabilities.supports(required_capabilities) {
        return Err(OpenClError::Unsupported(
            "raw-copy OpenCL 64-bit storage requires 64-bit integer support".into(),
        ));
    }
    debug_assert_eq!(copy.bytes(), extent * width);
    let input_abi = OpenClBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: copy.input_elements(),
        mutable: false,
        view: None,
    };
    let output_abi = OpenClBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    let buffers = vec![input_abi.clone(), output_abi];
    let entry = format!("rg_opencl_raw_copy_w{width}");
    let mut lines = vec![
        format!("// {OPENCL_RAW_COPY_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
        format!("__kernel void {entry}("),
        format!("    __global const {raw_type}* b0,"),
        format!("    __global {raw_type}* b1,"),
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
        "  if (gid >= extent) return;".into(),
    ];
    let source_index = if let Some(address) = copy
        .address()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?
    {
        lines.push(format!("  ulong rg_source = (ulong){}ul;", address.offset));
        for axis in address.axes {
            let output_axis = axis.output_axis;
            lines.push(format!(
                "  ulong rg_axis_{output_axis} = (gid / (ulong){}ul) % (ulong){}ul;",
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
        OPENCL_RAW_COPY_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        copy.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent,
        entry,
        cache_key,
        required_capabilities,
        transaction: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn render_static_positions(
    renderer: &OpenClRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedOpenCl, OpenClError> {
    root.validate()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
    let placement = plan
        .static_position_write()
        .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?
        .ok_or_else(|| OpenClError::InvalidBinding("missing static-position projection".into()))?;
    let input = placement.input();
    let extent = placement.elements();
    let width = placement.width();
    let raw_type = match width {
        1 => "uchar",
        2 => "ushort",
        4 => "uint",
        8 => "ulong",
        _ => {
            return Err(OpenClError::Unsupported(format!(
                "static-position OpenCL storage width {width}"
            )));
        }
    };
    let required_capabilities = OpenClCapabilities {
        int64: width == 8,
        fp64: false,
    };
    if !renderer.capabilities.supports(required_capabilities) {
        return Err(OpenClError::Unsupported(
            "static-position OpenCL 64-bit storage requires 64-bit integer support".into(),
        ));
    }
    debug_assert_eq!(placement.bytes(), extent * width);
    let input_abi = OpenClBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: placement.input_elements(),
        mutable: false,
        view: None,
    };
    let output_abi = OpenClBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    let buffers = vec![input_abi.clone(), output_abi];
    let entry = format!("rg_opencl_static_position_w{width}");
    let mut lines = vec![
        format!("// {OPENCL_STATIC_POSITION_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
        format!("__kernel void {entry}("),
        format!("    __global const {raw_type}* b0,"),
        format!("    __global {raw_type}* b1,"),
        "    ulong extent) {".into(),
        "  const ulong gid = (ulong)get_global_id(0);".into(),
        "  if (gid >= extent) return;".into(),
        "  int rg_mapped = 0;".into(),
        "  ulong rg_source = 0ul;".into(),
    ];
    if placement.has_source() {
        lines.push("  rg_mapped = 1;".into());
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
                "  if (rg_coordinate_{name} < (ulong){}ul || rg_delta_{name} % (ulong){}ul != 0ul || rg_quotient_{name} >= (ulong){}ul) rg_mapped = 0;",
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
        OPENCL_STATIC_POSITION_RENDERER_VERSION,
        OPENCL_ABI_VERSION,
        renderer.local_size,
        renderer.capabilities,
        placement.plan(),
        &source,
        &buffers,
    ));
    Ok(RenderedOpenCl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent,
        entry,
        cache_key,
        required_capabilities,
        transaction: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    })
}

fn supported_storage(dtype: DType, capabilities: OpenClCapabilities) -> Result<(), OpenClError> {
    match dtype {
        DType::Bool | DType::I32 | DType::U32 | DType::F32 => Ok(()),
        DType::F16 | DType::BF16 if capabilities.fp64 => Ok(()),
        DType::I64 | DType::U64 if capabilities.int64 => Ok(()),
        DType::F64 if capabilities.fp64 => Ok(()),
        DType::F16 | DType::BF16 => Err(OpenClError::Unsupported(
            "exact narrow-float execution requires fp64 capability".into(),
        )),
        DType::I64 | DType::U64 => Err(OpenClError::Unsupported(
            "64-bit integer storage requires device capability".into(),
        )),
        DType::F64 => Err(OpenClError::Unsupported(
            "F64 storage requires fp64 device capability".into(),
        )),
        _ => Err(OpenClError::Unsupported(format!(
            "dtype {dtype:?} is outside the exact OpenCL static subset"
        ))),
    }
}

fn supported_kernel_storage(
    dtype: DType,
    capabilities: OpenClCapabilities,
    preserves_raw_predicated_narrow: bool,
) -> Result<(), OpenClError> {
    if preserves_raw_predicated_narrow && narrow::is_narrow(dtype) {
        Ok(())
    } else {
        supported_storage(dtype, capabilities)
    }
}

pub(super) fn cl_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "uchar",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::I64 => "long",
        DType::U64 => "ulong",
        DType::F32 => "float",
        DType::F64 => "double",
        DType::F16 | DType::BF16 => "ushort",
        _ => unreachable!("validated by supported_storage"),
    }
}

fn expression_type(dtype: DType) -> &'static str {
    if narrow::is_narrow(dtype) {
        "double"
    } else {
        cl_type(dtype)
    }
}

pub(super) struct OpenClScalarDialect {
    pub(super) capabilities: OpenClCapabilities,
}

impl dialect_seal::Sealed for OpenClScalarDialect {}

impl ScalarLaneDialect for OpenClScalarDialect {
    fn name(&self) -> &'static str {
        "OpenCL"
    }

    fn supports_value(&self, dtype: DType) -> bool {
        supported_storage(dtype, self.capabilities).is_ok()
    }

    fn cast(&self, source: DType, target: DType, value: &str) -> Result<String, String> {
        let result = match (source, target) {
            (source, target) if source == target => value.into(),
            (DType::Bool, target) if self.supports_value(target) => {
                format!("(({})({value}))", cl_type(target))
            }
            (source, DType::Bool) if self.supports_value(source) => {
                format!("((uchar)(({value}) != ({})0))", expression_type(source))
            }
            (DType::I32, DType::U32) => format!("as_uint({value})"),
            (DType::U32, DType::I32) => format!("as_int({value})"),
            (DType::I64, DType::U64) => format!("as_ulong({value})"),
            (DType::U64, DType::I64) => format!("as_long({value})"),
            (DType::I32 | DType::U32 | DType::I64 | DType::U64, DType::F32) => {
                format!("((float)({value}))")
            }
            (DType::F32, DType::F64) => format!("((double)({value}))"),
            (DType::F64, DType::F32) => format!("((float)({value}))"),
            (source, target)
                if narrow::is_narrow(target)
                    && matches!(source, DType::F16 | DType::BF16 | DType::F32 | DType::F64) =>
            {
                narrow::quantize(target, value).expect("target is a narrow float")
            }
            (source, DType::F32) if narrow::is_narrow(source) => {
                format!("((float)({value}))")
            }
            (source, DType::F64) if narrow::is_narrow(source) => value.into(),
            _ => return Err("cast is outside the exact OpenCL subset".into()),
        };
        Ok(result)
    }

    fn finish_float(&self, dtype: DType, value: String) -> Result<String, String> {
        Ok(narrow::quantize(dtype, &value).unwrap_or(value))
    }

    fn signed_infix(
        &self,
        dtype: DType,
        operator: &'static str,
        lhs: &str,
        rhs: &str,
    ) -> Result<String, String> {
        match dtype {
            DType::I32 => Ok(format!("as_int(as_uint({lhs}) {operator} as_uint({rhs}))")),
            DType::I64 => Ok(format!(
                "as_long(as_ulong({lhs}) {operator} as_ulong({rhs}))"
            )),
            _ => Err("OpenCL signed wrapping requires I32 or I64".into()),
        }
    }

    fn signed_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        match dtype {
            DType::I32 => Ok(format!("as_int((uint)0u - as_uint({value}))")),
            DType::I64 => Ok(format!("as_long((ulong)0ul - as_ulong({value}))")),
            _ => Err("OpenCL signed negation requires I32 or I64".into()),
        }
    }

    fn unsigned_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        match dtype {
            DType::U32 => Ok(format!("((uint)0u - ({value}))")),
            DType::U64 => Ok(format!("((ulong)0ul - ({value}))")),
            _ => Err("OpenCL unsigned negation requires U32 or U64".into()),
        }
    }

    fn signed_abs(&self, dtype: DType, value: &str) -> Result<String, String> {
        match dtype {
            DType::I32 => Ok(format!(
                "as_int(({value}) < 0 ? ((uint)0u - as_uint({value})) : as_uint({value}))"
            )),
            DType::I64 => Ok(format!(
                "as_long(({value}) < 0 ? ((ulong)0ul - as_ulong({value})) : as_ulong({value}))"
            )),
            _ => Err("OpenCL signed absolute value requires I32 or I64".into()),
        }
    }

    fn float_abs(&self, value: &str) -> String {
        format!("fabs({value})")
    }

    fn bool_value(&self, expression: String) -> String {
        format!("((uchar)({expression}))")
    }

    fn select(&self, condition: &str, on_true: &str, on_false: &str) -> String {
        format!("(({condition}) ? ({on_true}) : ({on_false}))")
    }

    fn call_intrinsic(&self, canonical_name: &'static str, value: &str) -> String {
        format!("{canonical_name}({value})")
    }

    fn float_one(&self, dtype: DType) -> Result<&'static str, String> {
        match dtype {
            DType::F32 => Ok("1.0f"),
            DType::F16 | DType::BF16 | DType::F64 => Ok("1.0"),
            _ => Err("OpenCL reciprocal requires floating dtype".into()),
        }
    }
}

fn encode_store(dtype: DType, value: impl AsRef<str>) -> String {
    narrow::encode(dtype, value.as_ref()).unwrap_or_else(|| value.as_ref().into())
}

fn emit_raw_predicated_narrow_load(
    value: &UOp,
    ids: &BTreeMap<u64, usize>,
    linear: &str,
) -> Result<Option<String>, OpenClError> {
    let Some(plan) = crate::projected_index::ProjectedIndexPlan::from_direct_predicated_load(value)
        .map_err(|_| OpenClError::Unsupported("invalid predicated narrow load".into()))?
    else {
        return Ok(None);
    };
    let position = ids
        .get(&plan.buffer)
        .ok_or_else(|| OpenClError::InvalidBinding("load buffer absent from ABI".into()))?;
    let access = crate::projected_index::render_infix_projected_access(
        &plan,
        format!("((long)({linear}))"),
        |literal| {
            Ok(if literal == i64::MIN {
                "((-9223372036854775807l) - 1l)".into()
            } else {
                format!("((long){literal}l)")
            })
        },
        |boolean| if boolean { "1" } else { "0" }.into(),
    )
    .map_err(|_| OpenClError::Unsupported("invalid predicated narrow load".into()))?;
    let predicate = access
        .predicate
        .ok_or_else(|| OpenClError::Unsupported("predicated narrow load has no guard".into()))?;
    Ok(Some(format!(
        "(({predicate}) ? (b{position}[{}]) : ((ushort)0u))",
        access.offset
    )))
}

fn required_capabilities(buffers: &[OpenClBufferAbi], uses_narrow: bool) -> OpenClCapabilities {
    OpenClCapabilities {
        int64: buffers
            .iter()
            .any(|buffer| matches!(buffer.dtype, DType::I64 | DType::U64)),
        fp64: uses_narrow || buffers.iter().any(|buffer| buffer.dtype == DType::F64),
    }
}

pub(super) fn guarded_value(
    op: GuardedIntegerOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, OpenClError> {
    let signed = matches!(dtype, DType::I32 | DType::I64);
    let min = match dtype {
        DType::I32 => "as_int((uint)0x80000000u)",
        DType::I64 => "as_long((ulong)0x8000000000000000ul)",
        _ => "0",
    };
    let overflow = if signed {
        format!("(({lhs}) == {min} && ({rhs}) == ({})-1)", cl_type(dtype))
    } else {
        "0".into()
    };
    let div = format!("(({overflow}) ? ({min}) : (({lhs}) / ({rhs})))");
    let rem = format!(
        "(({overflow}) ? ({})0 : (({lhs}) % ({rhs})))",
        cl_type(dtype)
    );
    Ok(match op {
        GuardedIntegerOp::Div | GuardedIntegerOp::TruncDiv => div,
        GuardedIntegerOp::FMod => rem,
        GuardedIntegerOp::FloorDiv if !signed => format!("(({lhs}) / ({rhs}))"),
        GuardedIntegerOp::Mod if !signed => format!("(({lhs}) % ({rhs}))"),
        GuardedIntegerOp::FloorDiv => {
            // RustGrad's signed integer oracle uses Euclidean division: the remainder is
            // nonnegative even when the divisor is negative. OpenCL integer division
            // truncates toward zero, so correct only a negative remainder; its direction
            // depends on the divisor sign.
            let rem = format!("(({lhs}) % ({rhs}))");
            let correction = format!("((({rhs}) > 0) ? 1 : -1)");
            format!(
                "(({overflow}) ? ({min}) : (({rem} < 0) ? (({lhs}) / ({rhs}) - {correction}) : (({lhs}) / ({rhs}))))"
            )
        }
        GuardedIntegerOp::Mod => {
            let signed_name = if dtype == DType::I32 { "int" } else { "long" };
            let unsigned_name = if dtype == DType::I32 { "uint" } else { "ulong" };
            let rem = format!("(({lhs}) % ({rhs}))");
            let magnitude = format!(
                "((({rhs}) < 0) ? (({unsigned_name})0 - as_{unsigned_name}({rhs})) : as_{unsigned_name}({rhs}))"
            );
            format!(
                "(({overflow}) ? ({})0 : (({rem} < 0) ? as_{signed_name}(as_{unsigned_name}({rem}) + {magnitude}) : {rem}))",
                cl_type(dtype),
            )
        }
        GuardedIntegerOp::Shl => match dtype {
            DType::I32 => format!("as_int(as_uint({lhs}) << (uint)({rhs}))"),
            DType::I64 => format!("as_long(as_ulong({lhs}) << (ulong)({rhs}))"),
            _ => format!("(({lhs}) << ({rhs}))"),
        },
        GuardedIntegerOp::Shr => format!("(({lhs}) >> ({rhs}))"),
    })
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
    capabilities: OpenClCapabilities,
) -> Result<String, OpenClError> {
    emit_expr_with_substitution(node, ids, source_map, lines, linear, capabilities, None)
}

fn emit_expr_with_substitution(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
    capabilities: OpenClCapabilities,
    substitution: Option<(&UOp, &str)>,
) -> Result<String, OpenClError> {
    if let Some((target, value)) = substitution
        && node.shares_node_with(target)
    {
        return Ok(value.into());
    }
    let map_id = source_map.len();
    source_map.insert(map_id, lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| OpenClError::Unsupported(format!("untyped {:?}", node.operation())))?
        .scalar;
    supported_storage(dtype, capabilities)?;
    let child = |index: usize, source_map: &mut BTreeMap<usize, usize>, lines: &mut Vec<String>| {
        node.sources()
            .get(index)
            .ok_or_else(|| OpenClError::Unsupported("missing expression operand".into()))
            .and_then(|source| {
                emit_expr_with_substitution(
                    source,
                    ids,
                    source_map,
                    lines,
                    linear,
                    capabilities,
                    substitution,
                )
            })
    };
    match node.operation() {
        Operation::Const(value) => match value {
            LiteralValue::Scalar {
                dtype: DType::F32,
                bits,
            } => Ok(format!("as_float((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::Bool,
                bits,
            } if *bits <= 1 => Ok(format!("((uchar){bits}u)")),
            LiteralValue::Scalar {
                dtype: DType::I32,
                bits,
            } => Ok(format!("as_int((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::U32,
                bits,
            } => Ok(format!("((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::I64,
                bits,
            } => Ok(format!("as_long((ulong)0x{bits:016x}ul)")),
            LiteralValue::Scalar {
                dtype: DType::U64,
                bits,
            } => Ok(format!("((ulong)0x{bits:016x}ul)")),
            LiteralValue::Scalar {
                dtype: DType::F64,
                bits,
            } => Ok(format!("as_double((ulong)0x{bits:016x}ul)")),
            LiteralValue::Scalar {
                dtype: DType::F16,
                bits,
            } if dtype == DType::F16 => Ok(narrow::decode(
                DType::F16,
                format!("((ushort)0x{:04x}u)", *bits as u16),
            )
            .expect("F16 is a narrow float")),
            LiteralValue::Scalar {
                dtype: DType::BF16,
                bits,
            } if dtype == DType::BF16 => Ok(narrow::decode(
                DType::BF16,
                format!("((ushort)0x{:04x}u)", *bits as u16),
            )
            .expect("BF16 is a narrow float")),
            LiteralValue::Scalar { .. } => Err(OpenClError::Unsupported(
                "scalar literal/type mismatch".into(),
            )),
            _ => Err(OpenClError::Unsupported("invalid scalar literal".into())),
        },
        Operation::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::Unsupported("load has no index".into()))?;
            let (buffer, input_shape, output_shape, view) = match index.operation() {
                Operation::Index(IndexValue::Buffer { buffer, .. })
                    if crate::projected_index::ProjectedIndexPlan::is_projected(index) =>
                {
                    let plan = crate::projected_index::ProjectedIndexPlan::from_index(index)
                        .map_err(|_| OpenClError::Unsupported("invalid projected index".into()))?;
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
                    .map_err(|_| OpenClError::Unsupported("invalid projected index".into()))?;
                    let position = ids.get(buffer).ok_or_else(|| {
                        OpenClError::InvalidBinding("load buffer absent from ABI".into())
                    })?;
                    let raw = format!("b{position}[{}]", access.offset);
                    let value = if dtype == DType::Bool {
                        format!("(({raw}) != 0)")
                    } else if narrow::is_narrow(dtype) {
                        narrow::decode(dtype, raw).expect("validated narrow load")
                    } else {
                        raw
                    };
                    return Ok(access
                        .predicate
                        .map(|predicate| {
                            let zero = match dtype {
                                DType::Bool => "false",
                                DType::F16 | DType::BF16 | DType::F32 => "0.0f",
                                DType::F64 => "0.0",
                                DType::I32 => "((int)0)",
                                DType::U32 => "((uint)0u)",
                                DType::I64 => "((long)0l)",
                                DType::U64 => "((ulong)0ul)",
                                _ => unreachable!("validated OpenCL storage"),
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
                    return Err(OpenClError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            let position = ids
                .get(&buffer)
                .ok_or_else(|| OpenClError::InvalidBinding("load buffer absent from ABI".into()))?;
            let logical = broadcast_offset(input_shape, output_shape, linear)?;
            let offset = match view {
                Some(view) => OpenClViewAccess::new(view, dtype)?.expression(logical),
                None => logical,
            };
            let raw = format!("b{position}[{offset}]");
            Ok(narrow::decode(dtype, &raw).unwrap_or(raw))
        }
        other => {
            let mut sources = Vec::with_capacity(node.sources().len());
            for slot in 0..node.sources().len() {
                sources.push(child(slot, source_map, lines)?);
            }
            let instruction = project_scalar_lane(node, &sources)
                .map_err(OpenClError::Unsupported)?
                .ok_or_else(|| OpenClError::Unsupported(format!("{other:?}")))?;
            emit_scalar_lane(&OpenClScalarDialect { capabilities }, &instruction)
                .map_err(OpenClError::Unsupported)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_reduction(
    reduction: &crate::reduction_native::NativeReductionKernel<'_>,
    dtype: DType,
    output_position: usize,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    capabilities: OpenClCapabilities,
    transaction: Option<&OpenClTransactionAbi>,
) -> Result<(), OpenClError> {
    let value = reduction.producer;
    let plan = &reduction.plan;
    let reduction_len = plan.reduction_len();
    if reduction_len == 0 {
        let final_value =
            reduction_identity_expr(plan.output_dtype, plan.finalize(plan.identity()))?;
        let committed = reduction_finalize_commit_expr(
            plan.output_dtype,
            &final_value,
            reduction.has_epilogue(),
        );
        let final_value = if reduction.has_epilogue() {
            emit_expr_with_substitution(
                reduction.epilogue_root,
                ids,
                source_map,
                lines,
                "gid",
                capabilities,
                Some((reduction.finalize, committed.as_str())),
            )?
        } else {
            committed
        };
        lines.push(format!(
            "  b{output_position}[gid] = {};",
            encode_store(dtype, final_value)
        ));
        return Ok(());
    }
    if transaction.is_some() {
        lines.push("  uchar rg_ok = (uchar)1u;".into());
    }
    let accumulator_type = reduction_accumulator_type(plan.accumulator_dtype, plan.kind);
    let identity = reduction_identity_expr(plan.accumulator_dtype, plan.identity())?;
    lines.push(format!("  {accumulator_type} acc = {identity};"));
    lines.push(format!(
        "  for (ulong r = 0ul; r < {}ul; ++r) {{",
        reduction_len
    ));
    lines.push(format!(
        "    const ulong src_gid = {};",
        input_offset_expression(plan)?
    ));
    let expression = if let Some(transaction) = transaction {
        emit_transactional_reduction(value, transaction, ids, source_map, lines, capabilities)?
    } else {
        emit_expr(value, ids, source_map, lines, "src_gid", capabilities)?
    };
    let prefix = if transaction.is_some() {
        "    if (rg_ok) "
    } else {
        "    "
    };
    let expression = reduction_cast_expr(plan.source_dtype, plan.accumulator_dtype, &expression);
    if plan.is_singleton_identity() {
        lines.push(format!("{prefix}acc = ({accumulator_type})({expression});"));
    } else {
        match plan.kind {
            crate::ReduceKind::Any => {
                lines.push(format!("{prefix}acc = (uchar)(acc || ({expression}));"));
            }
            crate::ReduceKind::All => {
                lines.push(format!("{prefix}acc = (uchar)(acc && ({expression}));"));
            }
            crate::ReduceKind::Sum | crate::ReduceKind::Mean
                if plan.accumulator_dtype == DType::Bool =>
            {
                lines.push(format!("{prefix}acc = (uchar)(acc || ({expression}));"));
            }
            crate::ReduceKind::Sum | crate::ReduceKind::Mean | crate::ReduceKind::Product => {
                let product = plan.kind == crate::ReduceKind::Product;
                let update =
                    reduction_arithmetic_expr(plan.accumulator_dtype, "acc", &expression, product);
                lines.push(format!("{prefix}acc = {update};"));
            }
            crate::ReduceKind::Min | crate::ReduceKind::Max => {
                let comparison = if plan.kind == crate::ReduceKind::Max {
                    ">"
                } else {
                    "<"
                };
                lines.push(format!(
                    "    const {accumulator_type} next = ({accumulator_type})({expression});"
                ));
                lines.push(format!("{prefix}if (next {comparison} acc) acc = next;"));
            }
        }
    }
    lines.push("  }".into());
    if matches!(plan.kind, crate::ReduceKind::Mean) {
        let divisor = reduction_identity_expr(
            plan.accumulator_dtype,
            plan.mean_divisor()
                .expect("nonempty validated Mean divisor"),
        )?;
        let update = reduction_commit_expr(plan.accumulator_dtype, &format!("(acc / {divisor})"));
        lines.push(if transaction.is_some() {
            format!("  if (rg_ok) acc = {update};")
        } else {
            format!("  acc = {update};")
        });
    }
    let final_value = reduction_cast_expr(plan.accumulator_dtype, plan.output_dtype, "acc");
    let committed =
        reduction_finalize_commit_expr(plan.output_dtype, &final_value, reduction.has_epilogue());
    let final_value = if reduction.has_epilogue() {
        emit_expr_with_substitution(
            reduction.epilogue_root,
            ids,
            source_map,
            lines,
            "gid",
            capabilities,
            Some((reduction.finalize, committed.as_str())),
        )?
    } else {
        committed
    };
    let store = format!(
        "b{output_position}[gid] = {};",
        encode_store(dtype, final_value)
    );
    lines.push(if transaction.is_some() {
        format!("  if (rg_ok) {store}")
    } else {
        format!("  {store}")
    });
    Ok(())
}

fn reduction_accumulator_type(dtype: DType, kind: crate::ReduceKind) -> &'static str {
    match (dtype, kind) {
        (DType::I32, crate::ReduceKind::Sum | crate::ReduceKind::Product) => "uint",
        (DType::I64, crate::ReduceKind::Sum | crate::ReduceKind::Product) => "ulong",
        (DType::F16 | DType::BF16 | DType::F32, _) => "float",
        (DType::F64, _) => "double",
        _ => cl_type(dtype),
    }
}

fn reduction_identity_expr(dtype: DType, value: crate::Scalar) -> Result<String, OpenClError> {
    Ok(match value {
        crate::Scalar::Bool(value) => format!("(uchar){}u", u8::from(value)),
        crate::Scalar::I(value) if dtype == DType::I32 => {
            format!("as_int((uint)0x{:08x}u)", value as u32)
        }
        crate::Scalar::I(value) if dtype == DType::I64 => {
            format!("as_long((ulong)0x{:016x}ul)", value as u64)
        }
        crate::Scalar::I(value) => value.to_string(),
        crate::Scalar::U(value) if dtype == DType::U64 => format!("(ulong){value}ul"),
        crate::Scalar::U(value) => format!("(uint){value}u"),
        crate::Scalar::F(value) if value.is_nan() => "NAN".into(),
        crate::Scalar::F(value) if value == f64::NEG_INFINITY => "-INFINITY".into(),
        crate::Scalar::F(value) if value == f64::INFINITY => "INFINITY".into(),
        crate::Scalar::F(0.0) => "0.0".into(),
        crate::Scalar::F(1.0) => "1.0".into(),
        crate::Scalar::F(value) if value.is_finite() => format!("{value:.17e}"),
        crate::Scalar::F(_) => {
            return Err(OpenClError::Unsupported(
                "OpenCL reduction identity is not representable".into(),
            ));
        }
    })
}

fn reduction_cast_expr(source: DType, target: DType, value: &str) -> String {
    match (source, target) {
        (DType::I32, DType::I32) | (DType::I64, DType::I64) => value.into(),
        (_, DType::I32) => format!("(int)({value})"),
        (_, DType::U32) => format!("(uint)({value})"),
        (_, DType::I64) => format!("(long)({value})"),
        (_, DType::U64) => format!("(ulong)({value})"),
        (_, DType::F32) => format!("(float)({value})"),
        (_, DType::F64) => format!("(double)({value})"),
        (_, DType::F16 | DType::BF16) => format!("(float)({value})"),
        (_, DType::Bool) => format!("(uchar)(({value}) != 0)"),
        _ => value.into(),
    }
}

fn reduction_arithmetic_expr(dtype: DType, lhs: &str, rhs: &str, product: bool) -> String {
    let operator = if product { "*" } else { "+" };
    reduction_commit_expr(dtype, &format!("(({lhs}) {operator} ({rhs}))"))
}

fn reduction_commit_expr(dtype: DType, value: &str) -> String {
    match dtype {
        DType::I32 => format!("(uint)({value})"),
        DType::I64 => format!("(ulong)({value})"),
        DType::F32 => format!("(float)({value})"),
        DType::F16 | DType::BF16 => {
            let encoded = narrow::encode(dtype, value).expect("validated narrow reduction dtype");
            narrow::decode(dtype, &encoded).expect("validated narrow reduction dtype")
        }
        _ => format!("({})({value})", cl_type(dtype)),
    }
}

fn reduction_storage_expr(dtype: DType, value: &str) -> String {
    match dtype {
        DType::I32 => format!("as_int({value})"),
        DType::I64 => format!("as_long({value})"),
        _ => value.into(),
    }
}

fn reduction_finalize_commit_expr(dtype: DType, value: &str, has_epilogue: bool) -> String {
    if has_epilogue && matches!(dtype, DType::F16 | DType::BF16) {
        narrow::quantize(dtype, value).expect("validated narrow reduction output dtype")
    } else {
        reduction_storage_expr(dtype, value)
    }
}

pub(super) fn broadcast_offset(
    input: &Shape,
    output: &Shape,
    linear: &str,
) -> Result<String, OpenClError> {
    if input.rank() > output.rank() {
        return Err(OpenClError::Unsupported(
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
        let out_dim = output.dims()[pad + axis];
        if dim != 1 && dim != out_dim {
            return Err(OpenClError::Unsupported(
                "invalid broadcast metadata".into(),
            ));
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
