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

pub const METAL_RENDERER_VERSION: &str = "rustgrad-metal-static-v6";
pub const METAL_RAW_COPY_RENDERER_VERSION: &str = "rustgrad-metal-raw-copy-v1";
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
    /// Whether this entry is the unique output pointer.
    pub mutable: bool,
    /// Optional source-backed affine logical mapping.
    pub view: Option<AffineView>,
}

#[derive(Clone, Debug)]
/// Immutable MSL source plus the complete checked launch contract.
pub struct RenderedMetal {
    /// Deterministically emitted Metal Shading Language source.
    pub source: String,
    /// UOp expression IDs to one-based generated source lines.
    pub source_map: BTreeMap<usize, usize>,
    /// Ordered input pointers followed by the output pointer.
    pub buffers: Vec<MetalBufferAbi>,
    /// Logical output element count supplied as the final scalar ABI value.
    pub extent: usize,
    /// Generated MSL entry-point name.
    pub entry: String,
    /// Content-addressed renderer and capability identity.
    pub cache_key: String,
    /// Exact device capabilities used while rendering.
    pub capabilities: MetalCapabilities,
    /// Guard/status metadata when the output must be committed transactionally.
    pub transaction: Option<MetalTransactionAbi>,
    pub(super) schedule_inputs: Vec<MetalBufferAbi>,
    pub(super) semantic_program: Arc<super::dispatch::KernelSemanticProgram>,
}

impl RenderedMetal {
    /// Validates schedule-owned first-use ordering against the Metal pointer ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), MetalError> {
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
        if let Operation::Movement(value) = root.operation() {
            return match value {
                MovementValue::Plan(plan) => render_raw_copy(self, root, plan),
                MovementValue::QuantizedRowGather(_) => Err(MetalError::Unsupported(
                    "quantized movement is outside Metal contiguous-copy lowering".into(),
                )),
            };
        }
        if let Operation::Random(plan) = root.operation() {
            return super::random::render(self, plan);
        }
        if matches!(
            root.operation(),
            Operation::PrefixScan(_)
                | Operation::Sort(_)
                | Operation::TensorGuard(_)
                | Operation::Threefry(_)
        ) {
            return Err(MetalError::Unsupported(
                "prefix scans, sort pairs, guards, and live Threefry are outside Metal lowering"
                    .into(),
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

        let mut inventory = BTreeMap::<u64, MetalBufferAbi>::new();
        for node in &nodes {
            let (buffer, source_shape, elements, view) = match node.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements,
                    input_shape,
                    ..
                }) => (*buffer, input_shape.clone(), *elements, None),
                Operation::Index(IndexValue::View { buffer, view, .. }) => {
                    let access = MetalViewAccess::new(view)?;
                    let elements = access
                        .source_shape
                        .numel()
                        .map_err(|_| MetalError::Overflow)?;
                    (*buffer, access.source_shape, elements, Some(view.clone()))
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
                view,
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
            schedule_inputs,
            semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
                root.clone(),
            ))),
        })
    }
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
