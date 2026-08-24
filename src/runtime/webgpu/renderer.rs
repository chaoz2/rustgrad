//! Deterministic WGSL lowering for a static exact-storage elementwise subset.
use super::{
    WebGpuCapabilities, WebGpuError,
    guard::emit_transactional,
    narrow::{self, WEBGPU_NARROW_ABI_VERSION},
    transaction::WebGpuTransactionAbi,
};
use crate::{AffineView, DType, ScheduleInputBinding, Shape, UArg, UOp, UOpKind, ViewMap};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Deterministic renderer/source identity.
pub const WGSL_RENDERER_VERSION: &str = "rustgrad-wgsl-static-v3";
/// Ordered storage-plus-extent bind-group ABI version.
pub const WEBGPU_ABI_VERSION: u32 = 3;
/// Guarded candidate/status ABI version included in source and cache identity.
pub const WEBGPU_STATUS_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// One ordered storage-buffer entry in the WGSL bind-group ABI.
pub struct WgslBufferAbi {
    /// Stable scheduled buffer identity.
    pub id: u64,
    /// Exact logical storage dtype.
    pub dtype: DType,
    /// Physical source-storage shape.
    pub source_shape: Shape,
    /// Logical source-storage element count.
    pub elements: usize,
    /// Whether this is the unique output binding.
    pub mutable: bool,
    /// Optional source-backed affine logical mapping.
    pub view: Option<ViewMap>,
}

impl WgslBufferAbi {
    /// Logical RustGrad bytes. Native allocation rounds this value to four.
    pub fn logical_bytes(&self) -> Result<usize, WebGpuError> {
        self.elements
            .checked_mul(self.dtype.itemsize())
            .ok_or(WebGpuError::Overflow)
    }

    /// Native storage size after WebGPU's required four-byte rounding.
    pub fn physical_bytes(&self) -> Result<usize, WebGpuError> {
        let logical = self.logical_bytes()?;
        Ok(logical.checked_add(3).ok_or(WebGpuError::Overflow)? / 4 * 4)
    }
}

#[derive(Clone, Debug)]
/// Immutable WGSL source plus its complete checked launch contract.
pub struct RenderedWgsl {
    /// Deterministically emitted WGSL source.
    pub source: String,
    /// Expression IDs to one-based source lines.
    pub source_map: BTreeMap<usize, usize>,
    /// Ordered inputs followed by the unique output.
    pub buffers: Vec<WgslBufferAbi>,
    /// Logical output element count supplied through the final uniform.
    pub extent: usize,
    /// Generated compute entry point.
    pub entry: String,
    /// Content-addressed renderer/capability/ABI identity.
    pub cache_key: String,
    /// Exact adapter capabilities used for rendering.
    pub capabilities: WebGpuCapabilities,
    /// Checked workgroup width encoded in source and launch metadata.
    pub local_size: u32,
    /// Guard/status metadata when output must commit transactionally.
    pub transaction: Option<WebGpuTransactionAbi>,
    pub(super) schedule_inputs: Vec<WgslBufferAbi>,
    pub(super) semantic_program: Arc<super::dispatch::KernelSemanticProgram>,
}

impl RenderedWgsl {
    /// Validates schedule-owned first-load ordering against the bind-group ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), WebGpuError> {
        if bindings.len() != self.schedule_inputs.len() {
            return Err(WebGpuError::InvalidBinding(
                "schedule/WebGPU input count mismatch".into(),
            ));
        }
        for (position, (binding, expected)) in
            bindings.iter().zip(&self.schedule_inputs).enumerate()
        {
            let binding_view = binding.desc.view.as_ref().map(unsigned_view).transpose()?;
            if binding.abi_index != position
                || binding.desc.id != expected.id
                || binding.desc.dtype != expected.dtype
                || binding.desc.shape != expected.source_shape
                || binding_view != expected.view
                || binding.desc.bytes != expected.logical_bytes()?
            {
                return Err(WebGpuError::InvalidBinding(format!(
                    "schedule binding {position} mismatches WebGPU ABI"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn validate_artifact(&self) -> Result<(), WebGpuError> {
        if self.buffers.is_empty()
            || self.buffers.last().is_none_or(|buffer| !buffer.mutable)
            || self.buffers[..self.buffers.len() - 1]
                .iter()
                .any(|buffer| buffer.mutable)
        {
            return Err(WebGpuError::InvalidBinding(
                "artifact requires one final mutable output binding".into(),
            ));
        }
        if self.buffers.len() > self.capabilities.max_storage_buffers_per_shader_stage as usize
            || self.local_size == 0
            || self.local_size > self.capabilities.max_compute_workgroup_size_x
            || self.extent > u32::MAX as usize
        {
            return Err(WebGpuError::InvalidBinding(
                "artifact capability or indexing metadata mismatch".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for buffer in &self.buffers {
            supported_storage(buffer.dtype)?;
            let source_elements = buffer
                .source_shape
                .numel()
                .map_err(|_| WebGpuError::Overflow)?;
            if source_elements != buffer.elements
                || !ids.insert(buffer.id)
                || buffer.physical_bytes()? > self.capabilities.max_buffer_size
            {
                return Err(WebGpuError::InvalidBinding(
                    "artifact buffer storage metadata mismatch".into(),
                ));
            }
            if let Some(view) = &buffer.view {
                let access = WgslViewAccess::new(view)?;
                if access.source_shape != buffer.source_shape {
                    return Err(WebGpuError::InvalidBinding(
                        "artifact affine source shape mismatch".into(),
                    ));
                }
            }
        }
        if self.extent
            != self
                .buffers
                .last()
                .expect("nonempty checked above")
                .elements
        {
            return Err(WebGpuError::InvalidBinding(
                "artifact output extent mismatch".into(),
            ));
        }
        let Some(transaction) = &self.transaction else {
            return Ok(());
        };
        if transaction.output_abi_index >= self.buffers.len()
            || !self.buffers[transaction.output_abi_index].mutable
            || self.buffers.len() + 1
                > self.capabilities.max_storage_buffers_per_shader_stage as usize
        {
            return Err(WebGpuError::InvalidBinding(
                "transaction artifact binding mismatch".into(),
            ));
        }
        transaction.validate_launch(self.extent, transaction.output_abi_index)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Pure renderer bound to one immutable adapter capability identity.
pub struct WgslRenderer {
    /// Checked X workgroup width.
    pub local_size: u32,
    /// Adapter capability identity included in output identity.
    pub capabilities: WebGpuCapabilities,
}

impl WgslRenderer {
    /// Creates a renderer after validating the static workgroup width.
    pub fn new(local_size: u32, capabilities: WebGpuCapabilities) -> Result<Self, WebGpuError> {
        if local_size == 0 {
            return Err(WebGpuError::InvalidArgument("zero local size"));
        }
        if local_size > capabilities.max_compute_workgroup_size_x {
            return Err(WebGpuError::InvalidArgument(
                "local size exceeds adapter workgroup limit",
            ));
        }
        Ok(Self {
            local_size,
            capabilities,
        })
    }

    /// Lowers a validated scheduled UOp without executing or allocating.
    pub fn render(&self, root: &UOp) -> Result<RenderedWgsl, WebGpuError> {
        if matches!(root.kind(), UOpKind::Random) {
            let UArg::Random(plan) = root.arg() else {
                return Err(WebGpuError::Unsupported("random payload is absent".into()));
            };
            return super::random::render(self, plan);
        }
        root.validate()
            .map_err(|error| WebGpuError::Unsupported(error.to_string()))?;
        let nodes = root
            .topological()
            .map_err(|error| WebGpuError::Unsupported(error.to_string()))?;
        if nodes.iter().any(|node| {
            matches!(
                node.kind(),
                UOpKind::ReduceInit
                    | UOpKind::ReduceAccumulate
                    | UOpKind::ReduceFinalize
                    | UOpKind::Barrier
                    | UOpKind::If
                    | UOpKind::EndIf
            )
        }) {
            return Err(WebGpuError::Unsupported(
                "reductions, effects, and control flow are outside the exact WGSL subset".into(),
            ));
        }
        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.kind(), UOpKind::Store))
            .ok_or_else(|| WebGpuError::Unsupported("sink has no store".into()))?;
        let output_index = store
            .sources()
            .first()
            .ok_or_else(|| WebGpuError::Unsupported("store has no index".into()))?;
        let UArg::BufferIndex {
            buffer: output_id,
            elements: extent,
            input_shape: output_shape,
            output_shape: store_shape,
        } = output_index.arg()
        else {
            return Err(WebGpuError::Unsupported(
                "output requires a contiguous BufferIndex".into(),
            ));
        };
        if output_shape != store_shape {
            return Err(WebGpuError::Unsupported(
                "non-contiguous output addressing".into(),
            ));
        }
        if *extent > u32::MAX as usize {
            return Err(WebGpuError::Unsupported(
                "extent exceeds WGSL u32 indexing".into(),
            ));
        }
        let output_dtype = output_index
            .ty()
            .ok_or_else(|| WebGpuError::Unsupported("untyped output index".into()))?
            .scalar;
        supported_storage(output_dtype)?;

        let mut inventory = BTreeMap::<u64, WgslBufferAbi>::new();
        for node in &nodes {
            let (buffer, source_shape, elements, view) = match node.arg() {
                UArg::BufferIndex {
                    buffer,
                    elements,
                    input_shape,
                    ..
                } => (*buffer, input_shape.clone(), *elements, None),
                UArg::ViewBufferIndex { buffer, view, .. } => {
                    let view = unsigned_view(view)?;
                    let access = WgslViewAccess::new(&view)?;
                    let elements = access
                        .source_shape
                        .numel()
                        .map_err(|_| WebGpuError::Overflow)?;
                    (*buffer, access.source_shape, elements, Some(view))
                }
                _ => continue,
            };
            let dtype = node
                .ty()
                .ok_or_else(|| WebGpuError::Unsupported("untyped buffer index".into()))?
                .scalar;
            supported_storage(dtype)?;
            let abi = WgslBufferAbi {
                id: buffer,
                dtype,
                source_shape,
                elements,
                mutable: buffer == *output_id,
                view,
            };
            abi.logical_bytes()?;
            if let Some(previous) = inventory.insert(buffer, abi.clone())
                && previous != abi
            {
                return Err(WebGpuError::InvalidBinding(format!(
                    "buffer {buffer} has conflicting ABI metadata"
                )));
            }
        }

        let mut seen = BTreeSet::new();
        let mut schedule_inputs = Vec::new();
        for node in &nodes {
            if !matches!(node.kind(), UOpKind::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| WebGpuError::InvalidBinding("load lacks index".into()))?;
            let buffer = match index.arg() {
                UArg::BufferIndex { buffer, .. } | UArg::ViewBufferIndex { buffer, .. } => *buffer,
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            if seen.insert(buffer) {
                schedule_inputs.push(
                    inventory
                        .get(&buffer)
                        .ok_or_else(|| WebGpuError::InvalidBinding("load ABI missing".into()))?
                        .clone(),
                );
            }
        }
        let mut buffers = schedule_inputs.clone();
        if seen.insert(*output_id) {
            buffers.push(
                inventory
                    .get(output_id)
                    .ok_or_else(|| WebGpuError::InvalidBinding("output ABI missing".into()))?
                    .clone(),
            );
        }
        if buffers.last().is_none_or(|buffer| buffer.id != *output_id) {
            return Err(WebGpuError::InvalidBinding(
                "output aliases an input buffer".into(),
            ));
        }
        if buffers.len() > self.capabilities.max_storage_buffers_per_shader_stage as usize {
            return Err(WebGpuError::Unsupported(
                "ordered bindings exceed adapter storage-buffer limit".into(),
            ));
        }
        for abi in &buffers {
            if abi.physical_bytes()? > self.capabilities.max_buffer_size {
                return Err(WebGpuError::Unsupported(
                    "binding exceeds adapter buffer limit".into(),
                ));
            }
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
            .ok_or_else(|| WebGpuError::Unsupported("store has no value".into()))?;
        let transaction =
            WebGpuTransactionAbi::analyze(value, output_position, store_shape.clone())?;
        if transaction.is_some()
            && buffers.len() + 1 > self.capabilities.max_storage_buffers_per_shader_stage as usize
        {
            return Err(WebGpuError::Unsupported(
                "transaction status exceeds adapter storage-buffer limit".into(),
            ));
        }
        let entry = format!("rg_webgpu_e{}_b{}", extent, buffers.len());
        let uses_narrow = nodes
            .iter()
            .any(|node| node.ty().is_some_and(|ty| narrow::is_narrow(ty.scalar)));
        let mut lines = vec![
            format!(
                "// {WGSL_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION} STATUS {WEBGPU_STATUS_VERSION} NARROW {WEBGPU_NARROW_ABI_VERSION}"
            ),
            "struct RustGradExtent { value: u32, };".into(),
            "fn rg_f32_to_i32(value: f32) -> i32 {".into(),
            "  if (isNan(value)) { return 0i; }".into(),
            "  if (value >= 2147483648.0) { return bitcast<i32>(0x7fffffffu); }".into(),
            "  if (value <= -2147483648.0) { return bitcast<i32>(0x80000000u); }".into(),
            "  return i32(value);".into(),
            "}".into(),
            "fn rg_f32_to_u32(value: f32) -> u32 {".into(),
            "  if (isNan(value) || value <= 0.0) { return 0u; }".into(),
            "  if (value >= 4294967296.0) { return 0xffffffffu; }".into(),
            "  return u32(value);".into(),
            "}".into(),
            "fn rg_i32_trunc_div(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return bitcast<i32>(0x80000000u); }".into(),
            "  return lhs / rhs;".into(),
            "}".into(),
            "fn rg_i32_fmod(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return 0i; }".into(),
            "  return lhs % rhs;".into(),
            "}".into(),
            "fn rg_i32_floor_div(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return bitcast<i32>(0x80000000u); }".into(),
            "  let quotient: i32 = lhs / rhs;".into(),
            "  let remainder: i32 = lhs % rhs;".into(),
            "  if (remainder < 0i) { return quotient - select(-1i, 1i, rhs > 0i); }".into(),
            "  return quotient;".into(),
            "}".into(),
            "fn rg_i32_mod(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return 0i; }".into(),
            "  let remainder: i32 = lhs % rhs;".into(),
            "  if (remainder < 0i) {".into(),
            "    let magnitude: u32 = select(bitcast<u32>(rhs), 0u - bitcast<u32>(rhs), rhs < 0i);".into(),
            "    return bitcast<i32>(bitcast<u32>(remainder) + magnitude);".into(),
            "  }".into(),
            "  return remainder;".into(),
            "}".into(),
        ];
        if uses_narrow {
            lines.push(narrow::SOURCE.into());
        }
        for (position, buffer) in buffers.iter().enumerate() {
            let access = if buffer.mutable { "read_write" } else { "read" };
            let storage = wgsl_storage_decl(buffer.dtype, buffer.mutable);
            lines.push(format!(
                "@group(0) @binding({position}) var<storage, {access}> b{position}: array<{storage}>;"
            ));
        }
        lines.push(format!(
            "@group(0) @binding({}) var<uniform> rg_extent: RustGradExtent;",
            buffers.len()
        ));
        if transaction.is_some() {
            lines.push("struct RustGradStatus { value: atomic<u32>, };".into());
            lines.push(format!(
                "@group(0) @binding({}) var<storage, read_write> rg_status: RustGradStatus;",
                buffers.len() + 1
            ));
        }
        lines.push(format!(
            "@compute @workgroup_size({}, 1, 1)",
            self.local_size
        ));
        lines.push(format!(
            "fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"
        ));
        lines.push("  let gid: u32 = rg_global.x;".into());
        lines.push("  if (gid >= rg_extent.value) { return; }".into());
        let mut source_map = BTreeMap::new();
        let expression = if let Some(transaction) = &transaction {
            emit_transactional(value, transaction, &ids, &mut source_map, &mut lines)?
        } else {
            emit_expr(value, &ids, &mut source_map, &mut lines, "gid")?
        };
        if output_dtype == DType::Bool {
            if transaction.is_some() {
                lines.push("  if (rg_ok) {".into());
            }
            let indent = if transaction.is_some() { "    " } else { "  " };
            lines.push(format!("{indent}let rg_shift: u32 = (gid & 3u) * 8u;"));
            lines.push(format!(
                "{indent}atomicAnd(&b{output_position}[gid >> 2u], ~(0xffu << rg_shift));"
            ));
            lines.push(format!(
                "{indent}atomicOr(&b{output_position}[gid >> 2u], select(0u, 1u, {expression}) << rg_shift);"
            ));
            if transaction.is_some() {
                lines.push("  }".into());
            }
        } else if narrow::is_narrow(output_dtype) {
            if transaction.is_some() {
                return Err(WebGpuError::Unsupported(
                    "guarded narrow-float output is outside the exact WGSL subset".into(),
                ));
            }
            let encoded =
                narrow::encode(output_dtype, &expression).expect("validated narrow output dtype");
            lines.push("  let rg_shift: u32 = (gid & 1u) * 16u;".into());
            lines.push(format!(
                "  atomicAnd(&b{output_position}[gid >> 1u], ~(0xffffu << rg_shift));"
            ));
            lines.push(format!(
                "  atomicOr(&b{output_position}[gid >> 1u], ({encoded} & 0xffffu) << rg_shift);"
            ));
        } else {
            lines.push(if transaction.is_some() {
                format!("  if (rg_ok) {{ b{output_position}[gid] = {expression}; }}")
            } else {
                format!("  b{output_position}[gid] = {expression};")
            });
        }
        lines.push("}".into());
        let source = lines.join("\n") + "\n";
        let cache_key = stable_key(&(
            WGSL_RENDERER_VERSION,
            WEBGPU_ABI_VERSION,
            WEBGPU_STATUS_VERSION,
            WEBGPU_NARROW_ABI_VERSION,
            self.local_size,
            &self.capabilities,
            &source,
            &buffers,
            &schedule_inputs,
            &transaction,
        ));
        Ok(RenderedWgsl {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            capabilities: self.capabilities.clone(),
            local_size: self.local_size,
            transaction,
            schedule_inputs,
            semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
                root.clone(),
            ))),
        })
    }
}

fn supported_storage(dtype: DType) -> Result<(), WebGpuError> {
    match dtype {
        DType::F16 | DType::BF16 | DType::F32 | DType::Bool | DType::I32 | DType::U32 => Ok(()),
        _ => Err(WebGpuError::Unsupported(format!(
            "dtype {dtype:?} is outside the exact WGSL static subset"
        ))),
    }
}

fn wgsl_storage_decl(dtype: DType, mutable: bool) -> &'static str {
    match (dtype, mutable) {
        (DType::F32, _) => "f32",
        (DType::I32, _) => "i32",
        (DType::U32, _) => "u32",
        (DType::Bool, true) => "atomic<u32>",
        (DType::Bool, false) => "u32",
        (DType::F16 | DType::BF16, true) => "atomic<u32>",
        (DType::F16 | DType::BF16, false) => "u32",
        _ => unreachable!("validated WGSL storage"),
    }
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
) -> Result<String, WebGpuError> {
    source_map.insert(source_map.len(), lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| WebGpuError::Unsupported(format!("untyped {:?}", node.kind())))?
        .scalar;
    supported_storage(dtype)?;
    let child =
        |position: usize, source_map: &mut BTreeMap<usize, usize>, lines: &mut Vec<String>| {
            node.sources()
                .get(position)
                .ok_or_else(|| WebGpuError::Unsupported("missing expression operand".into()))
                .and_then(|source| emit_expr(source, ids, source_map, lines, linear))
        };
    match node.kind() {
        UOpKind::Const => match node.arg() {
            UArg::Scalar {
                dtype: DType::F32,
                bits,
            } => Ok(format!("bitcast<f32>(0x{:08x}u)", *bits as u32)),
            UArg::Scalar {
                dtype: DType::Bool,
                bits,
            } if *bits <= 1 => Ok(if *bits == 0 {
                "false".into()
            } else {
                "true".into()
            }),
            UArg::Scalar {
                dtype: DType::I32,
                bits,
            } => Ok(format!("bitcast<i32>(0x{:08x}u)", *bits as u32)),
            UArg::Scalar {
                dtype: DType::U32,
                bits,
            } => Ok(format!("0x{:08x}u", *bits as u32)),
            UArg::Scalar { dtype, bits } if narrow::is_narrow(*dtype) => {
                Ok(narrow::decode(*dtype, format!("0x{:04x}u", *bits as u16))
                    .expect("validated narrow scalar"))
            }
            _ => Err(WebGpuError::Unsupported(
                "invalid WGSL scalar literal".into(),
            )),
        },
        UOpKind::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| WebGpuError::Unsupported("load has no index".into()))?;
            let (buffer, input_shape, output_shape, view) = match index.arg() {
                UArg::BufferIndex {
                    buffer,
                    input_shape,
                    output_shape,
                    ..
                } => (*buffer, input_shape, output_shape, None),
                UArg::ViewBufferIndex {
                    buffer,
                    input_shape,
                    output_shape,
                    view,
                    ..
                } => (*buffer, input_shape, output_shape, Some(view)),
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            let position = ids
                .get(&buffer)
                .ok_or_else(|| WebGpuError::InvalidBinding("load buffer absent from ABI".into()))?;
            let logical = broadcast_offset(input_shape, output_shape, linear)?;
            let offset = match view {
                Some(view) => WgslViewAccess::new(&unsigned_view(view)?)?.expression(&logical),
                None => logical,
            };
            if dtype == DType::Bool {
                Ok(format!(
                    "(((b{position}[({offset}) >> 2u] >> ((({offset}) & 3u) * 8u)) & 0xffu) != 0u)"
                ))
            } else if narrow::is_narrow(dtype) {
                let raw = format!(
                    "((b{position}[({offset}) >> 1u] >> ((({offset}) & 1u) * 16u)) & 0xffffu)"
                );
                Ok(narrow::decode(dtype, raw).expect("validated narrow load"))
            } else {
                Ok(format!("b{position}[{offset}]"))
            }
        }
        UOpKind::Cast => {
            let value = child(0, source_map, lines)?;
            let source = node.sources()[0]
                .ty()
                .ok_or_else(|| WebGpuError::Unsupported("untyped cast source".into()))?
                .scalar;
            emit_cast(source, dtype, &value)
        }
        UOpKind::GraphBinary(op) => {
            let lhs = child(0, source_map, lines)?;
            let rhs = child(1, source_map, lines)?;
            emit_binary(*op, dtype, &lhs, &rhs)
        }
        UOpKind::Binary(op) => {
            let lhs = child(0, source_map, lines)?;
            let rhs = child(1, source_map, lines)?;
            use crate::uop::Binary::{Add, Eq, Le, Lt, Mul, Sub};
            match op {
                Add => emit_binary(crate::BinaryOp::Add, dtype, &lhs, &rhs),
                Sub => emit_binary(crate::BinaryOp::Sub, dtype, &lhs, &rhs),
                Mul => emit_binary(crate::BinaryOp::Mul, dtype, &lhs, &rhs),
                Eq => Ok(format!("(({lhs}) == ({rhs}))")),
                Lt | Le => {
                    let operand_dtype = node.sources()[0]
                        .ty()
                        .ok_or_else(|| WebGpuError::Unsupported("untyped compare source".into()))?
                        .scalar;
                    let lhs = ordered_compare_operand(operand_dtype, &lhs);
                    let rhs = ordered_compare_operand(operand_dtype, &rhs);
                    let operator = if matches!(op, Lt) { "<" } else { "<=" };
                    Ok(format!("(({lhs}) {operator} ({rhs}))"))
                }
                _ => Err(WebGpuError::Unsupported(format!(
                    "core binary {op:?} is outside the WGSL subset"
                ))),
            }
        }
        UOpKind::GraphCompare(op) => {
            let lhs = child(0, source_map, lines)?;
            let rhs = child(1, source_map, lines)?;
            let operand_dtype = node.sources()[0]
                .ty()
                .ok_or_else(|| WebGpuError::Unsupported("untyped compare source".into()))?
                .scalar;
            let operator = match op {
                crate::CompareOp::Eq => "==",
                crate::CompareOp::Ne => "!=",
                crate::CompareOp::Lt => "<",
                crate::CompareOp::Le => "<=",
                crate::CompareOp::Gt => ">",
                crate::CompareOp::Ge => ">=",
            };
            let lhs = if matches!(op, crate::CompareOp::Eq | crate::CompareOp::Ne) {
                lhs
            } else {
                ordered_compare_operand(operand_dtype, &lhs)
            };
            let rhs = if matches!(op, crate::CompareOp::Eq | crate::CompareOp::Ne) {
                rhs
            } else {
                ordered_compare_operand(operand_dtype, &rhs)
            };
            Ok(format!("(({lhs}) {operator} ({rhs}))"))
        }
        UOpKind::GraphLogical(op) => {
            let lhs = child(0, source_map, lines)?;
            Ok(match op {
                crate::LogicalOp::Not => format!("!({lhs})"),
                crate::LogicalOp::And => {
                    let rhs = child(1, source_map, lines)?;
                    format!("(({lhs}) && ({rhs}))")
                }
                crate::LogicalOp::Or => {
                    let rhs = child(1, source_map, lines)?;
                    format!("(({lhs}) || ({rhs}))")
                }
            })
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            let condition = child(0, source_map, lines)?;
            let yes = child(1, source_map, lines)?;
            let no = child(2, source_map, lines)?;
            let selected = format!("select(({no}), ({yes}), ({condition}))");
            Ok(narrow::quantize(dtype, &selected).unwrap_or(selected))
        }
        other => Err(WebGpuError::Unsupported(format!("{other:?}"))),
    }
}

fn emit_cast(source: DType, target: DType, value: &str) -> Result<String, WebGpuError> {
    Ok(match (source, target) {
        (a, b) if a == b => value.into(),
        (DType::Bool, DType::F32) => format!("select(0.0, 1.0, {value})"),
        (DType::F32, DType::Bool) => format!("(({value}) != 0.0)"),
        (DType::Bool, DType::I32) => format!("select(0i, 1i, {value})"),
        (DType::Bool, DType::U32) => format!("select(0u, 1u, {value})"),
        (DType::I32, DType::Bool) => format!("(({value}) != 0i)"),
        (DType::U32, DType::Bool) => format!("(({value}) != 0u)"),
        (DType::I32, DType::U32) => format!("bitcast<u32>({value})"),
        (DType::U32, DType::I32) => format!("bitcast<i32>({value})"),
        (DType::I32, DType::F32) => format!("f32({value})"),
        (DType::U32, DType::F32) => format!("f32({value})"),
        (DType::F32, DType::I32) => format!("rg_f32_to_i32({value})"),
        (DType::F32, DType::U32) => format!("rg_f32_to_u32({value})"),
        (source, target)
            if narrow::is_narrow(target)
                && matches!(source, DType::F16 | DType::BF16 | DType::F32) =>
        {
            narrow::quantize(target, value).expect("validated narrow cast target")
        }
        (source, DType::F32) if narrow::is_narrow(source) => value.into(),
        _ => {
            return Err(WebGpuError::Unsupported(
                "cast is outside the exact WGSL subset".into(),
            ));
        }
    })
}

pub(super) fn ordered_compare_operand(dtype: DType, value: &str) -> String {
    if dtype == DType::Bool {
        format!("select(0u, 1u, {value})")
    } else {
        value.into()
    }
}

fn emit_binary(
    op: crate::BinaryOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, WebGpuError> {
    use crate::BinaryOp::{Add, Mul, Sub};
    let operator = match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        _ => {
            return Err(WebGpuError::Unsupported(format!(
                "binary {op:?} for {dtype:?} is outside the WGSL subset"
            )));
        }
    };
    let value = match dtype {
        DType::F16 | DType::BF16 | DType::F32 | DType::U32 => {
            format!("(({lhs}) {operator} ({rhs}))")
        }
        DType::I32 => format!("bitcast<i32>(bitcast<u32>({lhs}) {operator} bitcast<u32>({rhs}))"),
        DType::Bool => match op {
            Add => format!("(({lhs}) || ({rhs}))"),
            Sub => format!("(({lhs}) != ({rhs}))"),
            Mul => format!("(({lhs}) && ({rhs}))"),
            _ => unreachable!(),
        },
        _ => {
            return Err(WebGpuError::Unsupported(format!(
                "binary {op:?} for {dtype:?} is outside the WGSL subset"
            )));
        }
    };
    Ok(narrow::quantize(dtype, &value).unwrap_or(value))
}

#[derive(Clone, Debug)]
pub(super) struct WgslViewAccess {
    source_shape: Shape,
    logical_shape: Shape,
    strides: Vec<usize>,
    offset: usize,
}

/// Adapts the shared signed view descriptor to the unsigned WGSL ABI.
pub(super) fn unsigned_view(view: &AffineView) -> Result<ViewMap, WebGpuError> {
    view.as_unsigned().map_err(|_| {
        WebGpuError::Unsupported("signed affine views are outside the WebGPU static subset".into())
    })
}

impl WgslViewAccess {
    pub(super) fn new(view: &ViewMap) -> Result<Self, WebGpuError> {
        if view.logical_shape.rank() != view.strides.len() {
            return Err(WebGpuError::Unsupported("view rank/stride mismatch".into()));
        }
        let source_elements = view
            .source_shape
            .numel()
            .map_err(|_| WebGpuError::Overflow)?;
        if source_elements > u32::MAX as usize
            || view.offset > u32::MAX as usize
            || view
                .strides
                .iter()
                .any(|stride| *stride > u32::MAX as usize)
        {
            return Err(WebGpuError::Unsupported(
                "view exceeds WGSL u32 indexing".into(),
            ));
        }
        let logical_elements = view
            .logical_shape
            .numel()
            .map_err(|_| WebGpuError::Overflow)?;
        if logical_elements != 0 {
            let last = view
                .element_offset(logical_elements - 1)
                .map_err(|_| WebGpuError::Unsupported("view exceeds source storage".into()))?;
            if last >= source_elements {
                return Err(WebGpuError::Unsupported(
                    "view exceeds source storage".into(),
                ));
            }
        } else if view.offset > source_elements {
            return Err(WebGpuError::Unsupported(
                "empty view offset exceeds source storage".into(),
            ));
        }
        Ok(Self {
            source_shape: view.source_shape.clone(),
            logical_shape: view.logical_shape.clone(),
            strides: view.strides.clone(),
            offset: view.offset,
        })
    }

    pub(super) fn expression(&self, logical: &str) -> String {
        if self.logical_shape.numel().ok() == Some(1) {
            return format!("{}u", self.offset);
        }
        let logical_strides = self.logical_shape.contiguous_strides();
        let mut terms = Vec::new();
        if self.offset != 0 {
            terms.push(format!("{}u", self.offset));
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
                    "((({logical}) / {logical_stride}u) % {dim}u) * {stride}u"
                ));
            }
        }
        if terms.is_empty() {
            "0u".into()
        } else {
            format!("({})", terms.join(" + "))
        }
    }
}

pub(super) fn broadcast_offset(
    input: &Shape,
    output: &Shape,
    linear: &str,
) -> Result<String, WebGpuError> {
    if input.rank() > output.rank() {
        return Err(WebGpuError::Unsupported(
            "input rank exceeds output rank".into(),
        ));
    }
    if input.rank() == 0 {
        return Ok("0u".into());
    }
    let input_strides = input.contiguous_strides();
    let output_strides = output.contiguous_strides();
    if input_strides
        .iter()
        .chain(&output_strides)
        .any(|value| *value > u32::MAX as usize)
    {
        return Err(WebGpuError::Unsupported(
            "shape exceeds WGSL u32 indexing".into(),
        ));
    }
    let pad = output.rank() - input.rank();
    let mut terms = Vec::new();
    for axis in 0..input.rank() {
        let dim = input.dims()[axis];
        let output_dim = output.dims()[pad + axis];
        if dim != 1 && dim != output_dim {
            return Err(WebGpuError::Unsupported(
                "invalid broadcast metadata".into(),
            ));
        }
        if dim != 1 {
            terms.push(format!(
                "(({linear} / {}u) % {}u) * {}u",
                output_strides[pad + axis],
                dim,
                input_strides[axis]
            ));
        }
    }
    Ok(if terms.is_empty() {
        "0u".into()
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
    fn signed_affine_view_is_rejected_before_wgsl_lowering() {
        let view = AffineView {
            source_shape: Shape::from([4]),
            logical_shape: Shape::from([4]),
            strides: vec![-1],
            offset: 3,
        };
        assert!(matches!(
            unsigned_view(&view),
            Err(WebGpuError::Unsupported(_))
        ));
    }
}
