//! Deterministic Metal Shading Language lowering for static F32/Bool UOps.
use super::{MetalCapabilities, MetalError};
use crate::{DType, ScheduleInputBinding, Shape, UArg, UOp, UOpKind, ViewMap};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

pub const METAL_RENDERER_VERSION: &str = "rustgrad-metal-static-v1";
pub const METAL_ABI_VERSION: u32 = 1;

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
    pub view: Option<ViewMap>,
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
    schedule_inputs: Vec<MetalBufferAbi>,
    pub(super) semantic_program: Arc<UOp>,
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

    /// Lowers a validated scheduled UOp into the exact F32/Bool static subset.
    pub fn render(&self, root: &UOp) -> Result<RenderedMetal, MetalError> {
        root.validate()
            .map_err(|error| MetalError::Unsupported(error.to_string()))?;
        let nodes = root
            .topological()
            .map_err(|error| MetalError::Unsupported(error.to_string()))?;
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
            return Err(MetalError::Unsupported(
                "reductions, effects, and control flow are outside the exact Metal subset".into(),
            ));
        }
        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.kind(), UOpKind::Store))
            .ok_or_else(|| MetalError::Unsupported("sink has no store".into()))?;
        let output_index = store
            .sources()
            .first()
            .ok_or_else(|| MetalError::Unsupported("store has no index".into()))?;
        let UArg::BufferIndex {
            buffer: output_id,
            elements: extent,
            input_shape: output_shape,
            output_shape: store_shape,
        } = output_index.arg()
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
            let (buffer, source_shape, elements, view) = match node.arg() {
                UArg::BufferIndex {
                    buffer,
                    elements,
                    input_shape,
                    ..
                } => (*buffer, input_shape.clone(), *elements, None),
                UArg::ViewBufferIndex { buffer, view, .. } => {
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
            if !matches!(node.kind(), UOpKind::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| MetalError::InvalidBinding("load lacks index".into()))?;
            let buffer = match index.arg() {
                UArg::BufferIndex { buffer, .. } | UArg::ViewBufferIndex { buffer, .. } => *buffer,
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
        lines.push("    uint gid [[thread_position_in_grid]]) {".into());
        lines.push("  if ((ulong)gid >= extent) return;".into());
        let value = store
            .sources()
            .get(1)
            .ok_or_else(|| MetalError::Unsupported("store has no value".into()))?;
        let mut source_map = BTreeMap::new();
        let expression = emit_expr(value, &ids, &mut source_map, &mut lines, "(ulong)gid")?;
        let stored = if output_dtype == DType::Bool {
            format!("(uchar)(({expression}) != 0)")
        } else {
            expression
        };
        lines.push(format!("  b{output_position}[gid] = {stored};"));
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
        ));
        Ok(RenderedMetal {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            capabilities: self.capabilities.clone(),
            schedule_inputs,
            semantic_program: Arc::new(root.clone()),
        })
    }
}

fn supported_storage(dtype: DType) -> Result<(), MetalError> {
    match dtype {
        DType::F32 | DType::Bool => Ok(()),
        _ => Err(MetalError::Unsupported(format!(
            "dtype {dtype:?} is outside the exact Metal static subset"
        ))),
    }
}

fn metal_storage_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::Bool => "uchar",
        _ => unreachable!("validated Metal storage"),
    }
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
) -> Result<String, MetalError> {
    let map_id = source_map.len();
    source_map.insert(map_id, lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| MetalError::Unsupported(format!("untyped {:?}", node.kind())))?
        .scalar;
    supported_storage(dtype)?;
    let child =
        |position: usize, source_map: &mut BTreeMap<usize, usize>, lines: &mut Vec<String>| {
            node.sources()
                .get(position)
                .ok_or_else(|| MetalError::Unsupported("missing expression operand".into()))
                .and_then(|source| emit_expr(source, ids, source_map, lines, linear))
        };
    match node.kind() {
        UOpKind::Const => match node.arg() {
            UArg::Scalar {
                dtype: DType::F32,
                bits,
            } => Ok(format!("as_type<float>((uint)0x{:08x}u)", *bits as u32)),
            UArg::Scalar {
                dtype: DType::Bool,
                bits,
            } if *bits <= 1 => Ok(format!("(uchar){bits}u")),
            _ => Err(MetalError::Unsupported(
                "invalid F32/Bool scalar literal".into(),
            )),
        },
        UOpKind::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| MetalError::Unsupported("load has no index".into()))?;
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
        UOpKind::Cast => {
            let value = child(0, source_map, lines)?;
            let source = node.sources()[0]
                .ty()
                .ok_or_else(|| MetalError::Unsupported("untyped cast source".into()))?
                .scalar;
            match (source, dtype) {
                (DType::F32, DType::F32) | (DType::Bool, DType::Bool) => Ok(value),
                (DType::Bool, DType::F32) => Ok(format!("(float)({value} != 0)")),
                (DType::F32, DType::Bool) => Ok(format!("(uchar)({value} != 0.0f)")),
                _ => Err(MetalError::Unsupported(
                    "cast is outside the F32/Bool Metal subset".into(),
                )),
            }
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
                Eq => Ok(format!("(uchar)(({lhs}) == ({rhs}))")),
                Lt => Ok(format!("(uchar)(({lhs}) < ({rhs}))")),
                Le => Ok(format!("(uchar)(({lhs}) <= ({rhs}))")),
                _ => Err(MetalError::Unsupported(format!(
                    "core binary {op:?} is outside the Metal subset"
                ))),
            }
        }
        UOpKind::GraphCompare(op) => {
            let lhs = child(0, source_map, lines)?;
            let rhs = child(1, source_map, lines)?;
            let operator = match op {
                crate::CompareOp::Eq => "==",
                crate::CompareOp::Ne => "!=",
                crate::CompareOp::Lt => "<",
                crate::CompareOp::Le => "<=",
                crate::CompareOp::Gt => ">",
                crate::CompareOp::Ge => ">=",
            };
            Ok(format!("(uchar)(({lhs}) {operator} ({rhs}))"))
        }
        UOpKind::GraphLogical(op) => {
            let lhs = child(0, source_map, lines)?;
            Ok(match op {
                crate::LogicalOp::Not => format!("(uchar)!({lhs})"),
                crate::LogicalOp::And => {
                    let rhs = child(1, source_map, lines)?;
                    format!("(uchar)(({lhs}) && ({rhs}))")
                }
                crate::LogicalOp::Or => {
                    let rhs = child(1, source_map, lines)?;
                    format!("(uchar)(({lhs}) || ({rhs}))")
                }
            })
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            let condition = child(0, source_map, lines)?;
            let yes = child(1, source_map, lines)?;
            let no = child(2, source_map, lines)?;
            Ok(format!("(({condition}) ? ({yes}) : ({no}))"))
        }
        other => Err(MetalError::Unsupported(format!("{other:?}"))),
    }
}

fn emit_binary(
    op: crate::BinaryOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, MetalError> {
    use crate::BinaryOp::{Add, Mul, Sub};
    match (op, dtype) {
        (Add | Sub | Mul, DType::F32) => {
            let operator = match op {
                Add => "+",
                Sub => "-",
                Mul => "*",
                _ => unreachable!(),
            };
            Ok(format!("(({lhs}) {operator} ({rhs}))"))
        }
        (Add, DType::Bool) => Ok(format!("(uchar)(({lhs}) || ({rhs}))")),
        (Sub, DType::Bool) => Ok(format!("(uchar)(({lhs}) != ({rhs}))")),
        (Mul, DType::Bool) => Ok(format!("(uchar)(({lhs}) && ({rhs}))")),
        _ => Err(MetalError::Unsupported(format!(
            "binary {op:?} for {dtype:?} is outside the Metal subset"
        ))),
    }
}

#[derive(Clone, Debug)]
struct MetalViewAccess {
    source_shape: Shape,
    logical_shape: Shape,
    strides: Vec<usize>,
    offset: usize,
}

impl MetalViewAccess {
    fn new(view: &ViewMap) -> Result<Self, MetalError> {
        if view.logical_shape.rank() != view.strides.len() {
            return Err(MetalError::Unsupported("view rank/stride mismatch".into()));
        }
        let source_elements = view
            .source_shape
            .numel()
            .map_err(|_| MetalError::Overflow)?;
        let logical_elements = view
            .logical_shape
            .numel()
            .map_err(|_| MetalError::Overflow)?;
        if logical_elements != 0 {
            let last = view
                .element_offset(logical_elements - 1)
                .map_err(|_| MetalError::Unsupported("view exceeds source storage".into()))?;
            if last >= source_elements {
                return Err(MetalError::Unsupported(
                    "view exceeds source storage".into(),
                ));
            }
        } else if view.offset > source_elements {
            return Err(MetalError::Unsupported(
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

    fn expression(&self, logical: &str) -> String {
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
}

fn broadcast_offset(input: &Shape, output: &Shape, linear: &str) -> Result<String, MetalError> {
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
