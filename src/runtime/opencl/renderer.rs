//! Pure deterministic OpenCL C lowering for a deliberately small UOp subset.
use super::{OpenClCapabilities, OpenClError, reduction::OpenClReduction, view::OpenClViewAccess};
use crate::{DType, ScheduleInputBinding, Shape, UArg, UOp, UOpKind, ViewMap};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

pub const OPENCL_RENDERER_VERSION: &str = "rustgrad-opencl-static-v2";
pub const OPENCL_ABI_VERSION: u32 = 2;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenClBufferAbi {
    pub id: u64,
    pub dtype: DType,
    pub source_shape: Shape,
    pub elements: usize,
    pub mutable: bool,
    pub view: Option<ViewMap>,
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
    schedule_inputs: Vec<OpenClBufferAbi>,
    pub(crate) semantic_program: Arc<UOp>,
}

impl RenderedOpenCl {
    /// Validates the schedule-owned first-use order against the pointer ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), OpenClError> {
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
        let nodes = root
            .topological()
            .map_err(|error| OpenClError::Unsupported(error.to_string()))?;
        if nodes
            .iter()
            .any(|node| matches!(node.kind(), UOpKind::Barrier | UOpKind::If | UOpKind::EndIf))
        {
            return Err(OpenClError::Unsupported(
                "effects and barriers are outside the OpenCL static subset".into(),
            ));
        }
        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.kind(), UOpKind::Store))
            .ok_or_else(|| OpenClError::Unsupported("sink has no store".into()))?;
        let output_index = store
            .sources()
            .first()
            .ok_or_else(|| OpenClError::Unsupported("store has no index".into()))?;
        let UArg::BufferIndex {
            buffer: output_id,
            elements: extent,
            input_shape: output_shape,
            output_shape: store_shape,
        } = output_index.arg()
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
        supported_storage(output_dtype, self.capabilities)?;

        let mut inventory = BTreeMap::<u64, OpenClBufferAbi>::new();
        for node in &nodes {
            let (buffer, source_shape, elements, view) = match node.arg() {
                UArg::BufferIndex {
                    buffer,
                    elements,
                    input_shape,
                    ..
                } => (*buffer, input_shape.clone(), *elements, None),
                UArg::ViewBufferIndex { buffer, view, .. } => {
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
            supported_storage(dtype, self.capabilities)?;
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

        let store_value = store
            .sources()
            .get(1)
            .ok_or_else(|| OpenClError::Unsupported("store has no value".into()))?;
        let reduction = matches!(store_value.kind(), UOpKind::ReduceFinalize)
            .then(|| OpenClReduction::from_finalize(store_value))
            .transpose()?;
        let mut schedule_inputs = Vec::new();
        let mut seen = BTreeSet::new();
        for node in &nodes {
            if !matches!(node.kind(), UOpKind::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::InvalidBinding("load lacks index".into()))?;
            let buffer = match index.arg() {
                UArg::BufferIndex { buffer, .. } | UArg::ViewBufferIndex { buffer, .. } => *buffer,
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
        let mut buffers = if reduction
            .as_ref()
            .is_some_and(|reduction| reduction.reduction_len == 0)
        {
            Vec::new()
        } else {
            schedule_inputs.clone()
        };
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

        let entry = format!("rg_opencl_e{}_b{}", extent, buffers.len());
        let mut required_capabilities = required_capabilities(&buffers);
        if let Some(reduction) = &reduction {
            reduction.validate_dtype(output_dtype, self.capabilities)?;
            let reduction_capabilities = reduction.required_capabilities(output_dtype);
            required_capabilities.int64 |= reduction_capabilities.int64;
            required_capabilities.fp64 |= reduction_capabilities.fp64;
        }
        let mut lines = Vec::new();
        if required_capabilities.fp64 {
            lines.push("#pragma OPENCL EXTENSION cl_khr_fp64 : enable".into());
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
                store_value,
                output_dtype,
                output_position,
                &ids,
                &mut source_map,
                &mut lines,
                self.capabilities,
            )?;
        } else {
            let value = emit_expr(
                store_value,
                &ids,
                &mut source_map,
                &mut lines,
                "gid",
                self.capabilities,
            )?;
            lines.push(format!("  b{output_position}[gid] = {value};"));
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
            &reduction,
        ));
        Ok(RenderedOpenCl {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            required_capabilities,
            schedule_inputs,
            semantic_program: Arc::new(root.clone()),
        })
    }
}

fn supported_storage(dtype: DType, capabilities: OpenClCapabilities) -> Result<(), OpenClError> {
    match dtype {
        DType::Bool | DType::I32 | DType::U32 | DType::F32 => Ok(()),
        DType::I64 | DType::U64 if capabilities.int64 => Ok(()),
        DType::F64 if capabilities.fp64 => Ok(()),
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

fn cl_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "uchar",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::I64 => "long",
        DType::U64 => "ulong",
        DType::F32 => "float",
        DType::F64 => "double",
        _ => unreachable!("validated by supported_storage"),
    }
}

fn required_capabilities(buffers: &[OpenClBufferAbi]) -> OpenClCapabilities {
    OpenClCapabilities {
        int64: buffers
            .iter()
            .any(|buffer| matches!(buffer.dtype, DType::I64 | DType::U64)),
        fp64: buffers.iter().any(|buffer| buffer.dtype == DType::F64),
    }
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
    capabilities: OpenClCapabilities,
) -> Result<String, OpenClError> {
    let map_id = source_map.len();
    source_map.insert(map_id, lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| OpenClError::Unsupported(format!("untyped {:?}", node.kind())))?
        .scalar;
    supported_storage(dtype, capabilities)?;
    let child = |index: usize, source_map: &mut BTreeMap<usize, usize>, lines: &mut Vec<String>| {
        node.sources()
            .get(index)
            .ok_or_else(|| OpenClError::Unsupported("missing expression operand".into()))
            .and_then(|source| emit_expr(source, ids, source_map, lines, linear, capabilities))
    };
    match node.kind() {
        UOpKind::Const => match node.arg() {
            UArg::Scalar {
                dtype: DType::F32,
                bits,
            } => Ok(format!("as_float((uint)0x{:08x}u)", *bits as u32)),
            UArg::Scalar {
                dtype: DType::Bool,
                bits,
            } if *bits <= 1 => Ok(format!("((uchar){bits}u)")),
            UArg::Scalar {
                dtype: DType::I32,
                bits,
            } => Ok(format!("as_int((uint)0x{:08x}u)", *bits as u32)),
            UArg::Scalar {
                dtype: DType::U32,
                bits,
            } => Ok(format!("((uint)0x{:08x}u)", *bits as u32)),
            UArg::Scalar {
                dtype: DType::I64,
                bits,
            } => Ok(format!("as_long((ulong)0x{bits:016x}ul)")),
            UArg::Scalar {
                dtype: DType::U64,
                bits,
            } => Ok(format!("((ulong)0x{bits:016x}ul)")),
            UArg::Scalar {
                dtype: DType::F64,
                bits,
            } => Ok(format!("as_double((ulong)0x{bits:016x}ul)")),
            UArg::Scalar { .. } => Err(OpenClError::Unsupported(
                "scalar literal/type mismatch".into(),
            )),
            _ => Err(OpenClError::Unsupported("invalid scalar literal".into())),
        },
        UOpKind::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::Unsupported("load has no index".into()))?;
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
            Ok(format!("b{position}[{offset}]"))
        }
        UOpKind::Cast => {
            let value = child(0, source_map, lines)?;
            match (node.sources()[0].ty().map(|ty| ty.scalar), dtype) {
                (Some(source), target) if source == target => Ok(value),
                (Some(DType::Bool), target) => Ok(format!("(({})({value}))", cl_type(target))),
                (Some(source), DType::Bool) => {
                    Ok(format!("((uchar)(({value}) != ({})0))", cl_type(source)))
                }
                (Some(DType::I32), DType::U32) => Ok(format!("as_uint({value})")),
                (Some(DType::U32), DType::I32) => Ok(format!("as_int({value})")),
                (Some(DType::I64), DType::U64) => Ok(format!("as_ulong({value})")),
                (Some(DType::U64), DType::I64) => Ok(format!("as_long({value})")),
                (Some(DType::F32), DType::F64) => Ok(format!("((double)({value}))")),
                (Some(DType::F64), DType::F32) => Ok(format!("((float)({value}))")),
                _ => Err(OpenClError::Unsupported(
                    "cast is outside the exact OpenCL subset".into(),
                )),
            }
        }
        UOpKind::GraphUnary(op) => {
            let value = child(0, source_map, lines)?;
            match (op, dtype) {
                (crate::UnaryOp::Neg, DType::F32 | DType::F64) => Ok(format!("(-({value}))")),
                (crate::UnaryOp::Abs, DType::F32) => Ok(format!("fabs({value})")),
                (crate::UnaryOp::Abs, DType::F64) => Ok(format!("fabs({value})")),
                (crate::UnaryOp::Neg, DType::I32) => {
                    Ok(format!("as_int((uint)0u - as_uint({value}))"))
                }
                (crate::UnaryOp::Neg, DType::I64) => {
                    Ok(format!("as_long((ulong)0ul - as_ulong({value}))"))
                }
                _ => Err(OpenClError::Unsupported(format!(
                    "unary {op:?} for {dtype:?}"
                ))),
            }
        }
        UOpKind::GraphBinary(op) => {
            let lhs = child(0, source_map, lines)?;
            let rhs = child(1, source_map, lines)?;
            emit_binary(*op, dtype, &lhs, &rhs)
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
            Ok(format!("((uchar)(({lhs}) {operator} ({rhs})))"))
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            let condition = child(0, source_map, lines)?;
            let yes = child(1, source_map, lines)?;
            let no = child(2, source_map, lines)?;
            Ok(format!("(({condition}) ? ({yes}) : ({no}))"))
        }
        other => Err(OpenClError::Unsupported(format!("{other:?}"))),
    }
}

fn emit_binary(
    op: crate::BinaryOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, OpenClError> {
    use crate::BinaryOp::{Add, Div, Mul, Sub};
    match (op, dtype) {
        (Add | Sub | Mul | Div, DType::F32 | DType::F64) => {
            let operator = match op {
                Add => "+",
                Sub => "-",
                Mul => "*",
                Div => "/",
                _ => unreachable!(),
            };
            Ok(format!("(({lhs}) {operator} ({rhs}))"))
        }
        (Add, DType::Bool) => Ok(format!("((uchar)(({lhs}) || ({rhs})))")),
        (Sub, DType::Bool) => Ok(format!("((uchar)(({lhs}) != ({rhs})))")),
        (Mul, DType::Bool) => Ok(format!("((uchar)(({lhs}) && ({rhs})))")),
        (Add | Sub | Mul, DType::U32 | DType::U64) => {
            let operator = match op {
                Add => "+",
                Sub => "-",
                Mul => "*",
                _ => unreachable!(),
            };
            Ok(format!("(({lhs}) {operator} ({rhs}))"))
        }
        (Add | Sub | Mul, DType::I32) => {
            let operator = match op {
                Add => "+",
                Sub => "-",
                Mul => "*",
                _ => unreachable!(),
            };
            Ok(format!("as_int(as_uint({lhs}) {operator} as_uint({rhs}))"))
        }
        (Add | Sub | Mul, DType::I64) => {
            let operator = match op {
                Add => "+",
                Sub => "-",
                Mul => "*",
                _ => unreachable!(),
            };
            Ok(format!(
                "as_long(as_ulong({lhs}) {operator} as_ulong({rhs}))"
            ))
        }
        _ => Err(OpenClError::Unsupported(format!(
            "binary {op:?} for {dtype:?}; guarded integer div/mod/shift have no status ABI"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_reduction(
    reduction: &OpenClReduction,
    finalize: &UOp,
    dtype: DType,
    output_position: usize,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    capabilities: OpenClCapabilities,
) -> Result<(), OpenClError> {
    let value = finalize
        .sources()
        .first()
        .and_then(|update| update.sources().get(1))
        .ok_or_else(|| OpenClError::Unsupported("malformed reduction value".into()))?;
    if reduction.reduction_len == 0 {
        let final_value = reduction_empty_value(reduction.kind, dtype)?;
        lines.push(format!("  b{output_position}[gid] = {final_value};"));
        return Ok(());
    }
    match reduction.kind {
        crate::ReduceKind::Sum | crate::ReduceKind::Mean => {
            lines.push("  double acc = 0.0;".into())
        }
        crate::ReduceKind::Product => lines.push(match dtype {
            DType::Bool => "  uchar acc = (uchar)1u;".into(),
            DType::I32 | DType::U32 => "  uint acc = (uint)1u;".into(),
            DType::I64 | DType::U64 => "  ulong acc = (ulong)1ul;".into(),
            DType::F32 | DType::F64 => "  double acc = 1.0;".into(),
            _ => unreachable!("validated OpenCL storage"),
        }),
        crate::ReduceKind::Min | crate::ReduceKind::Max => match dtype {
            DType::F32 | DType::F64 => lines.push(format!(
                "  {} acc = ({}){}INFINITY;",
                cl_type(dtype),
                cl_type(dtype),
                if reduction.kind == crate::ReduceKind::Max {
                    "-"
                } else {
                    "+"
                }
            )),
            _ => {
                lines.push(format!("  {} acc = ({})0;", cl_type(dtype), cl_type(dtype)));
                lines.push("  uchar initialized = (uchar)0u;".into());
                if matches!(dtype, DType::I64 | DType::U64) {
                    lines.push("  double acc_key = 0.0;".into());
                }
            }
        },
    }
    lines.push(format!(
        "  for (ulong r = 0ul; r < {}ul; ++r) {{",
        reduction.reduction_len
    ));
    lines.push(format!(
        "    const ulong src_gid = {};",
        reduction.input_offset_expression()?
    ));
    let expression = emit_expr(value, ids, source_map, lines, "src_gid", capabilities)?;
    match reduction.kind {
        crate::ReduceKind::Sum | crate::ReduceKind::Mean => {
            lines.push(format!("    acc += (double)({expression});"));
        }
        crate::ReduceKind::Product => lines.push(match dtype {
            DType::Bool => format!("    acc = (uchar)(acc && ({expression}));"),
            DType::I32 => format!("    acc *= as_uint({expression});"),
            DType::U32 => format!("    acc *= ({expression});"),
            DType::I64 => format!("    acc *= as_ulong({expression});"),
            DType::U64 => format!("    acc *= ({expression});"),
            DType::F32 | DType::F64 => format!("    acc *= (double)({expression});"),
            _ => unreachable!("validated OpenCL storage"),
        }),
        crate::ReduceKind::Min | crate::ReduceKind::Max => {
            let comparison = if reduction.kind == crate::ReduceKind::Max {
                ">"
            } else {
                "<"
            };
            match dtype {
                DType::F32 | DType::F64 => {
                    lines.push(format!("    const {} next = {expression};", cl_type(dtype)));
                    lines.push(format!(
                        "    if (!isnan(next) && next {comparison} acc) acc = next;"
                    ));
                }
                DType::I64 | DType::U64 => {
                    lines.push(format!("    const {} next = {expression};", cl_type(dtype)));
                    lines.push("    const double next_key = convert_double_rte(next);".into());
                    lines.push(format!(
                        "    if (!initialized || next_key {comparison} acc_key) {{ acc = next; acc_key = next_key; initialized = (uchar)1u; }}"
                    ));
                }
                _ => {
                    lines.push(format!("    const {} next = {expression};", cl_type(dtype)));
                    lines.push(format!(
                        "    if (!initialized || next {comparison} acc) {{ acc = next; initialized = (uchar)1u; }}"
                    ));
                }
            }
        }
    }
    lines.push("  }".into());
    if matches!(reduction.kind, crate::ReduceKind::Mean) {
        lines.push(format!("  acc /= (double){}ul;", reduction.reduction_len));
    }
    let final_value = match (reduction.kind, dtype) {
        (crate::ReduceKind::Product, DType::I32) => "as_int(acc)".into(),
        (crate::ReduceKind::Product, DType::I64) => "as_long(acc)".into(),
        (crate::ReduceKind::Product, DType::F32 | DType::F64)
        | (crate::ReduceKind::Sum | crate::ReduceKind::Mean, _) => {
            format!("({})acc", cl_type(dtype))
        }
        _ => "acc".into(),
    };
    lines.push(format!("  b{output_position}[gid] = {final_value};"));
    Ok(())
}

fn reduction_empty_value(
    kind: crate::ReduceKind,
    dtype: DType,
) -> Result<&'static str, OpenClError> {
    use crate::ReduceKind::{Mean, Product, Sum};
    match (kind, dtype) {
        (Sum, DType::F32) => Ok("as_float((uint)0u)"),
        (Sum, DType::F64) => Ok("as_double((ulong)0ul)"),
        (Mean, DType::F32) => Ok("as_float((uint)0x7fc00000u)"),
        (Mean, DType::F64) => Ok("as_double((ulong)0x7ff8000000000000ul)"),
        (Product, DType::Bool) => Ok("((uchar)1u)"),
        (Product, DType::I32) => Ok("as_int((uint)1u)"),
        (Product, DType::U32) => Ok("((uint)1u)"),
        (Product, DType::I64) => Ok("as_long((ulong)1ul)"),
        (Product, DType::U64) => Ok("((ulong)1ul)"),
        (Product, DType::F32) => Ok("as_float((uint)0x3f800000u)"),
        (Product, DType::F64) => Ok("as_double((ulong)0x3ff0000000000000ul)"),
        _ => Err(OpenClError::Unsupported(format!(
            "empty {kind:?} for {dtype:?} has no OpenCL identity"
        ))),
    }
}

fn broadcast_offset(input: &Shape, output: &Shape, linear: &str) -> Result<String, OpenClError> {
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
