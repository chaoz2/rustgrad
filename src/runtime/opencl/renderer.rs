//! Pure deterministic OpenCL C lowering for a deliberately small UOp subset.
use super::OpenClError;
use crate::{DType, ScheduleInputBinding, Shape, UArg, UOp, UOpKind};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

pub const OPENCL_RENDERER_VERSION: &str = "rustgrad-opencl-elementwise-v1";
pub const OPENCL_ABI_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenClBufferAbi {
    pub id: u64,
    pub dtype: DType,
    pub source_shape: Shape,
    pub elements: usize,
    pub mutable: bool,
}

#[derive(Clone, Debug)]
pub struct RenderedOpenCl {
    pub source: String,
    pub source_map: BTreeMap<usize, usize>,
    pub buffers: Vec<OpenClBufferAbi>,
    pub extent: usize,
    pub entry: String,
    pub cache_key: String,
    pub(crate) semantic_program: Arc<UOp>,
}

impl RenderedOpenCl {
    /// Validates the schedule-owned first-use order against the pointer ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), OpenClError> {
        let inputs = self.buffers.iter().filter(|buffer| !buffer.mutable);
        if bindings.len() != inputs.clone().count() {
            return Err(OpenClError::InvalidBinding(
                "schedule/OpenCL input count mismatch".into(),
            ));
        }
        for (index, (binding, expected)) in bindings.iter().zip(inputs).enumerate() {
            if binding.abi_index != index
                || binding.desc.id != expected.id
                || binding.desc.dtype != expected.dtype
                || binding.desc.shape != expected.source_shape
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
}

impl Default for OpenClRenderer {
    fn default() -> Self {
        Self { local_size: 64 }
    }
}

impl OpenClRenderer {
    pub fn new(local_size: usize) -> Result<Self, OpenClError> {
        if local_size == 0 {
            return Err(OpenClError::InvalidArgument("zero local size"));
        }
        Ok(Self { local_size })
    }

    pub fn render(&self, root: &UOp) -> Result<RenderedOpenCl, OpenClError> {
        let nodes = root
            .topological()
            .map_err(|error| OpenClError::Unsupported(error.to_string()))?;
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
            return Err(OpenClError::Unsupported(
                "reductions, effects, and barriers are outside the OpenCL phase-A subset".into(),
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
        supported_storage(output_dtype)?;

        let mut inventory = BTreeMap::<u64, OpenClBufferAbi>::new();
        for node in &nodes {
            let UArg::BufferIndex {
                buffer,
                elements,
                input_shape,
                ..
            } = node.arg()
            else {
                if matches!(node.arg(), UArg::ViewBufferIndex { .. }) {
                    return Err(OpenClError::Unsupported(
                        "view addressing is outside the OpenCL phase-A subset".into(),
                    ));
                }
                continue;
            };
            let dtype = node
                .ty()
                .ok_or_else(|| OpenClError::Unsupported("untyped buffer index".into()))?
                .scalar;
            supported_storage(dtype)?;
            let abi = OpenClBufferAbi {
                id: *buffer,
                dtype,
                source_shape: input_shape.clone(),
                elements: *elements,
                mutable: *buffer == *output_id,
            };
            if let Some(old) = inventory.insert(*buffer, abi.clone())
                && old != abi
            {
                return Err(OpenClError::InvalidBinding(format!(
                    "buffer {buffer} has conflicting ABI metadata"
                )));
            }
        }

        let mut buffers = Vec::new();
        let mut seen = BTreeSet::new();
        for node in &nodes {
            if !matches!(node.kind(), UOpKind::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::InvalidBinding("load lacks index".into()))?;
            let UArg::BufferIndex { buffer, .. } = index.arg() else {
                return Err(OpenClError::Unsupported(
                    "load requires contiguous/broadcast BufferIndex".into(),
                ));
            };
            if seen.insert(*buffer) {
                buffers.push(
                    inventory
                        .get(buffer)
                        .ok_or_else(|| OpenClError::InvalidBinding("load ABI missing".into()))?
                        .clone(),
                );
            }
        }
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
        let mut lines = vec![
            format!("// {OPENCL_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
            format!("__kernel void {entry}("),
        ];
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
        let value = emit_expr(
            store
                .sources()
                .get(1)
                .ok_or_else(|| OpenClError::Unsupported("store has no value".into()))?,
            &ids,
            &mut source_map,
            &mut lines,
        )?;
        lines.push(format!("  b{output_position}[gid] = {value};"));
        lines.push("}".into());
        let source = lines.join("\n") + "\n";
        let cache_key = stable_key(&(
            OPENCL_RENDERER_VERSION,
            OPENCL_ABI_VERSION,
            self.local_size,
            &source,
            &buffers,
        ));
        Ok(RenderedOpenCl {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            semantic_program: Arc::new(root.clone()),
        })
    }
}

fn supported_storage(dtype: DType) -> Result<(), OpenClError> {
    match dtype {
        DType::F32 | DType::Bool => Ok(()),
        _ => Err(OpenClError::Unsupported(format!(
            "dtype {dtype:?} is outside the OpenCL phase-A subset"
        ))),
    }
}

fn cl_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::Bool => "uchar",
        _ => unreachable!("validated by supported_storage"),
    }
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
) -> Result<String, OpenClError> {
    let map_id = source_map.len();
    source_map.insert(map_id, lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| OpenClError::Unsupported(format!("untyped {:?}", node.kind())))?
        .scalar;
    supported_storage(dtype)?;
    let child = |index: usize, source_map: &mut BTreeMap<usize, usize>, lines: &mut Vec<String>| {
        node.sources()
            .get(index)
            .ok_or_else(|| OpenClError::Unsupported("missing expression operand".into()))
            .and_then(|source| emit_expr(source, ids, source_map, lines))
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
            let UArg::BufferIndex {
                buffer,
                input_shape,
                output_shape,
                ..
            } = index.arg()
            else {
                return Err(OpenClError::Unsupported(
                    "load requires contiguous/broadcast BufferIndex".into(),
                ));
            };
            let position = ids
                .get(buffer)
                .ok_or_else(|| OpenClError::InvalidBinding("load buffer absent from ABI".into()))?;
            let offset = broadcast_offset(input_shape, output_shape)?;
            Ok(format!("b{position}[{offset}]"))
        }
        UOpKind::Cast => {
            let value = child(0, source_map, lines)?;
            match (node.sources()[0].ty().map(|ty| ty.scalar), dtype) {
                (Some(DType::Bool), DType::F32) => Ok(format!("((float)({value}))")),
                (Some(DType::F32), DType::Bool) => Ok(format!("((uchar)(({value}) != 0.0f))")),
                (Some(source), target) if source == target => Ok(value),
                _ => Err(OpenClError::Unsupported(
                    "cast outside F32/bool subset".into(),
                )),
            }
        }
        UOpKind::GraphUnary(op) => {
            let value = child(0, source_map, lines)?;
            match (op, dtype) {
                (crate::UnaryOp::Neg, DType::F32) => Ok(format!("(-({value}))")),
                (crate::UnaryOp::Abs, DType::F32) => Ok(format!("fabs({value})")),
                _ => Err(OpenClError::Unsupported(format!(
                    "unary {op:?} for {dtype:?}"
                ))),
            }
        }
        UOpKind::GraphBinary(op) => {
            let lhs = child(0, source_map, lines)?;
            let rhs = child(1, source_map, lines)?;
            let operator = match (op, dtype) {
                (crate::BinaryOp::Add, DType::F32) => "+",
                (crate::BinaryOp::Sub, DType::F32) => "-",
                (crate::BinaryOp::Mul, DType::F32) => "*",
                (crate::BinaryOp::Div, DType::F32) => "/",
                _ => {
                    return Err(OpenClError::Unsupported(format!(
                        "binary {op:?} for {dtype:?}"
                    )));
                }
            };
            Ok(format!("(({lhs}) {operator} ({rhs}))"))
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

fn broadcast_offset(input: &Shape, output: &Shape) -> Result<String, OpenClError> {
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
                "((gid / {}ul) % {}ul) * {}ul",
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
