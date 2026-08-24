//! Deterministic phase-one PTX rendering and Driver launch glue.
//!
//! The renderer intentionally accepts only the fused elementwise UOp subset
//! that has a clear PTX contract. The CPU UOp interpreter remains the semantic
//! oracle; reductions, narrow floats, guarded integer division/shifts and
//! device-status reporting are rejected instead of silently changing meaning.

use crate::cuda_profile::{Metadata, OperationKind, ProfilingSession, TimedSample, TimingError};
use crate::{
    BufferView, CudaError, DType, Function, LaunchConfig, Shape, Stream, UArg, UOp, UOpKind,
};
use std::{
    collections::{BTreeMap, HashMap},
    ffi::{CString, c_void},
    fmt,
    rc::Rc,
    sync::{Arc, Condvar, Mutex},
};

pub const PTX_RENDERER_VERSION: &str = "rustgrad-ptx-elementwise-v2";
pub const PTX_ABI_VERSION: u32 = 1;
pub const COLLECTIVE_ADD_ABI_VERSION: u32 = 1;
#[allow(dead_code)]
const COLLECTIVE_ADD_RENDERER_VERSION: &str = "rustgrad-ptx-collective-add-v1";

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PtxBufferAbi {
    pub id: u64,
    pub dtype: DType,
    /// Physical storage shape behind this ABI pointer. For a view this is the
    /// original source allocation, never its logical view shape.
    pub source_shape: Shape,
    pub elements: usize,
    pub mutable: bool,
}
#[derive(Clone, Debug)]
pub struct RenderedPtx {
    pub source: String,
    pub source_map: BTreeMap<usize, usize>,
    pub buffers: Vec<PtxBufferAbi>,
    pub extent: usize,
    pub cache_key: String,
    pub entry: String,
    pub semantic_program: Option<Arc<UOp>>,
}
impl RenderedPtx {
    /// Validates the schedule-owned order against PTX parameter order. The
    /// launch ABI itself remains an ordered slice of `PtxBinding`.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[crate::ScheduleInputBinding],
    ) -> Result<(), PtxError> {
        for (index, binding) in bindings.iter().enumerate() {
            if binding.abi_index != index {
                return Err(PtxError::InvalidBinding(
                    "non-contiguous schedule ABI index".into(),
                ));
            }
            let want = self.buffers.get(index).ok_or_else(|| {
                PtxError::InvalidBinding("schedule binding exceeds PTX ABI".into())
            })?;
            if want.id != binding.desc.id
                || want.dtype != binding.desc.dtype
                || want.mutable
                || want.elements.checked_mul(want.dtype.itemsize()) != Some(binding.desc.bytes)
            {
                return Err(PtxError::InvalidBinding(format!(
                    "schedule binding {index} mismatches PTX ABI"
                )));
            }
        }
        Ok(())
    }
}
/// Immutable test-dispatch metadata for one renderer-validated generic PTX kernel.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct GenericKernelSemantics {
    pub key: String,
    pub buffers: Vec<PtxBufferAbi>,
    pub extent: usize,
    pub program: Arc<UOp>,
}
impl GenericKernelSemantics {
    fn from_rendered(rendered: &RenderedPtx) -> Result<Self, PtxError> {
        if rendered.buffers.is_empty() && rendered.extent != 0 {
            return Err(PtxError::InvalidBinding(
                "generic semantics lacks ABI buffers".into(),
            ));
        }
        let program = rendered
            .semantic_program
            .clone()
            .ok_or_else(|| PtxError::Unsupported("generic semantic program is absent".into()))?;
        Ok(Self {
            key: rendered.cache_key.clone(),
            buffers: rendered.buffers.clone(),
            extent: rendered.extent,
            program,
        })
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PtxError {
    Unsupported(String),
    InvalidBinding(String),
    Cuda(CudaError),
    Overflow,
}
impl fmt::Display for PtxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "unsupported PTX UOp: {s}"),
            Self::InvalidBinding(s) => write!(f, "invalid PTX binding: {s}"),
            Self::Cuda(e) => e.fmt(f),
            Self::Overflow => write!(f, "PTX launch geometry overflow"),
        }
    }
}
impl std::error::Error for PtxError {}
impl From<CudaError> for PtxError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

/// Renders for a concrete SM target. The target is capped to an ISA we emit.
#[derive(Clone, Copy, Debug)]
pub struct PtxRenderer {
    pub sm: u32,
    pub block_size: u32,
}
impl PtxRenderer {
    pub fn new(sm: u32) -> Result<Self, PtxError> {
        if !(20..=90).contains(&sm) {
            return Err(PtxError::Unsupported(format!(
                "unsupported compute capability sm_{sm}"
            )));
        }
        Ok(Self {
            sm,
            block_size: 256,
        })
    }
    pub fn render(&self, kernel: &UOp) -> Result<RenderedPtx, PtxError> {
        render(self, kernel)
    }
}

fn render(renderer: &PtxRenderer, root: &UOp) -> Result<RenderedPtx, PtxError> {
    let nodes = root
        .topological()
        .map_err(|e| PtxError::Unsupported(e.to_string()))?;
    let store = root
        .sources()
        .iter()
        .find(|n| matches!(n.kind(), UOpKind::Store))
        .ok_or_else(|| PtxError::Unsupported("Sink without Store".into()))?;
    let out_index = store
        .sources()
        .first()
        .ok_or_else(|| PtxError::Unsupported("Store without index".into()))?;
    let UArg::BufferIndex {
        buffer: out_id,
        elements: extent,
        output_shape,
        ..
    } = out_index.arg()
    else {
        return Err(PtxError::Unsupported("Store needs BufferIndex".into()));
    };
    let mut abi = BTreeMap::new();
    for node in &nodes {
        if let Some((buffer, elements, source_shape)) = match node.arg() {
            UArg::BufferIndex {
                buffer,
                elements,
                input_shape,
                ..
            } => Some((buffer, *elements, input_shape.clone())),
            UArg::ViewBufferIndex { buffer, view, .. } => Some((
                buffer,
                view.source_shape.numel().map_err(|_| PtxError::Overflow)?,
                view.source_shape.clone(),
            )),
            _ => None,
        } {
            let dtype = node
                .ty()
                .ok_or_else(|| PtxError::Unsupported("untyped index".into()))?
                .scalar;
            reject_dtype(dtype)?;
            abi.entry(*buffer).or_insert(PtxBufferAbi {
                id: *buffer,
                dtype,
                source_shape,
                elements,
                mutable: false,
            });
        }
    }
    abi.get_mut(out_id)
        .ok_or_else(|| PtxError::Unsupported("output missing ABI".into()))?
        .mutable = true;
    // PTX parameter positions follow lowered Load first-use order, never a
    // sorted buffer-ID inventory. The output pointer follows the inputs.
    let mut buffers = Vec::new();
    let mut seen = BTreeMap::new();
    for node in &nodes {
        if !matches!(node.kind(), UOpKind::Load) {
            continue;
        }
        let Some(index) = node.sources().first() else {
            return Err(PtxError::InvalidBinding("load without index".into()));
        };
        let buffer = match index.arg() {
            UArg::BufferIndex { buffer, .. } | UArg::ViewBufferIndex { buffer, .. } => *buffer,
            _ => return Err(PtxError::InvalidBinding("load index lacks buffer".into())),
        };
        if seen.insert(buffer, ()).is_none() {
            buffers.push(
                abi.get(&buffer)
                    .ok_or_else(|| PtxError::InvalidBinding("load ABI missing".into()))?
                    .clone(),
            );
        }
    }
    if seen.insert(*out_id, ()).is_none() {
        buffers.push(
            abi.get(out_id)
                .ok_or_else(|| PtxError::InvalidBinding("output ABI missing".into()))?
                .clone(),
        );
    }
    let entry = format!("rg_e{}_b{}", extent, buffers.len());
    let mut lines = vec![
        format!("// {PTX_RENDERER_VERSION} ABI {PTX_ABI_VERSION}"),
        ".version 7.0".into(),
        format!(".target sm_{}", renderer.sm),
        ".address_size 64".into(),
        "".into(),
        format!(".visible .entry {entry}("),
    ];
    for (n, buffer) in buffers.iter().enumerate() {
        lines.push(format!("  .param .u64 p{n},"));
        let _ = buffer;
    }
    lines.push("  .param .u64 extent".into());
    lines.push(")".into());
    lines.push("{".into());
    lines.extend([
        "  .reg .pred %p<8>;".into(),
        "  .reg .b32 %r<32>;".into(),
        "  .reg .b64 %rd<32>;".into(),
        "  .reg .f32 %f<32>;".into(),
        "  .reg .f64 %fd<16>;".into(),
    ]);
    for n in 0..buffers.len() {
        lines.push(format!("  ld.param.u64 %rd{}0, [p{n}];", n + 1));
    }
    lines.extend([
        "  ld.param.u64 %rd0, [extent];".into(),
        "  mov.u32 %r0, %ctaid.x;".into(),
        "  mov.u32 %r1, %ntid.x;".into(),
        "  mov.u32 %r2, %tid.x;".into(),
        "  mad.lo.u32 %r3, %r0, %r1, %r2;".into(),
        "  cvt.u64.u32 %rd30, %r3;".into(),
        "  setp.ge.u64 %p0, %rd30, %rd0;".into(),
        "  @%p0 bra DONE;".into(),
    ]);
    let mut ids = BTreeMap::new();
    for (i, b) in buffers.iter().enumerate() {
        ids.insert(b.id, i);
    }
    let mut map = BTreeMap::new();
    let value = emit(
        store
            .sources()
            .get(1)
            .ok_or_else(|| PtxError::Unsupported("Store without value".into()))?,
        &ids,
        &mut lines,
        &mut map,
    )?;
    let out = buffers.iter().find(|b| b.id == *out_id).unwrap();
    let oi = ids[out_id] + 1;
    lines.push(format!(
        "  mul.wide.u32 %rd31, %r3, {};",
        out.dtype.itemsize()
    ));
    lines.push(format!("  add.u64 %rd31, %rd{oi}0, %rd31;"));
    lines.push(format!(
        "  st.global.{} [%rd31], {value};",
        ptx_type(out.dtype)
    ));
    lines.extend(["DONE:".into(), "  ret;".into(), "}".into()]);
    let source = lines.join("\n") + "\n";
    let key = stable_key(&(PTX_RENDERER_VERSION, renderer.sm, &source, &buffers));
    let _ = output_shape;
    Ok(RenderedPtx {
        source,
        source_map: map,
        buffers,
        extent: *extent,
        cache_key: key,
        entry,
        semantic_program: Some(Arc::new(root.clone())),
    })
}
fn reject_dtype(dtype: DType) -> Result<(), PtxError> {
    match dtype {
        DType::F16 | DType::BF16 => Err(PtxError::Unsupported(format!(
            "{dtype:?} requires capability-specific conversion support"
        ))),
        DType::Bool
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F32
        | DType::F64 => Ok(()),
        _ => Err(PtxError::Unsupported(format!("dtype {dtype:?}"))),
    }
}
fn ptx_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "u8",
        DType::I32 => "s32",
        DType::U32 => "u32",
        DType::I64 => "s64",
        DType::U64 => "u64",
        DType::F32 => "f32",
        DType::F64 => "f64",
        _ => unreachable!(),
    }
}
fn emit(
    n: &UOp,
    ids: &BTreeMap<u64, usize>,
    lines: &mut Vec<String>,
    map: &mut BTreeMap<usize, usize>,
) -> Result<String, PtxError> {
    let id = map.len();
    map.insert(id, lines.len() + 1);
    let ty = n
        .ty()
        .ok_or_else(|| PtxError::Unsupported(format!("untyped {:?}", n.kind())))?
        .scalar;
    reject_dtype(ty)?;
    let mut child = |i| emit(&n.sources()[i], ids, lines, map);
    let dst = match ty {
        DType::F32 => format!("%f{id}"),
        DType::F64 => format!("%fd{id}"),
        DType::Bool => format!("%r{id}"),
        _ => format!("%r{id}"),
    };
    match n.kind() {
        UOpKind::Const => match n.arg() {
            UArg::Int(v) => lines.push(format!("  mov.{} {dst}, {v};", ptx_type(ty))),
            UArg::Scalar { dtype, bits } if *dtype == ty => {
                let width = dtype.bits();
                let digits = width as usize / 4;
                lines.push(format!("  mov.b{width} {dst}, 0x{bits:0digits$x};"));
            }
            UArg::Scalar { .. } => {
                return Err(PtxError::Unsupported("scalar literal/type mismatch".into()));
            }
            _ => return Err(PtxError::Unsupported("invalid constant".into())),
        },
        UOpKind::Load => {
            let ix = n
                .sources()
                .first()
                .ok_or_else(|| PtxError::Unsupported("Load without index".into()))?;
            let (buffer, input_shape, output_shape, view) = match ix.arg() {
                UArg::BufferIndex {
                    buffer,
                    input_shape,
                    output_shape,
                    ..
                } => (buffer, input_shape, output_shape, None),
                UArg::ViewBufferIndex {
                    buffer,
                    input_shape,
                    output_shape,
                    view,
                    ..
                } => (buffer, input_shape, output_shape, Some(view)),
                _ => return Err(PtxError::Unsupported("Load index".into())),
            };
            let b = ids[buffer] + 1;
            let off = broadcast_offset(input_shape.dims(), output_shape.dims())?;
            lines.extend(off);
            if let Some(view) = view {
                lines.extend(view_offset(view)?);
            }
            lines.push(format!("  add.u64 %rd29, %rd{b}0, %rd28;"));
            lines.push(format!("  ld.global.{} {dst}, [%rd29];", ptx_type(ty)));
        }
        UOpKind::Cast => {
            let a = child(0)?;
            lines.push(format!(
                "  cvt.{}.{} {dst}, {a};",
                ptx_type(ty),
                ptx_type(n.sources()[0].ty().unwrap().scalar)
            ));
        }
        UOpKind::GraphUnary(op) => {
            // Keep this deliberately narrower than the CPU interpreter.  PTX
            // has exact scalar `neg` and `abs` instructions, including the
            // wrapping signed-min integer result, but the renderer has no
            // versioned libdevice contract for transcendental operations.
            let a = child(0)?;
            let mnemonic = match (op, ty) {
                (crate::UnaryOp::Neg, DType::I32 | DType::I64 | DType::F32 | DType::F64) => "neg",
                (crate::UnaryOp::Abs, DType::I32 | DType::I64 | DType::F32 | DType::F64) => "abs",
                _ => {
                    return Err(PtxError::Unsupported(format!(
                        "unary {op:?} for {ty:?} is outside the exact PTX subset"
                    )));
                }
            };
            lines.push(format!("  {mnemonic}.{} {dst}, {a};", ptx_type(ty)));
        }
        UOpKind::GraphBinary(op) => {
            let (a, b) = (child(0)?, child(1)?);
            let mnemonic = match op {
                crate::BinaryOp::Add => "add",
                crate::BinaryOp::Sub => "sub",
                crate::BinaryOp::Mul => "mul",
                crate::BinaryOp::Div if ty.is_float() => "div",
                crate::BinaryOp::Maximum => "max",
                crate::BinaryOp::Minimum => "min",
                crate::BinaryOp::Div
                | crate::BinaryOp::TruncDiv
                | crate::BinaryOp::Mod
                | crate::BinaryOp::FMod
                | crate::BinaryOp::Shl
                | crate::BinaryOp::Shr => {
                    return Err(PtxError::Unsupported(format!(
                        "guarded integer {op:?} needs status ABI"
                    )));
                }
                _ => return Err(PtxError::Unsupported(format!("binary {op:?}"))),
            };
            lines.push(format!("  {mnemonic}.{} {dst}, {a}, {b};", ptx_type(ty)));
        }
        UOpKind::GraphCompare(op) => {
            let (a, b) = (child(0)?, child(1)?);
            let pred = match op {
                crate::CompareOp::Eq => "eq",
                crate::CompareOp::Ne => "ne",
                crate::CompareOp::Lt => "lt",
                crate::CompareOp::Le => "le",
                crate::CompareOp::Gt => "gt",
                crate::CompareOp::Ge => "ge",
            };
            lines.push(format!(
                "  setp.{pred}.{} %p1, {a}, {b};",
                ptx_type(n.sources()[0].ty().unwrap().scalar)
            ));
            lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            let (p, a, b) = (child(0)?, child(1)?, child(2)?);
            lines.push(format!("  setp.ne.u32 %p2, {p}, 0;"));
            lines.push(format!("  selp.{} {dst}, {a}, {b}, %p2;", ptx_type(ty)));
        }
        _ => return Err(PtxError::Unsupported(format!("{:?}", n.kind()))),
    };
    Ok(dst)
}
fn broadcast_offset(input: &[usize], output: &[usize]) -> Result<Vec<String>, PtxError> {
    if input.len() > output.len() {
        return Err(PtxError::Unsupported("broadcast rank".into()));
    };
    let mut lines = vec![format!("  mul.wide.u32 %rd28, %r3, {};", 1)];
    if input == output {
        return Ok(lines);
    };
    let pad = output.len() - input.len();
    lines.clear();
    lines.push("  mov.u64 %rd28, 0;".into());
    for (i, &d) in input.iter().enumerate() {
        if d != 1 {
            let divisor = output[pad + i + 1..]
                .iter()
                .try_fold(1usize, |a, x| a.checked_mul(*x))
                .ok_or(PtxError::Overflow)?;
            let scale = input[i + 1..]
                .iter()
                .try_fold(1usize, |a, x| a.checked_mul(*x))
                .ok_or(PtxError::Overflow)?;
            lines.push(format!("  div.u32 %r20, %r3, {divisor};"));
            lines.push(format!("  rem.u32 %r20, %r20, {d};"));
            lines.push(format!("  mul.wide.u32 %rd27, %r20, {scale};"));
            lines.push("  add.u64 %rd28, %rd28, %rd27;".into());
        }
    }
    lines.push(format!("  mul.lo.u64 %rd28, %rd28, {};", 1));
    Ok(lines)
}
fn view_offset(view: &crate::uop::ViewMap) -> Result<Vec<String>, PtxError> {
    let mut lines = vec![
        "  mov.u64 %rd26, %rd28;".into(),
        format!("  mov.u64 %rd28, {};", view.offset),
    ];
    for (axis, (&dim, &stride)) in view
        .logical_shape
        .dims()
        .iter()
        .zip(&view.strides)
        .enumerate()
    {
        if dim == 0 {
            continue;
        }
        let divisor = view.logical_shape.dims()[axis + 1..]
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(PtxError::Overflow)?;
        lines.push(format!("  div.u64 %rd27, %rd26, {divisor};"));
        lines.push(format!("  rem.u64 %rd27, %rd27, {dim};"));
        lines.push(format!("  mad.lo.u64 %rd28, %rd27, {stride}, %rd28;"));
    }
    Ok(lines)
}
fn stable_key(value: &impl std::hash::Hash) -> String {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    format!("{:016x}", h.finish())
}

#[allow(dead_code)]
fn collective_add_dtype(dtype: DType) -> Result<(&'static str, &'static str), PtxError> {
    match dtype {
        DType::I8 => Ok(("s8", "s32")),
        DType::U8 => Ok(("u8", "u32")),
        DType::I32 => Ok(("s32", "s32")),
        DType::U32 => Ok(("u32", "u32")),
        DType::I64 => Ok(("s64", "s64")),
        DType::U64 => Ok(("u64", "u64")),
        DType::F32 => Ok(("f32", "f32")),
        DType::F64 => Ok(("f64", "f64")),
        DType::Bool | DType::F16 | DType::BF16 | DType::I16 | DType::U16 => Err(
            PtxError::Unsupported(format!("collective add does not yet support {dtype:?}")),
        ),
    }
}

/// Inspectable PTX for one-dimensional in-place local addition. Its pointer
/// ABI is `(destination, source, destination_offset, source_offset, count)`.
#[allow(dead_code)]
pub(crate) fn render_collective_add(
    renderer: &PtxRenderer,
    dtype: DType,
) -> Result<RenderedPtx, PtxError> {
    let (memory_type, alu_type) = collective_add_dtype(dtype)?;
    let entry = format!("rg_collective_add_{memory_type}");
    let value_registers = match dtype {
        DType::F32 => ("%f0", "%f1", "%f2"),
        DType::F64 => ("%fd0", "%fd1", "%fd2"),
        DType::I64 | DType::U64 => ("%rd10", "%rd11", "%rd12"),
        _ => ("%r4", "%r5", "%r6"),
    };
    let source = format!(
        "// {COLLECTIVE_ADD_RENDERER_VERSION} ABI {COLLECTIVE_ADD_ABI_VERSION}\n.version 7.0\n.target sm_{}\n.address_size 64\n\n.visible .entry {entry}(\n  .param .u64 destination,\n  .param .u64 source,\n  .param .u64 destination_offset,\n  .param .u64 source_offset,\n  .param .u64 count\n)\n{{\n  .reg .pred %p<2>;\n  .reg .b32 %r<16>;\n  .reg .b64 %rd<16>;\n  .reg .f32 %f<4>;\n  .reg .f64 %fd<4>;\n  ld.param.u64 %rd0, [destination];\n  ld.param.u64 %rd1, [source];\n  ld.param.u64 %rd2, [destination_offset];\n  ld.param.u64 %rd3, [source_offset];\n  ld.param.u64 %rd4, [count];\n  mov.u32 %r0, %ctaid.x;\n  mov.u32 %r1, %ntid.x;\n  mov.u32 %r2, %tid.x;\n  mad.lo.u32 %r3, %r0, %r1, %r2;\n  cvt.u64.u32 %rd5, %r3;\n  setp.ge.u64 %p0, %rd5, %rd4;\n  @%p0 bra DONE;\n  add.u64 %rd6, %rd2, %rd5;\n  add.u64 %rd7, %rd3, %rd5;\n  mul.lo.u64 %rd6, %rd6, {};\n  mul.lo.u64 %rd7, %rd7, {};\n  add.u64 %rd8, %rd0, %rd6;\n  add.u64 %rd9, %rd1, %rd7;\n  ld.global.{memory_type} {}, [%rd8];\n  ld.global.{memory_type} {}, [%rd9];\n  add.{alu_type} {}, {}, {};\n  st.global.{memory_type} [%rd8], {};\nDONE:\n  ret;\n}}\n",
        renderer.sm,
        dtype.itemsize(),
        dtype.itemsize(),
        value_registers.0,
        value_registers.1,
        value_registers.2,
        value_registers.0,
        value_registers.1,
        value_registers.2,
    );
    Ok(RenderedPtx {
        source_map: BTreeMap::from([(0, 29)]),
        cache_key: stable_key(&(
            COLLECTIVE_ADD_RENDERER_VERSION,
            COLLECTIVE_ADD_ABI_VERSION,
            renderer.sm,
            dtype,
            &source,
        )),
        source,
        buffers: vec![],
        extent: 0,
        entry,
        semantic_program: None,
    })
}

pub struct PtxBinding<'a> {
    pub buffer: BufferView<'a>,
    pub dtype: DType,
    pub mutable: bool,
}
pub struct PtxKernel {
    rendered: Rc<RenderedPtx>,
    module: Rc<crate::CudaModule>,
    function: Function,
    block_size: u32,
}
impl PtxKernel {
    pub fn load_metadata(&self) -> &crate::ModuleLoadMetadata {
        self.module.load_metadata()
    }
    pub fn load(
        context: &crate::Context,
        rendered: Rc<RenderedPtx>,
        block_size: u32,
    ) -> Result<Self, PtxError> {
        if block_size == 0 {
            return Err(PtxError::InvalidBinding("zero block size".into()));
        };
        let image = CString::new(rendered.source.clone())
            .map_err(|_| PtxError::Unsupported("PTX contains NUL".into()))?;
        let module = Rc::new(context.module_from_ptx(&image)?);
        let name = CString::new(rendered.entry.clone()).unwrap();
        let function = module.function(&name)?;
        Ok(Self {
            rendered,
            module,
            function,
            block_size,
        })
    }
    /// Primary-context counterpart to [`Self::load`].  The resulting module,
    /// function, and cache entry retain the primary owner through `CudaModule`.
    pub fn load_primary(
        context: &crate::PrimaryContext,
        rendered: Rc<RenderedPtx>,
        block_size: u32,
    ) -> Result<Self, PtxError> {
        if block_size == 0 {
            return Err(PtxError::InvalidBinding("zero block size".into()));
        }
        let image = CString::new(rendered.source.clone())
            .map_err(|_| PtxError::Unsupported("PTX contains NUL".into()))?;
        let module = Rc::new(context.module_from_ptx(&image)?);
        let name = CString::new(rendered.entry.clone()).unwrap();
        let function = module.function(&name)?;
        Ok(Self {
            rendered,
            module,
            function,
            block_size,
        })
    }
    pub fn launch(
        &self,
        stream: &Stream,
        bindings: &[PtxBinding<'_>],
        synchronize: bool,
    ) -> Result<(), PtxError> {
        if bindings.len() != self.rendered.buffers.len() {
            return Err(PtxError::InvalidBinding("wrong buffer count".into()));
        };
        if self.rendered.extent == 0 {
            return Ok(());
        };
        let mut words = Vec::with_capacity(bindings.len() + 1);
        for (want, got) in self.rendered.buffers.iter().zip(bindings) {
            if want.dtype != got.dtype || want.mutable != got.mutable {
                return Err(PtxError::InvalidBinding(format!(
                    "buffer {} ABI mismatch",
                    want.id
                )));
            };
            if got.buffer.device() != self.module_device() {
                return Err(PtxError::Cuda(CudaError::ContextMismatch));
            };
            let need = want
                .elements
                .checked_mul(want.dtype.itemsize())
                .ok_or(PtxError::Overflow)?;
            if got.buffer.len() < need {
                return Err(PtxError::InvalidBinding("buffer too small".into()));
            };
            words.push(got.buffer.device_ptr()?)
        }
        words.push(self.rendered.extent as u64);
        let mut args: Vec<*mut c_void> = words.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        let grid = self
            .rendered
            .extent
            .checked_add(self.block_size as usize - 1)
            .ok_or(PtxError::Overflow)?
            / self.block_size as usize;
        self.function.launch(
            LaunchConfig {
                grid: [u32::try_from(grid).map_err(|_| PtxError::Overflow)?, 1, 1],
                block: [self.block_size, 1, 1],
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )?;
        // The non-profiled API returns no completion token. For a pooled view,
        // wait before releasing its borrow so a physical block cannot reenter
        // the cache while this launch is still in flight.
        if synchronize || bindings.iter().any(|binding| binding.buffer.is_pooled()) {
            stream.synchronize()?
        };
        Ok(())
    }
    fn module_device(&self) -> crate::DeviceId {
        self.module.device()
    }
}
pub struct PtxCache {
    kernels: HashMap<String, Rc<PtxKernel>>,
}

/// A primary-context-only concurrent PTX cache.
///
/// The map lock protects only entry creation/removal. Per-key waiters use the
/// entry lock and condition variable; CUDA Driver calls always happen with no
/// cache lock held. `PrimaryPtxKernel` is Send + Sync because this cache admits
/// only `PrimaryContext` owners, whose push/pop currentness is per-thread.
/// Owned `Context` kernels intentionally remain in the non-concurrent
/// `PtxCache`: their thread affinity must not be erased by a sum type.
pub struct ConcurrentPtxCache {
    entries: Mutex<HashMap<(usize, String, u32), Arc<ConcurrentEntry>>>,
}
struct ConcurrentEntry {
    state: Mutex<EntryState>,
    ready: Condvar,
}
enum EntryState {
    Loading,
    Ready(Arc<PrimaryPtxKernel>),
    Failed(PtxError),
}
/// A primary-owned cached kernel. It never contains an owned `Context`.
pub struct PrimaryPtxKernel {
    rendered: Arc<RenderedPtx>,
    module: Arc<crate::CudaModule>,
    function: Function,
    block_size: u32,
    primary: crate::PrimaryContext,
}
/// In-flight primary PTX launch profiling. The sample borrows the launch
/// stream and bindings, retaining those resources and the kernel through an
/// explicit timing query, wait, collect, failure, or abandonment.
#[allow(dead_code)] // crate-private profiling API is consumed by submission integration.
pub(crate) struct ProfiledPrimaryPtxSample<'a> {
    timing: TimedSample<'a>,
    _kernel: &'a PrimaryPtxKernel,
    _buffers: Vec<BufferView<'a>>,
}
#[allow(dead_code)]
impl ProfiledPrimaryPtxSample<'_> {
    pub(crate) fn query(&mut self) -> Result<Option<u64>, PtxError> {
        self.timing.query().map_err(profile_error)
    }
    pub(crate) fn wait(&mut self) -> Result<u64, PtxError> {
        self.timing.wait().map_err(profile_error)
    }
    pub(crate) fn collect(self) -> Result<u64, PtxError> {
        self.timing.collect().map_err(profile_error)
    }
}
#[allow(dead_code)]
fn profile_error(error: TimingError) -> PtxError {
    match error {
        TimingError::Cuda(error) => PtxError::Cuda(error),
        other => PtxError::InvalidBinding(other.to_string()),
    }
}
// The constructor is private and accepts only PrimaryContext. CUDA primary
// contexts are shareable; each operation enters via push/pop before callbacks.
unsafe impl Send for PrimaryPtxKernel {}
unsafe impl Sync for PrimaryPtxKernel {}
impl PrimaryPtxKernel {
    #[allow(clippy::arc_with_non_send_sync)] // `Self`'s primary-only invariant makes this Arc sound.
    fn load(
        context: &crate::PrimaryContext,
        rendered: Arc<RenderedPtx>,
        block_size: u32,
    ) -> Result<Self, PtxError> {
        if block_size == 0 {
            return Err(PtxError::InvalidBinding("zero block size".into()));
        }
        let image = CString::new(rendered.source.clone())
            .map_err(|_| PtxError::Unsupported("PTX contains NUL".into()))?;
        let module = Arc::new(context.module_from_ptx(&image)?);
        let name = CString::new(rendered.entry.clone()).unwrap();
        let function = module.function(&name)?;
        if rendered.semantic_program.is_some() {
            context.register_generic_kernel_semantics(
                function.identity(),
                &rendered.cache_key,
                std::sync::Arc::new(GenericKernelSemantics::from_rendered(&rendered)?),
            );
        }
        Ok(Self {
            rendered,
            module,
            function,
            block_size,
            primary: context.clone(),
        })
    }
    pub fn load_metadata(&self) -> &crate::ModuleLoadMetadata {
        self.module.load_metadata()
    }
    pub fn launch(
        &self,
        stream: &Stream,
        bindings: &[PtxBinding<'_>],
        synchronize: bool,
    ) -> Result<(), PtxError> {
        if !self.module.belongs_to_primary(&self.primary)
            || !stream.belongs_to_primary(&self.primary)
        {
            return Err(PtxError::Cuda(CudaError::ContextMismatch));
        }
        if bindings.len() != self.rendered.buffers.len() {
            return Err(PtxError::InvalidBinding("wrong buffer count".into()));
        }
        if self.rendered.extent == 0 {
            return Ok(());
        }
        let mut words = Vec::with_capacity(bindings.len() + 1);
        for (want, got) in self.rendered.buffers.iter().zip(bindings) {
            if want.dtype != got.dtype || want.mutable != got.mutable {
                return Err(PtxError::InvalidBinding(format!(
                    "buffer {} ABI mismatch",
                    want.id
                )));
            }
            if !got.buffer.belongs_to_primary(&self.primary)
                || got.buffer.device() != self.module.device()
            {
                return Err(PtxError::Cuda(CudaError::ContextMismatch));
            }
            let need = want
                .elements
                .checked_mul(want.dtype.itemsize())
                .ok_or(PtxError::Overflow)?;
            if got.buffer.len() < need {
                return Err(PtxError::InvalidBinding("buffer too small".into()));
            }
            words.push(got.buffer.device_ptr()?);
        }
        words.push(self.rendered.extent as u64);
        let mut args: Vec<*mut c_void> = words.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        let grid = self
            .rendered
            .extent
            .checked_add(self.block_size as usize - 1)
            .ok_or(PtxError::Overflow)?
            / self.block_size as usize;
        self.function.launch(
            LaunchConfig {
                grid: [u32::try_from(grid).map_err(|_| PtxError::Overflow)?, 1, 1],
                block: [self.block_size, 1, 1],
                shared_bytes: 0,
            },
            stream,
            &mut args,
        )?;
        self.attach_primary_completion(stream, bindings)?;
        if synchronize {
            stream.synchronize()?;
        }
        Ok(())
    }
    fn attach_primary_completion(
        &self,
        stream: &Stream,
        bindings: &[PtxBinding<'_>],
    ) -> Result<(), PtxError> {
        let mut leases = Vec::new();
        for binding in bindings {
            if let Some(lease) = binding.buffer.primary_lease()
                && !leases
                    .iter()
                    .any(|old: &&crate::PrimaryBufferLease| std::ptr::eq(*old, lease))
            {
                leases.push(lease);
            }
        }
        let Some(first) = leases.first() else {
            return Ok(());
        };
        let primary = first.primary()?;
        for lease in &leases {
            if lease.primary()?.identity() != primary.identity() {
                return Err(PtxError::Cuda(CudaError::ContextMismatch));
            }
        }
        let fence = Arc::new(primary.event_fence()?);
        if let Err(error) = fence.record(stream) {
            // The kernel was submitted but no completion event exists. A
            // successful stream sync proves safety; otherwise quarantine.
            if stream.synchronize().is_err() {
                for lease in leases {
                    lease.quarantine();
                }
            }
            return Err(PtxError::Cuda(error));
        }
        for lease in leases {
            lease.attach_fence(fence.clone())?;
        }
        Ok(())
    }
    /// Launches with the crate-private CUDA timing adapter. Disabled profiling
    /// delegates directly to [`Self::launch`], preserving its Driver sequence.
    #[allow(dead_code)]
    pub(crate) fn launch_profiled<'a>(
        &'a self,
        session: &ProfilingSession,
        semantic_name: impl Into<String>,
        primary: &crate::PrimaryContext,
        stream: &'a Stream,
        bindings: &'a [PtxBinding<'a>],
        synchronize: bool,
    ) -> Result<Option<ProfiledPrimaryPtxSample<'a>>, PtxError> {
        if !session.is_enabled() {
            self.launch(stream, bindings, synchronize)?;
            return Ok(None);
        }
        let (mut words, config) = self.prepare_profiled_launch(primary, stream, bindings)?;
        let mut args: Vec<*mut c_void> = words
            .iter_mut()
            .map(|word| (word as *mut u64).cast())
            .collect();
        let metadata = Metadata {
            kind: OperationKind::Kernel,
            name: semantic_name.into(),
            owner: primary.identity(),
            device: primary.device(),
            stream: stream.identity(),
            bytes: None,
            geometry: Some((config.grid, config.block)),
            source_key: Some(self.rendered.cache_key.clone()),
            peer: None,
        };
        let retained: std::sync::Arc<dyn Send + Sync> = std::sync::Arc::new(());
        let Some(mut timing) = TimedSample::begin(session, metadata, primary, stream, retained)
            .map_err(profile_error)?
        else {
            return Ok(None);
        };
        if let Err(error) = self.function.launch(config, stream, &mut args) {
            timing.fail_due_to(TimingError::Cuda(error.clone()));
            return Err(PtxError::Cuda(error));
        }
        if let Err(error) = timing.record_end(stream) {
            return Err(profile_error(error));
        }
        self.attach_primary_completion(stream, bindings)?;
        if synchronize {
            // Preserve the existing launch option after the end marker so the
            // timing interval remains exactly the submitted kernel work.
            if let Err(error) = stream.synchronize() {
                timing.fail_due_to(TimingError::Cuda(error.clone()));
                return Err(PtxError::Cuda(error));
            }
        }
        Ok(Some(ProfiledPrimaryPtxSample {
            timing,
            _kernel: self,
            _buffers: bindings.iter().map(|binding| binding.buffer).collect(),
        }))
    }
    #[allow(dead_code)]
    fn prepare_profiled_launch(
        &self,
        primary: &crate::PrimaryContext,
        stream: &Stream,
        bindings: &[PtxBinding<'_>],
    ) -> Result<(Vec<u64>, LaunchConfig), PtxError> {
        if !self.module.belongs_to_primary(primary) || !stream.belongs_to_primary(primary) {
            return Err(PtxError::Cuda(CudaError::ContextMismatch));
        }
        if bindings.len() != self.rendered.buffers.len() {
            return Err(PtxError::InvalidBinding("wrong buffer count".into()));
        }
        if self.rendered.extent == 0 {
            return Err(PtxError::InvalidBinding(
                "zero-extent profiled launch".into(),
            ));
        }
        let mut words = Vec::with_capacity(bindings.len() + 1);
        for (want, got) in self.rendered.buffers.iter().zip(bindings) {
            if want.dtype != got.dtype || want.mutable != got.mutable {
                return Err(PtxError::InvalidBinding(format!(
                    "buffer {} ABI mismatch",
                    want.id
                )));
            }
            if !got.buffer.belongs_to_primary(primary)
                || got.buffer.device() != self.module.device()
            {
                return Err(PtxError::Cuda(CudaError::ContextMismatch));
            }
            let need = want
                .elements
                .checked_mul(want.dtype.itemsize())
                .ok_or(PtxError::Overflow)?;
            if got.buffer.len() < need {
                return Err(PtxError::InvalidBinding("buffer too small".into()));
            }
            words.push(got.buffer.device_ptr()?);
        }
        words.push(self.rendered.extent as u64);
        let grid = self
            .rendered
            .extent
            .checked_add(self.block_size as usize - 1)
            .ok_or(PtxError::Overflow)?
            / self.block_size as usize;
        let config = LaunchConfig {
            grid: [u32::try_from(grid).map_err(|_| PtxError::Overflow)?, 1, 1],
            block: [self.block_size, 1, 1],
            shared_bytes: 0,
        };
        primary.validate_launch(config)?;
        Ok((words, config))
    }
}
impl Drop for PrimaryPtxKernel {
    fn drop(&mut self) {
        self.primary
            .unregister_generic_kernel_semantics(self.function.identity());
    }
}
impl Default for ConcurrentPtxCache {
    fn default() -> Self {
        Self::new()
    }
}
impl ConcurrentPtxCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
    pub fn get_or_load(
        &self,
        context: &crate::PrimaryContext,
        rendered: RenderedPtx,
        block_size: u32,
    ) -> Result<Arc<PrimaryPtxKernel>, PtxError> {
        let key = (context.identity(), rendered.cache_key.clone(), block_size);
        let (entry, leader) = {
            let mut entries = self.entries.lock().expect("PTX cache mutex poisoned");
            match entries.get(&key) {
                Some(entry) => (entry.clone(), false),
                None => {
                    let entry = Arc::new(ConcurrentEntry {
                        state: Mutex::new(EntryState::Loading),
                        ready: Condvar::new(),
                    });
                    entries.insert(key.clone(), entry.clone());
                    (entry, true)
                }
            }
        };
        if leader {
            let result =
                PrimaryPtxKernel::load(context, Arc::new(rendered), block_size).map(Arc::new);
            let mut state = entry.state.lock().expect("PTX entry mutex poisoned");
            match &result {
                Ok(kernel) => *state = EntryState::Ready(kernel.clone()),
                Err(error) => *state = EntryState::Failed(error.clone()),
            }
            entry.ready.notify_all();
            drop(state);
            if result.is_err() {
                let mut entries = self.entries.lock().expect("PTX cache mutex poisoned");
                if entries
                    .get(&key)
                    .is_some_and(|current| Arc::ptr_eq(current, &entry))
                {
                    entries.remove(&key);
                }
            }
            return result;
        }
        let mut state = entry.state.lock().expect("PTX entry mutex poisoned");
        loop {
            match &*state {
                EntryState::Loading => {
                    state = entry.ready.wait(state).expect("PTX entry mutex poisoned")
                }
                EntryState::Ready(kernel) => return Ok(kernel.clone()),
                EntryState::Failed(error) => return Err(error.clone()),
            }
        }
    }
    pub fn len(&self) -> usize {
        self.entries.lock().expect("PTX cache mutex poisoned").len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Crate-private primary-context local-add kernel used as the CUDA building
/// block for a future collective executor. It has no collective scheduling.
#[allow(dead_code)]
pub(crate) struct PrimaryCollectiveAddKernel {
    kernel: Arc<PrimaryPtxKernel>,
    rendered: RenderedPtx,
    dtype: DType,
}
#[allow(dead_code)]
pub(crate) struct PrimaryCollectiveAddCache {
    kernels: ConcurrentPtxCache,
}
impl Default for PrimaryCollectiveAddCache {
    fn default() -> Self {
        Self::new()
    }
}
#[allow(dead_code)]
impl PrimaryCollectiveAddCache {
    pub(crate) fn new() -> Self {
        Self {
            kernels: ConcurrentPtxCache::new(),
        }
    }
    pub(crate) fn get_or_load(
        &self,
        primary: &crate::PrimaryContext,
        dtype: DType,
    ) -> Result<PrimaryCollectiveAddKernel, PtxError> {
        let renderer = PtxRenderer::new(primary.ptx_sm()?)?;
        let rendered = render_collective_add(&renderer, dtype)?;
        let kernel = self
            .kernels
            .get_or_load(primary, rendered.clone(), renderer.block_size)?;
        primary.register_collective_add_semantics(
            kernel.function.identity(),
            &rendered.cache_key,
            dtype,
            COLLECTIVE_ADD_ABI_VERSION,
        );
        Ok(PrimaryCollectiveAddKernel {
            kernel,
            rendered,
            dtype,
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.kernels.len()
    }
}
#[allow(dead_code)]
impl PrimaryCollectiveAddKernel {
    pub(crate) fn rendered(&self) -> &RenderedPtx {
        &self.rendered
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch(
        &self,
        destination: &crate::PrimaryBufferLease,
        destination_offset: usize,
        source: &crate::PrimaryBufferLease,
        source_offset: usize,
        count: usize,
        stream: &Stream,
        synchronize: bool,
    ) -> Result<(), PtxError> {
        collective_add_dtype(self.dtype)?;
        if count == 0 {
            return Ok(());
        }
        let destination_view = destination.view()?;
        let source_view = source.view()?;
        let primary = destination.primary()?;
        if source.primary()?.identity() != primary.identity()
            || !stream.belongs_to_primary(&primary)
        {
            return Err(PtxError::Cuda(CudaError::ContextMismatch));
        }
        let bytes = count
            .checked_mul(self.dtype.itemsize())
            .ok_or(PtxError::Overflow)?;
        let destination_byte_offset = destination_offset
            .checked_mul(self.dtype.itemsize())
            .ok_or(PtxError::Overflow)?;
        let source_byte_offset = source_offset
            .checked_mul(self.dtype.itemsize())
            .ok_or(PtxError::Overflow)?;
        let destination_end = destination_byte_offset
            .checked_add(bytes)
            .ok_or(PtxError::Overflow)?;
        let source_end = source_byte_offset
            .checked_add(bytes)
            .ok_or(PtxError::Overflow)?;
        if destination_end > destination_view.len() || source_end > source_view.len() {
            return Err(PtxError::InvalidBinding(
                "collective add range exceeds logical lease".into(),
            ));
        }
        let destination_ptr = destination_view
            .device_ptr()?
            .checked_add(u64::try_from(destination_byte_offset).map_err(|_| PtxError::Overflow)?)
            .ok_or(PtxError::Overflow)?;
        let source_ptr = source_view
            .device_ptr()?
            .checked_add(u64::try_from(source_byte_offset).map_err(|_| PtxError::Overflow)?)
            .ok_or(PtxError::Overflow)?;
        if destination_ptr % self.dtype.itemsize() as u64 != 0
            || source_ptr % self.dtype.itemsize() as u64 != 0
        {
            return Err(PtxError::InvalidBinding(
                "unaligned collective add pointer".into(),
            ));
        }
        let grid = count
            .checked_add(self.kernel.block_size as usize - 1)
            .ok_or(PtxError::Overflow)?
            / self.kernel.block_size as usize;
        let config = LaunchConfig {
            grid: [u32::try_from(grid).map_err(|_| PtxError::Overflow)?, 1, 1],
            block: [self.kernel.block_size, 1, 1],
            shared_bytes: 0,
        };
        primary.validate_launch(config)?;
        primary.register_collective_add_semantics(
            self.kernel.function.identity(),
            &self.rendered.cache_key,
            self.dtype,
            COLLECTIVE_ADD_ABI_VERSION,
        );
        let mut words = [
            destination_view.device_ptr()?,
            source_view.device_ptr()?,
            destination_offset as u64,
            source_offset as u64,
            count as u64,
        ];
        let mut args: Vec<*mut c_void> = words
            .iter_mut()
            .map(|word| (word as *mut u64).cast())
            .collect();
        self.kernel.function.launch(config, stream, &mut args)?;
        self.kernel.attach_primary_completion(
            stream,
            &[
                PtxBinding {
                    buffer: destination_view,
                    dtype: self.dtype,
                    mutable: true,
                },
                PtxBinding {
                    buffer: source_view,
                    dtype: self.dtype,
                    mutable: false,
                },
            ],
        )?;
        if synchronize {
            stream.synchronize()?;
        }
        Ok(())
    }
}
impl Default for PtxCache {
    fn default() -> Self {
        Self::new()
    }
}
impl PtxCache {
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }
    pub fn get_or_load(
        &mut self,
        context: &crate::Context,
        rendered: RenderedPtx,
        block_size: u32,
    ) -> Result<Rc<PtxKernel>, PtxError> {
        let key = rendered.cache_key.clone();
        if let Some(k) = self.kernels.get(&key) {
            return Ok(k.clone());
        };
        let k = Rc::new(PtxKernel::load(context, Rc::new(rendered), block_size)?);
        self.kernels.insert(key, k.clone());
        Ok(k)
    }
    pub fn get_or_load_primary(
        &mut self,
        context: &crate::PrimaryContext,
        rendered: RenderedPtx,
        block_size: u32,
    ) -> Result<Rc<PtxKernel>, PtxError> {
        let key = rendered.cache_key.clone();
        if let Some(k) = self.kernels.get(&key) {
            return Ok(k.clone());
        }
        let k = Rc::new(PtxKernel::load_primary(
            context,
            Rc::new(rendered),
            block_size,
        )?);
        self.kernels.insert(key, k.clone());
        Ok(k)
    }
    pub fn len(&self) -> usize {
        self.kernels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Driver, UOp, UType};
    use std::sync::{Arc, Barrier};

    fn concurrent_rendered(key: &str) -> RenderedPtx {
        RenderedPtx {
            source: ".version 7.0".into(),
            source_map: BTreeMap::new(),
            buffers: vec![],
            extent: 0,
            cache_key: key.into(),
            entry: "kernel".into(),
            semantic_program: None,
        }
    }
    fn primary(mock: &Arc<crate::cuda::tests::Mock>) -> crate::PrimaryContext {
        Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap()
    }
    #[test]
    fn generic_semantics_registration_follows_primary_cache_lifetime() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let context = primary(&mock);
        let cache = ConcurrentPtxCache::new();
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&unary_kernel(
                DType::F32,
                crate::UnaryOp::Abs,
                crate::Shape::new(vec![4]),
            ))
            .unwrap();
        let first = cache.get_or_load(&context, rendered.clone(), 32).unwrap();
        assert_eq!(mock.generic_kernel_count(), 1);

        let second = cache.get_or_load(&context, rendered, 32).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(mock.generic_kernel_count(), 1);
        drop(second);
        drop(first);
        drop(cache);
        assert_eq!(mock.generic_kernel_count(), 0);
    }
    fn kernel(dtype: DType) -> UOp {
        let range = UOp::constant(4, UType::scalar(DType::I64));
        let addr = UOp::new(
            UOpKind::DefineGlobal,
            Some(UType::scalar(dtype)),
            vec![],
            UArg::None,
        );
        let ix = UOp::new(
            UOpKind::Index,
            Some(UType::scalar(dtype)),
            vec![addr, range],
            UArg::BufferIndex {
                buffer: 1,
                elements: 4,
                input_shape: crate::Shape::new(vec![4]),
                output_shape: crate::Shape::new(vec![4]),
            },
        );
        let load = UOp::new(
            UOpKind::Load,
            Some(UType::scalar(dtype)),
            vec![ix.clone()],
            UArg::None,
        );
        let value = UOp::new(
            UOpKind::GraphBinary(crate::BinaryOp::Add),
            Some(UType::scalar(dtype)),
            vec![load, UOp::constant(1, UType::scalar(dtype))],
            UArg::None,
        );
        UOp::sink(vec![UOp::new(
            UOpKind::Store,
            None,
            vec![ix, value],
            UArg::None,
        )])
    }
    fn unary_kernel(dtype: DType, op: crate::UnaryOp, shape: crate::Shape) -> UOp {
        let elements = shape.numel().unwrap();
        let range = UOp::constant(elements as i64, UType::scalar(DType::I64));
        let index = UOp::new(
            UOpKind::Index,
            Some(UType::scalar(dtype)),
            vec![
                UOp::new(
                    UOpKind::DefineGlobal,
                    Some(UType::scalar(dtype)),
                    vec![],
                    UArg::None,
                ),
                range,
            ],
            UArg::BufferIndex {
                buffer: 1,
                elements,
                input_shape: shape.clone(),
                output_shape: shape,
            },
        );
        let value = UOp::new(
            UOpKind::GraphUnary(op),
            Some(UType::scalar(dtype)),
            vec![UOp::new(
                UOpKind::Load,
                Some(UType::scalar(dtype)),
                vec![index.clone()],
                UArg::None,
            )],
            UArg::None,
        );
        UOp::sink(vec![UOp::new(
            UOpKind::Store,
            None,
            vec![index, value],
            UArg::None,
        )])
    }
    fn broadcast_unary_kernel(dtype: DType, op: crate::UnaryOp) -> UOp {
        let range = UOp::constant(4, UType::scalar(DType::I64));
        let index = |buffer, input_shape: crate::Shape| {
            UOp::new(
                UOpKind::Index,
                Some(UType::scalar(dtype)),
                vec![
                    UOp::new(
                        UOpKind::DefineGlobal,
                        Some(UType::scalar(dtype)),
                        vec![],
                        UArg::None,
                    ),
                    range.clone(),
                ],
                UArg::BufferIndex {
                    buffer,
                    elements: input_shape.numel().unwrap(),
                    input_shape,
                    output_shape: crate::Shape::new(vec![2, 2]),
                },
            )
        };
        let input = index(1, crate::Shape::new(vec![1, 2]));
        let output = index(2, crate::Shape::new(vec![2, 2]));
        let value = UOp::new(
            UOpKind::GraphUnary(op),
            Some(UType::scalar(dtype)),
            vec![UOp::new(
                UOpKind::Load,
                Some(UType::scalar(dtype)),
                vec![input],
                UArg::None,
            )],
            UArg::None,
        );
        UOp::sink(vec![UOp::new(
            UOpKind::Store,
            None,
            vec![output, value],
            UArg::None,
        )])
    }
    fn static_view_unary_kernel(dtype: DType, op: crate::UnaryOp, offset: usize) -> UOp {
        let output_shape = crate::Shape::new(vec![2, 2]);
        let view = crate::ViewMap {
            source_shape: crate::Shape::new(vec![4, 2]),
            logical_shape: output_shape.clone(),
            strides: vec![2, 1],
            offset,
        };
        let range = UOp::constant(4, UType::scalar(DType::I64));
        let input = UOp::new(
            UOpKind::Index,
            Some(UType::scalar(dtype)),
            vec![
                UOp::new(
                    UOpKind::DefineGlobal,
                    Some(UType::scalar(dtype)),
                    vec![],
                    UArg::None,
                ),
                range.clone(),
            ],
            UArg::ViewBufferIndex {
                buffer: 1,
                elements: 4,
                input_shape: output_shape.clone(),
                output_shape: output_shape.clone(),
                view,
            },
        );
        let output = UOp::new(
            UOpKind::Index,
            Some(UType::scalar(dtype)),
            vec![
                UOp::new(
                    UOpKind::DefineGlobal,
                    Some(UType::scalar(dtype)),
                    vec![],
                    UArg::None,
                ),
                range,
            ],
            UArg::BufferIndex {
                buffer: 2,
                elements: 4,
                input_shape: output_shape.clone(),
                output_shape,
            },
        );
        let value = UOp::new(
            UOpKind::GraphUnary(op),
            Some(UType::scalar(dtype)),
            vec![UOp::new(
                UOpKind::Load,
                Some(UType::scalar(dtype)),
                vec![input],
                UArg::None,
            )],
            UArg::None,
        );
        UOp::sink(vec![UOp::new(
            UOpKind::Store,
            None,
            vec![output, value],
            UArg::None,
        )])
    }
    fn broadcast_add_kernel(dtype: DType) -> UOp {
        let range = UOp::constant(4, UType::scalar(DType::I64));
        let index = |buffer, shape: Vec<usize>| {
            UOp::new(
                UOpKind::Index,
                Some(UType::scalar(dtype)),
                vec![
                    UOp::new(
                        UOpKind::DefineGlobal,
                        Some(UType::scalar(dtype)),
                        vec![],
                        UArg::None,
                    ),
                    range.clone(),
                ],
                UArg::BufferIndex {
                    buffer,
                    elements: shape.iter().product(),
                    input_shape: crate::Shape::new(shape),
                    output_shape: crate::Shape::new(vec![2, 2]),
                },
            )
        };
        let left = index(1, vec![2, 2]);
        let right = index(2, vec![1, 2]);
        let out = index(3, vec![2, 2]);
        let add = UOp::new(
            UOpKind::GraphBinary(crate::BinaryOp::Add),
            Some(UType::scalar(dtype)),
            vec![
                UOp::new(
                    UOpKind::Load,
                    Some(UType::scalar(dtype)),
                    vec![left],
                    UArg::None,
                ),
                UOp::new(
                    UOpKind::Load,
                    Some(UType::scalar(dtype)),
                    vec![right],
                    UArg::None,
                ),
            ],
            UArg::None,
        );
        UOp::sink(vec![UOp::new(
            UOpKind::Store,
            None,
            vec![out, add],
            UArg::None,
        )])
    }

    fn static_view_add_kernel(dtype: DType, offset: usize) -> UOp {
        let output_shape = crate::Shape::new(vec![2, 2]);
        let source_shape = crate::Shape::new(vec![4, 2]);
        let view = crate::ViewMap {
            source_shape: source_shape.clone(),
            logical_shape: output_shape.clone(),
            strides: vec![2, 1],
            offset,
        };
        let range = UOp::constant(4, UType::scalar(DType::I64));
        let index = |buffer| {
            UOp::new(
                UOpKind::Index,
                Some(UType::scalar(dtype)),
                vec![
                    UOp::new(
                        UOpKind::DefineGlobal,
                        Some(UType::scalar(dtype)),
                        vec![],
                        UArg::None,
                    ),
                    range.clone(),
                ],
                UArg::ViewBufferIndex {
                    buffer,
                    elements: 4,
                    input_shape: output_shape.clone(),
                    output_shape: output_shape.clone(),
                    view: view.clone(),
                },
            )
        };
        let left = index(1);
        let right = index(2);
        let output = UOp::new(
            UOpKind::Index,
            Some(UType::scalar(dtype)),
            vec![
                UOp::new(
                    UOpKind::DefineGlobal,
                    Some(UType::scalar(dtype)),
                    vec![],
                    UArg::None,
                ),
                range,
            ],
            UArg::BufferIndex {
                buffer: 3,
                elements: 4,
                input_shape: output_shape.clone(),
                output_shape,
            },
        );
        let value = UOp::new(
            UOpKind::GraphBinary(crate::BinaryOp::Add),
            Some(UType::scalar(dtype)),
            vec![
                UOp::new(
                    UOpKind::Load,
                    Some(UType::scalar(dtype)),
                    vec![left],
                    UArg::None,
                ),
                UOp::new(
                    UOpKind::Load,
                    Some(UType::scalar(dtype)),
                    vec![right],
                    UArg::None,
                ),
            ],
            UArg::None,
        );
        UOp::sink(vec![UOp::new(
            UOpKind::Store,
            None,
            vec![output, value],
            UArg::None,
        )])
    }

    #[test]
    fn unary_rendering_is_exact_and_rejects_unowned_math_contracts() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, op, instruction) in [
            (DType::I32, crate::UnaryOp::Neg, "neg.s32"),
            (DType::I64, crate::UnaryOp::Abs, "abs.s64"),
            (DType::F32, crate::UnaryOp::Abs, "abs.f32"),
            (DType::F64, crate::UnaryOp::Neg, "neg.f64"),
        ] {
            let rendered = renderer
                .render(&unary_kernel(dtype, op, crate::Shape::new(vec![4])))
                .unwrap();
            assert!(rendered.source.contains(instruction), "{dtype:?} {op:?}");
            assert!(!rendered.source_map.is_empty(), "{dtype:?} {op:?}");
        }
        for (dtype, op) in [
            (DType::Bool, crate::UnaryOp::Neg),
            (DType::U32, crate::UnaryOp::Abs),
            (DType::F16, crate::UnaryOp::Neg),
            (DType::F32, crate::UnaryOp::Exp),
            (DType::F64, crate::UnaryOp::Sqrt),
        ] {
            assert!(matches!(
                renderer.render(&unary_kernel(dtype, op, crate::Shape::new(vec![4]))),
                Err(PtxError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn mock_unary_semantics_are_exact_for_scalars_and_signed_float_edges() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let pool = primary.allocator();
        let cache = ConcurrentPtxCache::new();
        let i32_bytes = |values: &[i32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let f64_bytes = |values: &[f64]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let cases = [
            (
                "i32 wrapping neg",
                DType::I32,
                crate::UnaryOp::Neg,
                crate::Shape::new(vec![4]),
                i32_bytes(&[i32::MIN, -7, 0, 5]),
                i32_bytes(&[i32::MIN, 7, 0, -5]),
            ),
            (
                "f32 abs clears signed zero",
                DType::F32,
                crate::UnaryOp::Abs,
                crate::Shape::new(vec![4]),
                f32_bytes(&[-0.0, -2.5, 3.0, f32::NEG_INFINITY]),
                f32_bytes(&[0.0, 2.5, 3.0, f32::INFINITY]),
            ),
            (
                "f64 scalar neg preserves arithmetic semantics",
                DType::F64,
                crate::UnaryOp::Neg,
                crate::Shape::new(vec![]),
                f64_bytes(&[-2.5]),
                f64_bytes(&[2.5]),
            ),
        ];
        for (name, dtype, op, shape, input, expected) in cases {
            let lease = pool
                .allocate(NonZeroUsize::new(input.len()).unwrap())
                .unwrap();
            lease.view().unwrap().copy_from(0, &input).unwrap();
            let rendered = PtxRenderer::new(80)
                .unwrap()
                .render(&unary_kernel(dtype, op, shape))
                .unwrap();
            let kernel = cache.get_or_load(&primary, rendered, 32).unwrap();
            kernel
                .launch(
                    &stream,
                    &[PtxBinding {
                        buffer: lease.view().unwrap(),
                        dtype,
                        mutable: true,
                    }],
                    true,
                )
                .unwrap();
            let mut actual = vec![0; input.len()];
            lease.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected, "{name}");
        }
    }

    #[test]
    fn mock_unary_semantics_preserve_broadcast_view_and_zero_extent_contracts() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let pool = primary.allocator();
        let cache = ConcurrentPtxCache::new();
        let input = pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let broadcast_out = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        input
            .view()
            .unwrap()
            .copy_from(
                0,
                &[-3_i32, i32::MIN]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let broadcast = cache
            .get_or_load(
                &primary,
                PtxRenderer::new(80)
                    .unwrap()
                    .render(&broadcast_unary_kernel(DType::I32, crate::UnaryOp::Abs))
                    .unwrap(),
                32,
            )
            .unwrap();
        broadcast
            .launch(
                &stream,
                &[
                    PtxBinding {
                        buffer: input.view().unwrap(),
                        dtype: DType::I32,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: broadcast_out.view().unwrap(),
                        dtype: DType::I32,
                        mutable: true,
                    },
                ],
                true,
            )
            .unwrap();
        let mut broadcast_bytes = [0; 16];
        broadcast_out
            .view()
            .unwrap()
            .copy_to(0, &mut broadcast_bytes)
            .unwrap();
        assert_eq!(
            broadcast_bytes,
            [3_i32, i32::MIN, 3, i32::MIN]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
                .as_slice()
        );

        let view_input = pool.allocate(NonZeroUsize::new(32).unwrap()).unwrap();
        let view_out = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        view_input
            .view()
            .unwrap()
            .copy_from(
                0,
                &[1.0_f32, 2.0, 3.0, 4.0, 5.0, -6.0, 7.0, -8.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let view = cache
            .get_or_load(
                &primary,
                PtxRenderer::new(80)
                    .unwrap()
                    .render(&static_view_unary_kernel(
                        DType::F32,
                        crate::UnaryOp::Neg,
                        4,
                    ))
                    .unwrap(),
                32,
            )
            .unwrap();
        view.launch(
            &stream,
            &[
                PtxBinding {
                    buffer: view_input.view().unwrap(),
                    dtype: DType::F32,
                    mutable: false,
                },
                PtxBinding {
                    buffer: view_out.view().unwrap(),
                    dtype: DType::F32,
                    mutable: true,
                },
            ],
            true,
        )
        .unwrap();
        let mut view_bytes = [0; 16];
        view_out
            .view()
            .unwrap()
            .copy_to(0, &mut view_bytes)
            .unwrap();
        assert_eq!(
            view_bytes,
            [-5.0_f32, 6.0, -7.0, 8.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
                .as_slice()
        );

        let zero = cache
            .get_or_load(
                &primary,
                PtxRenderer::new(80)
                    .unwrap()
                    .render(&unary_kernel(
                        DType::I32,
                        crate::UnaryOp::Neg,
                        crate::Shape::new(vec![0]),
                    ))
                    .unwrap(),
                32,
            )
            .unwrap();
        let untouched = pool.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        untouched
            .view()
            .unwrap()
            .copy_from(0, &[9, 0, 0, 0])
            .unwrap();
        zero.launch(
            &stream,
            &[PtxBinding {
                buffer: untouched.view().unwrap(),
                dtype: DType::I32,
                mutable: true,
            }],
            true,
        )
        .unwrap();
        let mut bytes = [0; 4];
        untouched.view().unwrap().copy_to(0, &mut bytes).unwrap();
        assert_eq!(bytes, [9, 0, 0, 0]);
    }

    #[test]
    fn mock_generic_semantics_executes_broadcast_add_without_cpu_backend() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let pool = primary.allocator();
        let left = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        let right = pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let out = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        left.view()
            .unwrap()
            .copy_from(
                0,
                &crate::TensorData::from_le_bytes(
                    [2, 2],
                    DType::F32,
                    &[0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64],
                )
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            )
            .unwrap();
        right
            .view()
            .unwrap()
            .copy_from(
                0,
                &crate::TensorData::from_le_bytes(
                    [1, 2],
                    DType::F32,
                    &[0, 0, 32, 65, 0, 0, 160, 65],
                )
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            )
            .unwrap();
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&broadcast_add_kernel(DType::F32))
            .unwrap();
        let cache = ConcurrentPtxCache::new();
        let kernel = cache.get_or_load(&primary, rendered, 32).unwrap();
        kernel
            .launch(
                &stream,
                &[
                    PtxBinding {
                        buffer: left.view().unwrap(),
                        dtype: DType::F32,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: right.view().unwrap(),
                        dtype: DType::F32,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: out.view().unwrap(),
                        dtype: DType::F32,
                        mutable: true,
                    },
                ],
                true,
            )
            .unwrap();
        let mut actual = vec![0; 16];
        out.view().unwrap().copy_to(0, &mut actual).unwrap();
        assert_eq!(
            actual,
            crate::TensorData::from_le_bytes(
                [2, 2],
                DType::F32,
                &[0, 0, 48, 65, 0, 0, 176, 65, 0, 0, 80, 65, 0, 0, 192, 65,]
            )
            .unwrap()
            .to_le_bytes()
            .unwrap()
        );
        assert_eq!(mock.generic_kernel_count(), 1);
    }

    #[test]
    fn generic_semantics_reads_static_views_from_physical_source_storage() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let pool = primary.allocator();
        let left = pool.allocate(NonZeroUsize::new(32).unwrap()).unwrap();
        let right = pool.allocate(NonZeroUsize::new(32).unwrap()).unwrap();
        let output = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        let stream = primary.stream().unwrap();
        for (name, offset, expected) in [
            ("first", 0, vec![11_i32, 22, 33, 44]),
            ("second", 4, vec![55_i32, 66, 77, 88]),
        ] {
            let rendered = PtxRenderer::new(80)
                .unwrap()
                .render(&static_view_add_kernel(DType::I32, offset))
                .unwrap();
            assert!(
                rendered
                    .buffers
                    .iter()
                    .filter(|abi| !abi.mutable)
                    .all(|abi| abi.source_shape == crate::Shape::new(vec![4, 2])
                        && abi.elements == 8)
            );
            left.view()
                .unwrap()
                .copy_from(
                    0,
                    &[1_i32, 2, 3, 4, 5, 6, 7, 8]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            right
                .view()
                .unwrap()
                .copy_from(
                    0,
                    &[10_i32, 20, 30, 40, 50, 60, 70, 80]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let cache = ConcurrentPtxCache::new();
            let kernel = cache.get_or_load(&primary, rendered, 32).unwrap();
            kernel
                .launch(
                    &stream,
                    &[
                        PtxBinding {
                            buffer: left.view().unwrap(),
                            dtype: DType::I32,
                            mutable: false,
                        },
                        PtxBinding {
                            buffer: right.view().unwrap(),
                            dtype: DType::I32,
                            mutable: false,
                        },
                        PtxBinding {
                            buffer: output.view().unwrap(),
                            dtype: DType::I32,
                            mutable: true,
                        },
                    ],
                    true,
                )
                .unwrap();
            let mut bytes = [0; 16];
            output.view().unwrap().copy_to(0, &mut bytes).unwrap();
            let actual = bytes
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{name}");
        }
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        left.view()
            .unwrap()
            .copy_from(0, &f32_bytes(&[1., 2., 3., 4., 5., 6., 7., 8.]))
            .unwrap();
        right
            .view()
            .unwrap()
            .copy_from(0, &f32_bytes(&[10., 20., 30., 40., 50., 60., 70., 80.]))
            .unwrap();
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&static_view_add_kernel(DType::F32, 4))
            .unwrap();
        let cache = ConcurrentPtxCache::new();
        let kernel = cache.get_or_load(&primary, rendered, 32).unwrap();
        kernel
            .launch(
                &stream,
                &[
                    PtxBinding {
                        buffer: left.view().unwrap(),
                        dtype: DType::F32,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: right.view().unwrap(),
                        dtype: DType::F32,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: output.view().unwrap(),
                        dtype: DType::F32,
                        mutable: true,
                    },
                ],
                true,
            )
            .unwrap();
        let mut bytes = [0; 16];
        output.view().unwrap().copy_to(0, &mut bytes).unwrap();
        assert_eq!(f32_bytes(&[55., 66., 77., 88.]), bytes);
    }

    #[test]
    fn generic_mock_semantics_table_wraps_and_partitions_primary_owners() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let first = primary(&mock);
        let second = primary(&mock);
        let cache = ConcurrentPtxCache::new();
        let cases = [
            (
                DType::I32,
                i32::MAX.to_ne_bytes().to_vec(),
                i32::MIN.to_ne_bytes().to_vec(),
            ),
            (
                DType::U64,
                u64::MAX.to_ne_bytes().to_vec(),
                0_u64.to_ne_bytes().to_vec(),
            ),
            (
                DType::F32,
                1.5_f32.to_ne_bytes().to_vec(),
                2.5_f32.to_ne_bytes().to_vec(),
            ),
            (
                DType::F64,
                1.5_f64.to_ne_bytes().to_vec(),
                2.5_f64.to_ne_bytes().to_vec(),
            ),
        ];
        for (dtype, first_value, expected) in cases {
            let pool = first.allocator();
            let lease = pool
                .allocate(NonZeroUsize::new(dtype.itemsize() * 4).unwrap())
                .unwrap();
            lease.view().unwrap().copy_from(0, &first_value).unwrap();
            let rendered = PtxRenderer::new(80)
                .unwrap()
                .render(&kernel(dtype))
                .unwrap();
            let compiled = cache.get_or_load(&first, rendered, 32).unwrap();
            let stream = first.stream().unwrap();
            compiled
                .launch(
                    &stream,
                    &[PtxBinding {
                        buffer: lease.view().unwrap(),
                        dtype,
                        mutable: true,
                    }],
                    true,
                )
                .unwrap();
            let mut actual = vec![0; dtype.itemsize()];
            lease.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected, "{dtype:?}");
        }
        let _first_buffer = first.allocate(NonZeroUsize::new(32).unwrap()).unwrap();
        let second_buffer = second.allocate(NonZeroUsize::new(32).unwrap()).unwrap();
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&kernel(DType::I32))
            .unwrap();
        let compiled = cache.get_or_load(&first, rendered, 32).unwrap();
        let stream = first.stream().unwrap();
        assert!(matches!(
            compiled.launch(
                &stream,
                &[PtxBinding {
                    buffer: second_buffer.view(),
                    dtype: DType::I32,
                    mutable: true
                }],
                true
            ),
            Err(PtxError::Cuda(CudaError::ContextMismatch))
        ));
        assert_eq!(mock.generic_kernel_count(), 4);
    }

    #[test]
    fn mock_dtod_failure_is_one_shot_and_precedes_mutation() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let source = primary.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        let destination = primary.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        source.copy_from(0, &[1, 2, 3, 4]).unwrap();
        mock.fail_dtod_after(0, 2);
        assert!(destination.copy_from_device(0, &source, 0, 4).is_err());
        let mut bytes = [0; 4];
        destination.copy_to(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0; 4]);
        destination.copy_from_device(0, &source, 0, 4).unwrap();
        destination.copy_to(0, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
    }
    #[test]
    fn snapshot_is_deterministic_and_has_abi() {
        let a = PtxRenderer::new(80)
            .unwrap()
            .render(&kernel(DType::F32))
            .unwrap();
        let b = PtxRenderer::new(80)
            .unwrap()
            .render(&kernel(DType::F32))
            .unwrap();
        assert_eq!(a.source, b.source);
        assert!(a.source.contains(".target sm_80\n.address_size 64"));
        assert!(a.source.contains(".param .u64 p0"));
        assert!(a.source.contains("mad.lo.u32"));
        assert_eq!(a.buffers[0].dtype, DType::F32);
    }
    #[test]
    fn narrow_floats_are_explicitly_rejected() {
        assert!(matches!(
            PtxRenderer::new(80).unwrap().render(&kernel(DType::F16)),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn collective_add_rendering_has_stable_five_word_abi() {
        let renderer = PtxRenderer::new(80).unwrap();
        let first = render_collective_add(&renderer, DType::I32).unwrap();
        let second = render_collective_add(&renderer, DType::I32).unwrap();
        assert_eq!(first.source, second.source);
        assert_eq!(first.cache_key, second.cache_key);
        assert!(first.source.contains("destination_offset"));
        assert!(first.source.contains("add.s32"));
        assert!(matches!(
            render_collective_add(&renderer, DType::Bool),
            Err(PtxError::Unsupported(_))
        ));
        assert!(matches!(
            render_collective_add(&renderer, DType::F16),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn mock_collective_add_mutates_exact_scalar_bytes_and_reuses_cache() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let cache = PrimaryCollectiveAddCache::new();
        let stream = primary.stream().unwrap();
        let cases = vec![
            (DType::I8, vec![127], vec![1], vec![128]),
            (DType::U8, vec![255], vec![1], vec![0]),
            (
                DType::I32,
                i32::MAX.to_ne_bytes().to_vec(),
                1_i32.to_ne_bytes().to_vec(),
                i32::MIN.to_ne_bytes().to_vec(),
            ),
            (
                DType::U32,
                u32::MAX.to_ne_bytes().to_vec(),
                1_u32.to_ne_bytes().to_vec(),
                0_u32.to_ne_bytes().to_vec(),
            ),
            (
                DType::I64,
                i64::MAX.to_ne_bytes().to_vec(),
                1_i64.to_ne_bytes().to_vec(),
                i64::MIN.to_ne_bytes().to_vec(),
            ),
            (
                DType::U64,
                u64::MAX.to_ne_bytes().to_vec(),
                1_u64.to_ne_bytes().to_vec(),
                0_u64.to_ne_bytes().to_vec(),
            ),
            (
                DType::F32,
                1.5_f32.to_ne_bytes().to_vec(),
                2.25_f32.to_ne_bytes().to_vec(),
                3.75_f32.to_ne_bytes().to_vec(),
            ),
            (
                DType::F64,
                1.5_f64.to_ne_bytes().to_vec(),
                2.25_f64.to_ne_bytes().to_vec(),
                3.75_f64.to_ne_bytes().to_vec(),
            ),
        ];
        for (dtype, destination_bytes, source_bytes, expected) in cases {
            let pool = primary.allocator();
            let destination = pool.allocate(NonZeroUsize::new(32).unwrap()).unwrap();
            let source = pool.allocate(NonZeroUsize::new(32).unwrap()).unwrap();
            destination
                .view()
                .unwrap()
                .copy_from(dtype.itemsize(), &destination_bytes)
                .unwrap();
            source
                .view()
                .unwrap()
                .copy_from(dtype.itemsize() * 2, &source_bytes)
                .unwrap();
            let kernel = cache.get_or_load(&primary, dtype).unwrap();
            kernel
                .launch(&destination, 1, &source, 2, 1, &stream, true)
                .unwrap();
            let mut actual = vec![0; dtype.itemsize()];
            destination
                .view()
                .unwrap()
                .copy_to(dtype.itemsize(), &mut actual)
                .unwrap();
            assert_eq!(actual, expected, "{dtype:?}");
        }
        let first = cache.get_or_load(&primary, DType::I32).unwrap();
        let second = cache.get_or_load(&primary, DType::I32).unwrap();
        assert_eq!(first.rendered().cache_key, second.rendered().cache_key);
        assert_eq!(cache.len(), 8);
    }

    #[test]
    fn collective_add_keeps_colliding_owners_and_failed_launches_isolated() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let first = primary(&mock);
        let second = primary(&mock);
        let cache = PrimaryCollectiveAddCache::new();
        let mut jobs = Vec::new();
        for (primary, value) in [(&first, 1_u32), (&second, 10_u32)] {
            let pool = primary.allocator();
            let destination = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
            let source = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
            destination
                .view()
                .unwrap()
                .copy_from(4, &value.to_ne_bytes())
                .unwrap();
            source
                .view()
                .unwrap()
                .copy_from(4, &2_u32.to_ne_bytes())
                .unwrap();
            jobs.push((primary, destination, source, value + 2));
        }
        assert_eq!(
            jobs[0].1.view().unwrap().device_ptr().unwrap(),
            jobs[1].1.view().unwrap().device_ptr().unwrap()
        );
        for (primary, destination, source, expected) in &jobs {
            let kernel = cache.get_or_load(primary, DType::U32).unwrap();
            let stream = primary.stream().unwrap();
            kernel
                .launch(destination, 1, source, 1, 1, &stream, true)
                .unwrap();
            let mut bytes = [0; 4];
            destination.view().unwrap().copy_to(4, &mut bytes).unwrap();
            assert_eq!(u32::from_ne_bytes(bytes), *expected);
        }
        let (primary, destination, source, expected) = &jobs[0];
        let kernel = cache.get_or_load(primary, DType::U32).unwrap();
        let stream = primary.stream().unwrap();
        mock.set_launch_result(2);
        assert!(
            kernel
                .launch(destination, 1, source, 1, 1, &stream, true)
                .is_err()
        );
        mock.set_launch_result(0);
        let mut bytes = [0; 4];
        destination.view().unwrap().copy_to(4, &mut bytes).unwrap();
        assert_eq!(u32::from_ne_bytes(bytes), *expected);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn mock_module_cache_and_launch_follow_abi() {
        use crate::cuda::tests::{Mock, context};
        use std::{num::NonZeroUsize, sync::Arc};
        let mock = Arc::new(Mock::default());
        let context = context(&mock);
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&kernel(DType::F32))
            .unwrap();
        let mut cache = PtxCache::new();
        let first = cache.get_or_load(&context, rendered.clone(), 32).unwrap();
        let second = cache.get_or_load(&context, rendered, 32).unwrap();
        assert!(Rc::ptr_eq(&first, &second));
        let buffer = context.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        let stream = context.stream().unwrap();
        first
            .launch(
                &stream,
                &[PtxBinding {
                    buffer: buffer.view(),
                    dtype: DType::F32,
                    mutable: true,
                }],
                true,
            )
            .unwrap();
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|&&x| x == "module_load").count(), 1);
        assert!(
            calls.contains(&"function")
                && calls.contains(&"launch")
                && calls.contains(&"stream_sync")
        );
    }

    #[test]
    fn primary_profiled_launch_records_and_collects() {
        use crate::cuda_profile::ProfilingSession;
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        mock.set_elapsed_support(true);
        mock.set_event_ready(true);
        let context = primary(&mock);
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&kernel(DType::F32))
            .unwrap();
        let cache = ConcurrentPtxCache::new();
        let kernel = cache.get_or_load(&context, rendered, 32).unwrap();
        let buffer = context.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        let stream = context.stream().unwrap();
        let session = ProfilingSession::enabled(context.identity(), context.device());
        let bindings = [PtxBinding {
            buffer: buffer.view(),
            dtype: DType::F32,
            mutable: true,
        }];
        let sample = kernel
            .launch_profiled(&session, "add", &context, &stream, &bindings, false)
            .unwrap()
            .unwrap();
        assert_eq!(sample.collect().unwrap(), 1_500_000);
        assert!(mock.calls().contains(&"launch"));
    }

    #[test]
    fn concurrent_primary_same_key_deduplicates_driver_load() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let context = Arc::new(primary(&mock));
        let cache = Arc::new(ConcurrentPtxCache::new());
        let start = Arc::new(Barrier::new(5));
        let mut workers = Vec::new();
        for _ in 0..4 {
            let cache = cache.clone();
            let context = context.clone();
            let start = start.clone();
            workers.push(std::thread::spawn(move || {
                start.wait();
                cache
                    .get_or_load(&context, concurrent_rendered("same"), 32)
                    .unwrap()
            }));
        }
        start.wait();
        let kernels: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert!(
            kernels
                .windows(2)
                .all(|pair| Arc::ptr_eq(&pair[0], &pair[1]))
        );
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|&&x| x == "module_load").count(), 1);
        assert_eq!(calls.iter().filter(|&&x| x == "function").count(), 1);
    }

    #[test]
    fn concurrent_primary_keys_and_owners_are_isolated() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let first = Arc::new(primary(&mock));
        let second = Arc::new(primary(&mock));
        let cache = Arc::new(ConcurrentPtxCache::new());
        let start = Arc::new(Barrier::new(4));
        let jobs = [(first.clone(), "a"), (first, "b"), (second.clone(), "a")];
        let workers: Vec<_> = jobs
            .into_iter()
            .map(|(context, key)| {
                let cache = cache.clone();
                let start = start.clone();
                std::thread::spawn(move || {
                    start.wait();
                    cache
                        .get_or_load(&context, concurrent_rendered(key), 32)
                        .unwrap()
                })
            })
            .collect();
        start.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(cache.len(), 3);
        assert_eq!(
            mock.calls().iter().filter(|&&x| x == "module_load").count(),
            3
        );
    }

    #[test]
    fn concurrent_primary_failure_is_structured_and_retryable() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let context = primary(&mock);
        let cache = ConcurrentPtxCache::new();
        mock.set_module_result(2);
        let first = cache.get_or_load(&context, concurrent_rendered("retry"), 32);
        assert!(matches!(
            first,
            Err(PtxError::Cuda(CudaError::Driver { code: 2, .. }))
        ));
        assert_eq!(cache.len(), 0);
        mock.set_module_result(0);
        assert!(
            cache
                .get_or_load(&context, concurrent_rendered("retry"), 32)
                .is_ok()
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(
            mock.calls().iter().filter(|&&x| x == "module_load").count(),
            2
        );
    }
}
