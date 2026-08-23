//! Deterministic phase-one PTX rendering and Driver launch glue.
//!
//! The renderer intentionally accepts only the fused elementwise UOp subset
//! that has a clear PTX contract. The CPU UOp interpreter remains the semantic
//! oracle; reductions, narrow floats, guarded integer division/shifts and
//! device-status reporting are rejected instead of silently changing meaning.

use crate::cuda_profile::{Metadata, OperationKind, ProfilingSession, TimedSample, TimingError};
use crate::{CudaError, DType, DeviceBuffer, Function, LaunchConfig, Stream, UArg, UOp, UOpKind};
use std::{
    collections::{BTreeMap, HashMap},
    ffi::{CString, c_void},
    fmt,
    rc::Rc,
    sync::{Arc, Condvar, Mutex},
};

pub const PTX_RENDERER_VERSION: &str = "rustgrad-ptx-elementwise-v1";
pub const PTX_ABI_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PtxBufferAbi {
    pub id: u64,
    pub dtype: DType,
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
        if let UArg::BufferIndex {
            buffer, elements, ..
        } = node.arg()
        {
            let dtype = node
                .ty()
                .ok_or_else(|| PtxError::Unsupported("untyped index".into()))?
                .scalar;
            reject_dtype(dtype)?;
            abi.entry(*buffer).or_insert(PtxBufferAbi {
                id: *buffer,
                dtype,
                elements: *elements,
                mutable: false,
            });
        }
    }
    abi.get_mut(out_id)
        .ok_or_else(|| PtxError::Unsupported("output missing ABI".into()))?
        .mutable = true;
    let buffers: Vec<_> = abi.into_values().collect();
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
    let key = stable_key(&(PTX_RENDERER_VERSION, renderer.sm, &source));
    let _ = output_shape;
    Ok(RenderedPtx {
        source,
        source_map: map,
        buffers,
        extent: *extent,
        cache_key: key,
        entry,
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
        UOpKind::Const => {
            let UArg::Int(v) = n.arg() else {
                return Err(PtxError::Unsupported("non-integer constant".into()));
            };
            lines.push(format!("  mov.{} {dst}, {v};", ptx_type(ty)));
        }
        UOpKind::Load => {
            let ix = n
                .sources()
                .first()
                .ok_or_else(|| PtxError::Unsupported("Load without index".into()))?;
            let UArg::BufferIndex {
                buffer,
                input_shape,
                output_shape,
                ..
            } = ix.arg()
            else {
                return Err(PtxError::Unsupported("Load index".into()));
            };
            let b = ids[buffer] + 1;
            let off = broadcast_offset(input_shape.dims(), output_shape.dims())?;
            lines.extend(off);
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
fn stable_key(value: &impl std::hash::Hash) -> String {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    format!("{:016x}", h.finish())
}

pub struct PtxBinding<'a> {
    pub buffer: &'a DeviceBuffer,
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
        if synchronize {
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
}
/// In-flight primary PTX launch profiling. The sample borrows the launch
/// stream and bindings, retaining those resources and the kernel through an
/// explicit timing query, wait, collect, failure, or abandonment.
#[allow(dead_code)] // crate-private profiling API is consumed by submission integration.
pub(crate) struct ProfiledPrimaryPtxSample<'a> {
    timing: TimedSample<'a>,
    _kernel: &'a PrimaryPtxKernel,
    _buffers: Vec<&'a DeviceBuffer>,
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
        Ok(Self {
            rendered,
            module,
            function,
            block_size,
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
            if got.buffer.device() != self.module.device() {
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
        if synchronize {
            stream.synchronize()?;
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
                    buffer: &buffer,
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
            buffer: &buffer,
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
