//! Scalar C11 CPU renderer and shared-library JIT.
//!
//! The native entry point is deliberately small and stable: `void kernel(void
//! **buffers, const int64_t *symbols)`. Buffers are ordered by ascending UOp
//! buffer id; shapes and dtypes are validated by the caller before this unsafe
//! boundary is crossed.  This module never allocates executable memory: the OS
//! dynamic loader owns executable mappings and `JitKernel` owns the library.
use crate::{DType, SymbolicShape, SymbolicVar, UArg, UOp, UOpKind};
use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_int, c_void},
    fmt, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, OnceLock},
};

pub const RENDERER_VERSION: &str = "rustgrad-c11-scalar-v4";
pub const ABI_VERSION: u32 = 2;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum JitError {
    Unsupported(String),
    InvalidBuffer(String),
    DivisionByZero { index: usize },
    InvalidShift { index: usize },
    IndexOutOfBounds { index: usize },
    Compiler { status: Option<i32>, stderr: String },
    Loader(String),
    Io(String),
    Symbolic(String),
}
impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "unsupported CPU JIT UOp: {s}"),
            Self::InvalidBuffer(s) => write!(f, "invalid CPU JIT buffer: {s}"),
            Self::DivisionByZero { index } => write!(f, "CPU JIT division by zero at {index}"),
            Self::InvalidShift { index } => write!(f, "CPU JIT invalid shift at {index}"),
            Self::IndexOutOfBounds { index } => {
                write!(f, "CPU JIT movement index out of bounds at {index}")
            }
            Self::Compiler { status, stderr } => {
                write!(f, "C compiler failed ({status:?}): {stderr}")
            }
            Self::Loader(s) => write!(f, "dynamic loader failed: {s}"),
            Self::Io(s) => write!(f, "CPU JIT I/O failed: {s}"),
            Self::Symbolic(s) => write!(f, "CPU JIT specialization failed: {s}"),
        }
    }
}
impl std::error::Error for JitError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAbi {
    pub version: u32,
    pub buffers: Vec<BufferAbi>,
    pub symbol_count: usize,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferAbi {
    pub id: u64,
    pub dtype: DType,
    pub elements: usize,
    pub mutable: bool,
}
#[derive(Clone, Debug)]
pub struct RenderedC {
    pub source: String,
    pub source_map: BTreeMap<usize, usize>,
    pub abi: KernelAbi,
    pub cache_key: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VectorPlan {
    pub lanes: usize,
    pub enabled: bool,
    pub reason: String,
}

/// An owned byte buffer for the C ABI. Alignment is checked before invocation;
/// callers may use it for externally allocated buffers by constructing it from
/// an aligned `Vec<u8>` allocation only.
#[derive(Clone, Debug)]
pub struct JitBuffer {
    pub dtype: DType,
    pub elements: usize,
    pub mutable: bool,
    bytes: Vec<u8>,
}
impl JitBuffer {
    pub fn from_tensor(data: &crate::TensorData, mutable: bool) -> Self {
        let mut out = Self::zeroed(data.dtype(), data.len(), mutable);
        macro_rules! copy {
            ($values:expr) => {
                for (dst, value) in out
                    .bytes
                    .chunks_exact_mut(data.dtype().itemsize())
                    .zip($values)
                {
                    dst.copy_from_slice(&value.to_ne_bytes());
                }
            };
        }
        match data.storage() {
            crate::Storage::Bool(values) => {
                for (dst, value) in out.bytes.iter_mut().zip(values) {
                    *dst = u8::from(*value);
                }
            }
            crate::Storage::I8(values) => {
                for (dst, value) in out.bytes.iter_mut().zip(values) {
                    *dst = *value as u8;
                }
            }
            crate::Storage::U8(values) => out.bytes.copy_from_slice(values),
            crate::Storage::I16(values) => copy!(values),
            crate::Storage::U16(values)
            | crate::Storage::F16(values)
            | crate::Storage::BF16(values) => copy!(values),
            crate::Storage::I32(values) => copy!(values),
            crate::Storage::U32(values) => copy!(values),
            crate::Storage::I64(values) => copy!(values),
            crate::Storage::U64(values) => copy!(values),
            crate::Storage::F32(values) => copy!(values),
            crate::Storage::F64(values) => copy!(values),
        }
        out
    }
    pub fn into_tensor(self, shape: crate::Shape) -> crate::Result<crate::TensorData> {
        if matches!(self.dtype, DType::F16 | DType::BF16) {
            let raw: Vec<u16> = self
                .bytes
                .chunks_exact(2)
                .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            return crate::TensorData::from_storage(
                shape,
                if self.dtype == DType::F16 {
                    crate::Storage::F16(raw)
                } else {
                    crate::Storage::BF16(raw)
                },
            );
        }
        let scalars: Vec<crate::Scalar> = match self.dtype {
            DType::Bool => self
                .bytes
                .into_iter()
                .map(|v| crate::Scalar::Bool(v != 0))
                .collect(),
            DType::I8 => self
                .bytes
                .into_iter()
                .map(|v| crate::Scalar::I(v as i8 as i64))
                .collect(),
            DType::U8 => self
                .bytes
                .into_iter()
                .map(|v| crate::Scalar::U(v as u64))
                .collect(),
            _ => (0..self.elements)
                .map(|i| {
                    let start = i * self.dtype.itemsize();
                    let b = &self.bytes[start..start + self.dtype.itemsize()];
                    match self.dtype {
                        DType::I16 => {
                            crate::Scalar::I(i16::from_ne_bytes(b.try_into().unwrap()) as i64)
                        }
                        DType::U16 => {
                            crate::Scalar::U(u16::from_ne_bytes(b.try_into().unwrap()) as u64)
                        }
                        DType::I32 => {
                            crate::Scalar::I(i32::from_ne_bytes(b.try_into().unwrap()) as i64)
                        }
                        DType::U32 => {
                            crate::Scalar::U(u32::from_ne_bytes(b.try_into().unwrap()) as u64)
                        }
                        DType::I64 => crate::Scalar::I(i64::from_ne_bytes(b.try_into().unwrap())),
                        DType::U64 => crate::Scalar::U(u64::from_ne_bytes(b.try_into().unwrap())),
                        DType::F16 | DType::BF16 => {
                            crate::Scalar::U(u16::from_ne_bytes(b.try_into().unwrap()) as u64)
                        }
                        DType::F32 => {
                            crate::Scalar::F(f32::from_ne_bytes(b.try_into().unwrap()) as f64)
                        }
                        DType::F64 => crate::Scalar::F(f64::from_ne_bytes(b.try_into().unwrap())),
                        _ => unreachable!(),
                    }
                })
                .collect(),
        };
        crate::TensorData::from_scalars(shape, self.dtype, scalars)
    }
    pub fn zeroed(dtype: DType, elements: usize, mutable: bool) -> Self {
        Self {
            dtype,
            elements,
            mutable,
            bytes: vec![0; dtype.itemsize().saturating_mul(elements)],
        }
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
    fn validate(&self, want: &BufferAbi) -> Result<(), JitError> {
        if self.dtype != want.dtype || self.elements != want.elements {
            return Err(JitError::InvalidBuffer(format!(
                "buffer {} expects {:?}[{}], got {:?}[{}]",
                want.id, want.dtype, want.elements, self.dtype, self.elements
            )));
        }
        if want.mutable && !self.mutable {
            return Err(JitError::InvalidBuffer(format!(
                "buffer {} is output but is immutable",
                want.id
            )));
        }
        if self.bytes.len()
            != want
                .elements
                .checked_mul(want.dtype.itemsize())
                .ok_or_else(|| JitError::InvalidBuffer("byte length overflow".into()))?
        {
            return Err(JitError::InvalidBuffer("wrong byte length".into()));
        }
        if !self.bytes.is_empty()
            && (self.bytes.as_ptr() as usize) % want.dtype.itemsize().max(1) != 0
        {
            return Err(JitError::InvalidBuffer("unaligned buffer".into()));
        }
        Ok(())
    }
}

pub struct CpuJit;
impl CpuJit {
    /// Checks a schedule's immutable input order against this rendered pointer
    /// ABI without changing the `void **buffers` calling convention.
    pub fn validate_schedule_bindings(
        rendered: &RenderedC,
        bindings: &[crate::ScheduleInputBinding],
    ) -> Result<(), JitError> {
        for (index, binding) in bindings.iter().enumerate() {
            if binding.abi_index != index {
                return Err(JitError::InvalidBuffer(
                    "non-contiguous schedule ABI index".into(),
                ));
            }
            let want =
                rendered.abi.buffers.get(index).ok_or_else(|| {
                    JitError::InvalidBuffer("schedule binding exceeds ABI".into())
                })?;
            if want.id != binding.desc.id
                || want.dtype != binding.desc.dtype
                || want.elements.checked_mul(want.dtype.itemsize()) != Some(binding.desc.bytes)
                || want.mutable
            {
                return Err(JitError::InvalidBuffer(format!(
                    "schedule binding {index} mismatches native ABI"
                )));
            }
        }
        Ok(())
    }
    pub fn render(kernel: &UOp) -> Result<RenderedC, JitError> {
        render(kernel)
    }
    pub fn compile(kernel: &UOp) -> Result<JitKernel, JitError> {
        let rendered = render(kernel)?;
        JitKernel::load(&rendered)
    }
    pub fn vector_plan(kernel: &UOp) -> Result<VectorPlan, JitError> {
        vector_plan(kernel)
    }
    pub fn linearize(kernel: &UOp) -> Result<crate::LinearKernel, JitError> {
        crate::LinearKernel::from_uop(kernel)
            .map_err(|error| JitError::Unsupported(error.to_string()))
    }
    pub fn render_vectorized(kernel: &UOp) -> Result<RenderedC, JitError> {
        render_with_policy(kernel, true)
    }
    pub fn compile_vectorized(kernel: &UOp) -> Result<JitKernel, JitError> {
        let rendered = render_with_policy(kernel, true)?;
        JitKernel::load(&rendered)
    }
    /// Validates a complete symbolic environment at the graph/JIT allocation
    /// boundary. The supplied UOp must already have been lowered from graphs
    /// built with these concrete shapes (`Graph::input_symbolic`); this API
    /// deliberately does not create runtime-polymorphic native kernels.
    pub fn compile_specialized(
        kernel: &UOp,
        shapes: &[SymbolicShape],
        bindings: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<JitKernel, JitError> {
        let concrete = specialize_shapes(shapes, bindings)?;
        let mut rendered = render(kernel)?;
        for shape in &concrete {
            let elements = shape.numel().map_err(|_| {
                JitError::Symbolic("specialized shape element count overflows".into())
            })?;
            if !rendered
                .abi
                .buffers
                .iter()
                .any(|buffer| buffer.elements == elements)
            {
                return Err(JitError::Symbolic(format!(
                    "specialized shape {shape} does not match any kernel buffer domain"
                )));
            }
        }
        let symbolic = format!("{shapes:?}{bindings:?}{concrete:?}");
        rendered.cache_key = key(&(rendered.cache_key + &symbolic));
        JitKernel::load(&rendered)
    }
}

fn specialize_shapes(
    shapes: &[SymbolicShape],
    bindings: &BTreeMap<SymbolicVar, i64>,
) -> Result<Vec<crate::Shape>, JitError> {
    let used = shapes
        .iter()
        .flat_map(|shape| shape.dims().iter())
        .flat_map(|dim| dim.expression().variables())
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(extra) = bindings.keys().find(|v| !used.contains(*v)) {
        return Err(JitError::Symbolic(format!(
            "extra binding {}#{}",
            extra.name(),
            extra.id()
        )));
    }
    for variable in &used {
        let value = bindings.get(variable).ok_or_else(|| {
            JitError::Symbolic(format!(
                "missing binding {}#{}",
                variable.name(),
                variable.id()
            ))
        })?;
        let (min, max) = variable.bounds();
        if *value < min || *value > max {
            return Err(JitError::Symbolic(format!(
                "binding {}={} outside [{min}, {max}]",
                variable.name(),
                value
            )));
        }
    }
    shapes
        .iter()
        .map(|shape| {
            let projected = shape
                .dims()
                .iter()
                .flat_map(|dim| dim.expression().variables())
                .filter_map(|var| bindings.get(&var).map(|value| (var, *value)))
                .collect();
            shape
                .bind(&projected)
                .map_err(|e| JitError::Symbolic(e.to_string()))
        })
        .collect()
}

#[derive(Clone)]
pub struct JitKernel {
    abi: KernelAbi,
    _library: Arc<Library>,
    call: unsafe extern "C" fn(*mut *mut c_void, *const i64, *mut u64) -> c_int,
}
impl JitKernel {
    fn load(r: &RenderedC) -> Result<Self, JitError> {
        let path = compile_cached(r)?;
        let lib = Arc::new(Library::open(&path)?);
        let call = unsafe { lib.symbol(b"rustgrad_kernel\0")? };
        Ok(Self {
            abi: r.abi.clone(),
            _library: lib,
            call,
        })
    }
    pub fn abi(&self) -> &KernelAbi {
        &self.abi
    }
    /// All raw pointers are derived from the supplied `JitBuffer`s and the
    /// vectors remain borrowed for the entire call. The ABI has no retained
    /// pointers, so it is safe to release them once this method returns.
    pub fn call(&self, buffers: &mut [JitBuffer], symbols: &[i64]) -> Result<(), JitError> {
        if buffers.len() != self.abi.buffers.len() {
            return Err(JitError::InvalidBuffer(format!(
                "expected {} buffers, got {}",
                self.abi.buffers.len(),
                buffers.len()
            )));
        }
        if symbols.len() != self.abi.symbol_count {
            return Err(JitError::InvalidBuffer(format!(
                "expected {} symbols, got {}",
                self.abi.symbol_count,
                symbols.len()
            )));
        }
        for (b, w) in buffers.iter().zip(&self.abi.buffers) {
            b.validate(w)?;
        }
        let mut ptrs: Vec<*mut c_void> = buffers
            .iter_mut()
            .map(|b| b.bytes.as_mut_ptr().cast())
            .collect();
        let mut failure = [u64::MAX, 0];
        let status =
            unsafe { (self.call)(ptrs.as_mut_ptr(), symbols.as_ptr(), failure.as_mut_ptr()) };
        match status {
            0 => {}
            1 => {
                return Err(JitError::DivisionByZero {
                    index: failure[0] as usize,
                });
            }
            2 => {
                return Err(JitError::InvalidShift {
                    index: failure[0] as usize,
                });
            }
            3 => {
                return Err(JitError::IndexOutOfBounds {
                    index: failure[0] as usize,
                });
            }
            _ => return Err(JitError::Loader(format!("unknown native status {status}"))),
        }
        Ok(())
    }
}

fn render(root: &UOp) -> Result<RenderedC, JitError> {
    render_with_policy(root, false)
}
fn vector_plan(root: &UOp) -> Result<VectorPlan, JitError> {
    if matches!(root.kind(), UOpKind::Matmul | UOpKind::Movement) {
        return Ok(VectorPlan {
            lanes: 1,
            enabled: false,
            reason: "static matmul uses a scalar contraction loop".into(),
        });
    }
    let linear = crate::LinearKernel::from_uop(root)
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    linear
        .validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    Ok(VectorPlan {
        lanes: linear.lanes,
        enabled: linear.enabled,
        reason: linear.reason,
    })
}
fn render_with_policy(root: &UOp, request_vector: bool) -> Result<RenderedC, JitError> {
    if matches!(root.kind(), UOpKind::Matmul)
        && let Some(plan) = root.arg().matmul_plan()
    {
        return render_matmul(plan);
    }
    if let (UOpKind::Movement, UArg::Movement(plan)) = (root.kind(), root.arg()) {
        return render_movement(plan);
    }
    let nodes = root
        .topological()
        .map_err(|e| JitError::Unsupported(e.to_string()))?;
    let store = root
        .sources()
        .iter()
        .find(|x| matches!(x.kind(), UOpKind::Store))
        .ok_or_else(|| JitError::Unsupported("Sink without Store".into()))?;
    let out_index = store
        .sources()
        .first()
        .ok_or_else(|| JitError::Unsupported("Store without index".into()))?;
    let UArg::BufferIndex {
        buffer: out_id,
        elements: extent,
        ..
    } = out_index.arg()
    else {
        return Err(JitError::Unsupported("Store needs BufferIndex".into()));
    };
    // Pointer ABI is first-use Load order, then its output. Do not let buffer
    // IDs redefine an expression's operand order.
    let mut buffers = Vec::<BufferAbi>::new();
    let mut seen = BTreeMap::<u64, usize>::new();
    for n in &nodes {
        if !matches!(n.kind(), UOpKind::Load) {
            continue;
        }
        let Some(index) = n.sources().first() else {
            return Err(JitError::Unsupported("load without index".into()));
        };
        let (buffer, elements) = match index.arg() {
            UArg::BufferIndex {
                buffer, elements, ..
            } => (*buffer, *elements),
            UArg::ViewBufferIndex { buffer, view, .. } => (
                *buffer,
                view.source_shape
                    .numel()
                    .map_err(|_| JitError::Unsupported("view size".into()))?,
            ),
            _ => return Err(JitError::Unsupported("load index".into())),
        };
        if seen.contains_key(&buffer) {
            continue;
        }
        let ty = n
            .ty()
            .ok_or_else(|| JitError::Unsupported("untyped load".into()))?
            .scalar;
        seen.insert(buffer, buffers.len());
        buffers.push(BufferAbi {
            id: buffer,
            dtype: ty,
            elements,
            mutable: false,
        });
    }
    if !seen.contains_key(out_id) {
        let ty = out_index
            .ty()
            .ok_or_else(|| JitError::Unsupported("untyped output index".into()))?
            .scalar;
        seen.insert(*out_id, buffers.len());
        buffers.push(BufferAbi {
            id: *out_id,
            dtype: ty,
            elements: *extent,
            mutable: true,
        });
    } else if let Some(index) = seen.get(out_id) {
        buffers[*index].mutable = true;
    }
    let abi = KernelAbi {
        version: ABI_VERSION,
        buffers,
        symbol_count: 0,
    };
    let mut ids = BTreeMap::new();
    for (i, b) in abi.buffers.iter().enumerate() {
        ids.insert(b.id, i);
    }
    let out_id = *out_id;
    let extent = *extent;
    let out = abi.buffers.iter().find(|b| b.id == out_id).unwrap();
    let (plan, linear_key, b1_program) = if request_vector {
        let linear = CpuJit::linearize(root)?;
        linear
            .validate()
            .map_err(|error| JitError::Unsupported(error.to_string()))?;
        let memory_spaces = crate::MemorySpacePlan::from_linear(&linear)
            .map_err(|error| JitError::Unsupported(error.to_string()))?;
        let vector_program = crate::VectorProgram::from_linear(&linear, &memory_spaces)
            .map_err(|error| JitError::Unsupported(error.to_string()))?;
        let b1 = vector_program
            .b1_eligibility()
            .ok()
            .map(|_| vector_program.clone());
        (
            VectorPlan {
                lanes: linear.lanes,
                enabled: linear.enabled,
                reason: linear.reason,
            },
            Some((
                linear.cache_key,
                linear.program.instructions.len(),
                linear.program.peak_scalar,
                linear.program.peak_vector,
                vector_program.cache_key,
            )),
            b1,
        )
    } else {
        (
            VectorPlan {
                lanes: 1,
                enabled: false,
                reason: "disabled".into(),
            },
            None,
            None,
        )
    };
    if let Some(program) = b1_program {
        return render_vector_program(&program, &abi, &ids, extent);
    }
    let mut lines = vec![
        "#include <stdint.h>".into(),
        "#include <stddef.h>".into(),
        "#include <math.h>".into(),
        "#include <limits.h>".into(),
        format!("/* rustgrad C11 ABI v2; vector lanes={} ({}) linear={linear_key:?} */", plan.lanes, plan.reason),
        "static float rg_f16_to_f32(uint16_t h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,o;if(!e)o=m? s|((uint32_t)(113-__builtin_clz(m))<<23)|((uint32_t)(m<<(126-__builtin_clz(m)))<<13):s;else o=e==31?s|0x7f800000|(m<<13):s|((e+112)<<23)|(m<<13);union{uint32_t u;float f;}v={o};return v.f;}".into(),
        "static uint16_t rg_f32_to_f16(float x){union{float f;uint32_t u;}v={x};uint32_t b=v.u,s=(b>>16)&0x8000,e=(b>>23)&255,m=b&0x7fffff;if(e==255)return(uint16_t)(s|0x7c00|(m?((m>>13)|1):0));int q=(int)e-112;if(q<=0){if(q<-10)return(uint16_t)s;uint32_t z=m|0x800000,sh=(uint32_t)(14-q),r=z>>sh,rem=z&((1u<<sh)-1),half=1u<<(sh-1);return(uint16_t)(s+r+(rem>half||(rem==half&&(r&1))));}if(q>=31)return(uint16_t)(s|0x7c00);uint32_t r=m>>13,rem=m&0x1fff; r+=rem>0x1000||(rem==0x1000&&(r&1));if(r==0x400){if(q==30)return(uint16_t)(s|0x7c00);q++;r=0;}return(uint16_t)(s|((uint32_t)q<<10)|r);}".into(),
        "static float rg_bf16_to_f32(uint16_t b){union{uint32_t u;float f;}v={(uint32_t)b<<16};return v.f;}".into(),
        "static uint16_t rg_f32_to_bf16(float x){union{float f;uint32_t u;}v={x};return(uint16_t)((v.u+0x7fff+((v.u>>16)&1))>>16);}".into(),
        "static int64_t rg_sdiv(int64_t a,int64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return(a==INT64_MIN&&b==-1)?INT64_MIN:a/b;}".into(),
        "static uint64_t rg_udiv(uint64_t a,uint64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return a/b;}".into(),
        "static int64_t rg_smod(int64_t a,int64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return(a==INT64_MIN&&b==-1)?0:a%b;}".into(),
        "static uint64_t rg_umod(uint64_t a,uint64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return a%b;}".into(),
        "static uint64_t rg_shl(uint64_t a,int64_t b,unsigned bits,uint64_t i,uint64_t *f){if(b<0||(uint64_t)b>=bits){if(!f[1]){f[0]=i;f[1]=2;}return 0;}return a<<b;}".into(),
        "static uint64_t rg_shr(uint64_t a,int64_t b,unsigned bits,uint64_t i,uint64_t *f){if(b<0||(uint64_t)b>=bits){if(!f[1]){f[0]=i;f[1]=2;}return 0;}return a>>b;}".into(),
        "int rustgrad_kernel(void **buffers, const int64_t *symbols, uint64_t *failure) { (void)symbols; failure[0]=UINT64_MAX; failure[1]=0;".into(),
        if plan.enabled { format!("  for (size_t rg_base = 0; rg_base + {}u <= {extent}u; rg_base += {}u) {{ for (size_t rg_lane = 0; rg_lane < {}u; ++rg_lane) {{ size_t rg_i = rg_base + rg_lane;", plan.lanes, plan.lanes, plan.lanes) } else { format!("  for (size_t rg_i = 0; rg_i < {extent}u; ++rg_i) {{") },
    ];
    if let Some(rendered) = render_reduction(store, &abi, &ids, out, &mut lines)? {
        let source = rendered;
        let cache_key = key(&(RENDERER_VERSION.to_owned()
            + std::env::consts::ARCH
            + std::env::consts::OS
            + &source));
        return Ok(RenderedC {
            source,
            source_map: BTreeMap::new(),
            abi,
            cache_key,
        });
    }
    let mut map = BTreeMap::new();
    let value = emit(
        store
            .sources()
            .get(1)
            .ok_or_else(|| JitError::Unsupported("Store missing value".into()))?,
        &ids,
        &mut map,
        &mut lines,
    )?;
    let store_value: String = match out.dtype {
        DType::F16 => format!("rg_f32_to_f16({value})"),
        DType::BF16 => format!("rg_f32_to_bf16({value})"),
        _ => value,
    };
    lines.push(format!(
        "    (({}*)buffers[{}])[rg_i] = ({});",
        ctype(out.dtype),
        ids[&out_id],
        store_value
    ));
    if plan.enabled {
        lines.push("  }}".into());
        lines.push(format!(
            "  for (size_t rg_i = ({extent}u / {}u) * {}u; rg_i < {extent}u; ++rg_i) {{",
            plan.lanes, plan.lanes
        ));
        lines.push(format!(
            "    (({}*)buffers[{}])[rg_i] = ({});",
            ctype(out.dtype),
            ids[&out_id],
            store_value
        ));
        lines.push("  }".into());
    } else {
        lines.push("  }".into());
    }
    lines.push("  return failure[1] ? (int)failure[1] : 0;".into());
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = key(&(RENDERER_VERSION.to_owned()
        + std::env::consts::ARCH
        + std::env::consts::OS
        + &source));
    Ok(RenderedC {
        source,
        source_map: map,
        abi,
        cache_key,
    })
}

fn render_matmul(plan: &crate::MatmulKernelPlan) -> Result<RenderedC, JitError> {
    plan.validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    if !matches!(
        (plan.lhs_dtype, plan.rhs_dtype, plan.dtype),
        (DType::F32, DType::F32, DType::F32) | (DType::F64, DType::F64, DType::F64)
    ) {
        return Err(JitError::Unsupported(
            "static matmul CPU JIT supports only homogeneous F32 or F64".into(),
        ));
    }
    let elements = |shape: &crate::Shape| {
        shape
            .numel()
            .map_err(|_| JitError::Unsupported("matmul shape overflow".into()))
    };
    let mut buffers = Vec::with_capacity(3);
    for (id, dtype, shape) in [
        (plan.lhs.index() as u64, plan.lhs_dtype, &plan.lhs_shape),
        (plan.rhs.index() as u64, plan.rhs_dtype, &plan.rhs_shape),
    ] {
        if buffers.iter().any(|buffer: &BufferAbi| buffer.id == id) {
            continue;
        }
        buffers.push(BufferAbi {
            id,
            dtype,
            elements: elements(shape)?,
            mutable: false,
        });
    }
    buffers.push(BufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        elements: elements(&plan.output_shape)?,
        mutable: true,
    });
    let abi = KernelAbi {
        version: ABI_VERSION,
        buffers,
        symbol_count: 0,
    };
    let ids = abi
        .buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| (buffer.id, index))
        .collect::<BTreeMap<_, _>>();
    let batch_offset = |shape: &crate::Shape, vector: bool| {
        if vector || plan.batch_shape.contains(&0) {
            return "0u".to_owned();
        }
        let input = &shape.dims()[..shape.rank() - 2];
        let pad = plan.batch_shape.len() - input.len();
        let terms = input
            .iter()
            .enumerate()
            .filter(|(_, dim)| **dim != 1)
            .map(|(axis, _)| {
                let normalized_axis = pad + axis;
                let normalized_stride = plan.batch_shape[normalized_axis + 1..]
                    .iter()
                    .product::<usize>();
                let input_stride = input[axis + 1..].iter().product::<usize>();
                format!(
                    "((rg_batch / {normalized_stride}u) % {}u) * {input_stride}u",
                    plan.batch_shape[normalized_axis]
                )
            })
            .collect::<Vec<_>>();
        if terms.is_empty() {
            "0u".into()
        } else {
            terms.join(" + ")
        }
    };
    let lhs_batch = batch_offset(&plan.lhs_shape, plan.lhs_vector);
    let rhs_batch = batch_offset(&plan.rhs_shape, plan.rhs_vector);
    let lhs_offset = if plan.lhs_vector {
        "rg_k".into()
    } else {
        format!("((rg_lbatch * {}u + rg_row) * {}u + rg_k)", plan.m, plan.k)
    };
    let rhs_offset = if plan.rhs_vector {
        "rg_k".into()
    } else {
        format!("((rg_rbatch * {}u + rg_k) * {}u + rg_col)", plan.k, plan.n)
    };
    let storage = if plan.dtype == DType::F32 {
        "float"
    } else {
        "double"
    };
    let mut lines = vec![
        "#include <stdint.h>".into(),
        "#include <stddef.h>".into(),
        format!(
            "/* {RENDERER_VERSION} matmul plan={} M={} N={} K={} */",
            plan.cache_key, plan.m, plan.n, plan.k
        ),
        "int rustgrad_kernel(void **buffers, const int64_t *symbols, uint64_t *failure) { (void)symbols; failure[0]=UINT64_MAX; failure[1]=0;".into(),
        format!("  const {storage} *rg_lhs = (const {storage}*)buffers[{}];", ids[&(plan.lhs.index() as u64)]),
        format!("  const {storage} *rg_rhs = (const {storage}*)buffers[{}];", ids[&(plan.rhs.index() as u64)]),
        format!("  {storage} *rg_out = ({storage}*)buffers[{}];", ids[&(plan.output.index() as u64)]),
        format!("  for (size_t rg_i=0; rg_i<{}u; ++rg_i) {{", elements(&plan.output_shape)?),
        "    size_t rg_q=rg_i, rg_col=0, rg_row=0;".into(),
    ];
    if !plan.rhs_vector && plan.n != 0 {
        lines.push(format!("    rg_col=rg_q%{}u; rg_q/={}u;", plan.n, plan.n));
    }
    if !plan.lhs_vector && plan.m != 0 {
        lines.push(format!("    rg_row=rg_q%{}u; rg_q/={}u;", plan.m, plan.m));
    }
    lines.extend([
        "    size_t rg_batch=rg_q;".into(),
        format!("    size_t rg_lbatch={lhs_batch};"),
        format!("    size_t rg_rbatch={rhs_batch};"),
        "    double rg_acc=0.0;".into(),
        format!("    for (size_t rg_k=0; rg_k<{}u; ++rg_k) rg_acc += (double)rg_lhs[{lhs_offset}] * (double)rg_rhs[{rhs_offset}];", plan.k),
        format!("    rg_out[rg_i]=({storage})rg_acc;"),
        "  }".into(),
        "  return 0;".into(),
        "}".into(),
    ]);
    let source = lines.join("\n") + "\n";
    let cache_key = key(&(RENDERER_VERSION.to_owned()
        + std::env::consts::ARCH
        + std::env::consts::OS
        + &plan.cache_key.to_string()
        + &source));
    Ok(RenderedC {
        source,
        source_map: BTreeMap::from([(0, 1)]),
        abi,
        cache_key,
    })
}

fn render_movement(plan: &crate::MovementKernelPlan) -> Result<RenderedC, JitError> {
    plan.validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let elements = |shape: &crate::Shape| {
        shape
            .numel()
            .map_err(|_| JitError::Unsupported("movement shape overflow".into()))
    };
    let mut buffers = Vec::new();
    for operand in plan.input_operands() {
        let id = operand.node.index() as u64;
        if buffers.iter().any(|buffer: &BufferAbi| buffer.id == id) {
            continue;
        }
        buffers.push(BufferAbi {
            id,
            dtype: operand.dtype,
            elements: elements(&operand.shape)?,
            mutable: false,
        });
    }
    buffers.push(BufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        elements: elements(&plan.output_shape)?,
        mutable: true,
    });
    let abi = KernelAbi {
        version: ABI_VERSION,
        buffers,
        symbol_count: 0,
    };
    let ids = abi
        .buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| (buffer.id, index))
        .collect::<BTreeMap<_, _>>();
    let output_slot = ids[&(plan.output.index() as u64)];
    let mut lines = vec![
        "#include <stdint.h>".into(),
        "#include <stddef.h>".into(),
        "#include <string.h>".into(),
        format!("/* {RENDERER_VERSION} movement plan={} */", plan.cache_key),
        "int rustgrad_kernel(void **buffers, const int64_t *symbols, uint64_t *failure) { (void)symbols; failure[0]=UINT64_MAX; failure[1]=0;".into(),
    ];
    let output_ty = ctype(plan.dtype);
    match &plan.kind {
        crate::MovementKernelKind::Concat { inputs, axis } => {
            let output_len = elements(&plan.output_shape)?;
            let inner = plan.output_shape.dims()[axis + 1..]
                .iter()
                .product::<usize>();
            let output_axis = plan.output_shape.dims()[*axis];
            if output_len == 0 {
                lines.push("  /* empty concat domain */".into());
            } else {
                lines.push(format!(
                "  for (size_t rg_i=0; rg_i<{}u; ++rg_i) {{ size_t rg_axis=(rg_i/{}u)%{}u, rg_outer=rg_i/({}u*{}u), rg_inner=rg_i%{}u;",
                output_len, inner, output_axis, output_axis, inner, inner
            ));
                let mut start = 0usize;
                for (position, input) in inputs.iter().enumerate() {
                    let width = input.shape.dims()[*axis];
                    let prefix = if position == 0 { "if" } else { "else if" };
                    lines.push(format!(
                    "    {prefix} (rg_axis < {}u) (({output_ty}*)buffers[{output_slot}])[rg_i] = ((const {output_ty}*)buffers[{}])[(rg_outer*{}u+(rg_axis-{}u))*{}u+rg_inner];",
                    start + width,
                    ids[&(input.node.index() as u64)],
                    width,
                    start,
                    inner,
                ));
                    start += width;
                }
                lines.push("  }".into());
            }
        }
        crate::MovementKernelKind::Gather { input, index, axis } => {
            let index_len = elements(&index.shape)?;
            if index_len == 0 {
                lines.push("  /* empty gather domain */".into());
            } else {
                lines.push(format!(
                    "  for (size_t rg_i=0; rg_i<{}u; ++rg_i) {{",
                    index_len
                ));
                let selected = index_expression(index, &ids, "rg_i");
                lines.push(format!("    int64_t rg_selected=(int64_t)({selected});"));
                lines.push(format!(
                "    if (rg_selected < 0 || (uint64_t)rg_selected >= {}u) {{ failure[0]=rg_i; failure[1]=3; continue; }}",
                input.shape.dims()[*axis]
            ));
                let source =
                    indexed_offset(&input.shape, &index.shape, *axis, "rg_selected", "rg_i");
                lines.push(format!(
                "    (({output_ty}*)buffers[{output_slot}])[rg_i] = ((const {output_ty}*)buffers[{}])[{source}];",
                ids[&(input.node.index() as u64)]
            ));
                lines.push("  }".into());
            }
        }
        crate::MovementKernelKind::Scatter {
            base,
            index,
            updates,
            axis,
            add,
        } => {
            lines.push(format!(
                "  memcpy(buffers[{output_slot}], buffers[{}], {}u);",
                ids[&(base.node.index() as u64)],
                elements(&base.shape)? * plan.dtype.itemsize()
            ));
            let index_len = elements(&index.shape)?;
            if index_len != 0 {
                lines.push(format!(
                    "  for (size_t rg_i=0; rg_i<{index_len}u; ++rg_i) {{"
                ));
                let selected = index_expression(index, &ids, "rg_i");
                lines.push(format!("    int64_t rg_selected=(int64_t)({selected});"));
                lines.push(format!(
                "    if (rg_selected < 0 || (uint64_t)rg_selected >= {}u) {{ failure[0]=rg_i; failure[1]=3; continue; }}",
                base.shape.dims()[*axis]
            ));
                let destination =
                    indexed_offset(&base.shape, &index.shape, *axis, "rg_selected", "rg_i");
                let update = coordinate_offset(&updates.shape, &index.shape, "rg_i");
                let operator = if *add { "+=" } else { "=" };
                lines.push(format!(
                "    (({output_ty}*)buffers[{output_slot}])[{destination}] {operator} ((const {output_ty}*)buffers[{}])[{update}];",
                ids[&(updates.node.index() as u64)]
            ));
                lines.push("  }".into());
            }
        }
    }
    lines.push("  return failure[1] ? (int)failure[1] : 0;".into());
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key =
        key(&(RENDERER_VERSION.to_owned() + "-movement-" + &plan.cache_key.to_string() + &source));
    Ok(RenderedC {
        source,
        source_map: BTreeMap::from([(0, 1)]),
        abi,
        cache_key,
    })
}

fn index_expression(
    index: &crate::MovementOperand,
    ids: &BTreeMap<u64, usize>,
    linear: &str,
) -> String {
    format!(
        "((const {}*)buffers[{}])[{}]",
        ctype(index.dtype),
        ids[&(index.node.index() as u64)],
        linear
    )
}

fn indexed_offset(
    target: &crate::Shape,
    coordinates: &crate::Shape,
    axis: usize,
    selected: &str,
    linear: &str,
) -> String {
    coordinate_terms(target, coordinates, linear)
        .into_iter()
        .enumerate()
        .map(|(dimension, term)| {
            let coordinate = if dimension == axis {
                selected.to_owned()
            } else {
                term
            };
            let stride = target.dims()[dimension + 1..].iter().product::<usize>();
            format!("(({coordinate})*{stride}u)")
        })
        .collect::<Vec<_>>()
        .join("+")
}

fn coordinate_offset(target: &crate::Shape, coordinates: &crate::Shape, linear: &str) -> String {
    coordinate_terms(target, coordinates, linear)
        .into_iter()
        .enumerate()
        .map(|(dimension, coordinate)| {
            let stride = target.dims()[dimension + 1..].iter().product::<usize>();
            format!("(({coordinate})*{stride}u)")
        })
        .collect::<Vec<_>>()
        .join("+")
}

fn coordinate_terms(
    target: &crate::Shape,
    coordinates: &crate::Shape,
    linear: &str,
) -> Vec<String> {
    debug_assert_eq!(target.rank(), coordinates.rank());
    coordinates
        .dims()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| {
            let divisor = coordinates.dims()[axis + 1..].iter().product::<usize>();
            format!("((({linear})/{divisor}u)%{dimension}u)")
        })
        .collect()
}
/// B1/B2 consume only the physical VectorProgram.  It intentionally does not
/// inspect the retained UOp DAG or call the legacy expression renderer.
fn render_vector_program(
    program: &crate::VectorProgram,
    abi: &KernelAbi,
    ids: &BTreeMap<u64, usize>,
    elements: usize,
) -> Result<RenderedC, JitError> {
    program
        .b1_eligibility()
        .map_err(|e| JitError::Unsupported(e.to_string()))?;
    let lanes = usize::from(program.lanes);
    if lanes == 0 || program.main_elements > elements || program.main_elements % lanes != 0 {
        return Err(JitError::Unsupported(
            "invalid portable lane/tail control".into(),
        ));
    }
    let mut lines = vec![
        "#include <stdint.h>".into(), "#include <stddef.h>".into(), "#include <math.h>".into(), "#include <string.h>".into(), "#include <limits.h>".into(),
        format!("/* rustgrad B2 VectorProgram key={} lanes={} */", program.cache_key, lanes),
        "static int8_t rg_i8(uint8_t x){int8_t r;memcpy(&r,&x,1);return r;} static int16_t rg_i16(uint16_t x){int16_t r;memcpy(&r,&x,2);return r;} static int32_t rg_i32(uint32_t x){int32_t r;memcpy(&r,&x,4);return r;} static int64_t rg_i64(uint64_t x){int64_t r;memcpy(&r,&x,8);return r;}".into(),
        "static void rg_fail(uint64_t*f,uint64_t i,uint64_t c){if(!f[1]||i<f[0]){f[0]=i;f[1]=c;}}".into(),
        "static uint64_t rg_udiv(uint64_t a,uint64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}return a/b;} static uint64_t rg_umod(uint64_t a,uint64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}return a%b;}".into(),
        "static int64_t rg_sdiv(int64_t a,int64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}if(a==INT64_MIN&&b==-1)return INT64_MIN;return a/b;} static int64_t rg_sfdiv(int64_t a,int64_t b,uint64_t i,uint64_t*f){int64_t q=rg_sdiv(a,b,i,f),r;if(!b|| (a==INT64_MIN&&b==-1))return q;r=a%b;return r<0?q-(b>0?1:-1):q;} static int64_t rg_srem(int64_t a,int64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}return(a==INT64_MIN&&b==-1)?0:a%b;} static int64_t rg_smod(int64_t a,int64_t b,uint64_t i,uint64_t*f){int64_t r=rg_srem(a,b,i,f);if(!b||r>=0)return r;return b>0?r+b:r-b;}".into(),
        "static uint64_t rg_shl(uint64_t a,int64_t b,unsigned n,uint64_t i,uint64_t*f){if(b<0||(uint64_t)b>=n){rg_fail(f,i,2);return 0;}return a<<(unsigned)b;} static uint64_t rg_ushr(uint64_t a,int64_t b,unsigned n,uint64_t i,uint64_t*f){if(b<0||(uint64_t)b>=n){rg_fail(f,i,2);return 0;}return a>>(unsigned)b;} static uint64_t rg_sshr(uint64_t a,int64_t b,unsigned n,uint64_t i,uint64_t*f){uint64_t mask=n==64?UINT64_MAX:((UINT64_C(1)<<n)-1),r;if(b<0||(uint64_t)b>=n){rg_fail(f,i,2);return 0;}r=(a&mask)>>(unsigned)b;if(b&&((a>>(n-1))&1))r|=mask^(mask>>((unsigned)b));return r;}".into(),
        "static float rg_f16_to_f32(uint16_t h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,o;if(!e)o=m? s|((uint32_t)(113-__builtin_clz(m))<<23)|((uint32_t)(m<<(126-__builtin_clz(m)))<<13):s;else o=e==31?s|0x7f800000|(m<<13):s|((e+112)<<23)|(m<<13);union{uint32_t u;float f;}v={o};return v.f;} static uint16_t rg_f32_to_f16(float x){union{float f;uint32_t u;}v={x};uint32_t b=v.u,s=(b>>16)&0x8000,e=(b>>23)&255,m=b&0x7fffff;if(e==255)return(uint16_t)(s|0x7c00|(m?((m>>13)|1):0));int q=(int)e-112;if(q<=0){if(q<-10)return(uint16_t)s;uint32_t z=m|0x800000,sh=(uint32_t)(14-q),r=z>>sh,rem=z&((1u<<sh)-1),half=1u<<(sh-1);return(uint16_t)(s+r+(rem>half||(rem==half&&(r&1))));}if(q>=31)return(uint16_t)(s|0x7c00);uint32_t r=m>>13,rem=m&0x1fff;r+=rem>0x1000||(rem==0x1000&&(r&1));if(r==0x400){if(q==30)return(uint16_t)(s|0x7c00);q++;r=0;}return(uint16_t)(s|((uint32_t)q<<10)|r);} static float rg_bf16_to_f32(uint16_t b){union{uint32_t u;float f;}v={(uint32_t)b<<16};return v.f;} static uint16_t rg_f32_to_bf16(float x){union{float f;uint32_t u;}v={x};return(uint16_t)((v.u+0x7fff+((v.u>>16)&1))>>16);}".into(),
        "int rustgrad_kernel(void **buffers, const int64_t *symbols, uint64_t *failure) { (void)symbols; failure[0]=UINT64_MAX; failure[1]=0;".into(),
        format!("  for (size_t rg_base=0; rg_base<{}u; rg_base+={}u) {{", program.main_elements, lanes),
    ];
    emit_vector_insts(&mut lines, program, abi, ids, "rg_base", lanes)?;
    lines.push("  }".into());
    if program.tail_elements != 0 {
        lines.push(format!(
            "  for (size_t rg_base={}u; rg_base<{}u; rg_base+={}u) {{",
            program.main_elements, elements, lanes
        ));
        emit_vector_insts(
            &mut lines,
            program,
            abi,
            ids,
            "rg_base",
            program.tail_elements,
        )?;
        lines.push("  }".into());
    }
    lines.push("  return failure[1] ? (int)failure[1] : 0;".into());
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key =
        key(&(RENDERER_VERSION.to_owned() + "-b1-" + &program.cache_key.to_string() + &source));
    Ok(RenderedC {
        source,
        source_map: program
            .instructions
            .iter()
            .enumerate()
            .map(|(i, x)| (i, x.index as usize))
            .collect(),
        abi: abi.clone(),
        cache_key,
    })
}
fn vector_reg(
    op: &crate::VectorOperand,
    names: &BTreeMap<(u32, DType), String>,
) -> Result<String, JitError> {
    match op {
        crate::VectorOperand::Register {
            physical, dtype, ..
        } => names.get(&(*physical, *dtype)).cloned().ok_or_else(|| {
            JitError::Unsupported(format!("B1 use before physical register r{physical}"))
        }),
        crate::VectorOperand::Global { .. } => {
            Err(JitError::Unsupported("global operand used as value".into()))
        }
    }
}
fn vector_dtype(op: &crate::VectorOperand) -> Result<DType, JitError> {
    match op {
        crate::VectorOperand::Register { dtype, .. } => Ok(*dtype),
        crate::VectorOperand::Global { .. } => {
            Err(JitError::Unsupported("global operand type".into()))
        }
    }
}
fn unsigned_ctype(dtype: DType) -> Result<&'static str, JitError> {
    match dtype {
        DType::I8 | DType::U8 | DType::Bool => Ok("uint8_t"),
        DType::I16 | DType::U16 => Ok("uint16_t"),
        DType::I32 | DType::U32 => Ok("uint32_t"),
        DType::I64 | DType::U64 => Ok("uint64_t"),
        _ => Err(JitError::Unsupported(format!(
            "not an exact dtype {dtype:?}"
        ))),
    }
}
fn wrap_expr(dtype: DType, value: String) -> Result<String, JitError> {
    let u = unsigned_ctype(dtype)?;
    Ok(match dtype {
        DType::Bool => format!("((uint8_t)(({value})!=0))"),
        DType::I8 => format!("rg_i8((uint8_t)({value}))"),
        DType::I16 => format!("rg_i16((uint16_t)({value}))"),
        DType::I32 => format!("rg_i32((uint32_t)({value}))"),
        DType::I64 => format!("rg_i64((uint64_t)({value}))"),
        _ => format!("(({u})({value}))"),
    })
}
fn vector_binary_expr(
    dtype: DType,
    op: crate::BinaryOp,
    a: &str,
    b: &str,
    index: &str,
) -> Result<String, JitError> {
    if dtype.is_float() {
        let symbol = match op {
            crate::BinaryOp::Add => "+",
            crate::BinaryOp::Sub => "-",
            crate::BinaryOp::Mul => "*",
            _ => {
                return Err(JitError::Unsupported(format!(
                    "portable float binary {op:?}"
                )));
            }
        };
        return Ok(format!("({a}[l]{symbol}{b}[l])"));
    }
    if dtype == DType::Bool {
        let expr = match op {
            crate::BinaryOp::Add => format!("({a}[l]||{b}[l])"),
            crate::BinaryOp::Sub => format!("({a}[l]!={b}[l])"),
            crate::BinaryOp::Mul | crate::BinaryOp::Div => format!("({a}[l]&&{b}[l])"),
            _ => {
                return Err(JitError::Unsupported(format!(
                    "portable bool binary {op:?}"
                )));
            }
        };
        return wrap_expr(dtype, expr);
    }
    let u = unsigned_ctype(dtype)?;
    let left = format!("({u}){a}[l]");
    let right = format!("({u}){b}[l]");
    let value = match op {
        crate::BinaryOp::Add => format!("{left}+{right}"),
        crate::BinaryOp::Sub => format!("{left}-{right}"),
        crate::BinaryOp::Mul => format!("{left}*{right}"),
        crate::BinaryOp::Div | crate::BinaryOp::TruncDiv => {
            if matches!(dtype.category(), crate::DTypeCategory::Signed) {
                format!("rg_sdiv((int64_t){a}[l],(int64_t){b}[l],{index},failure)")
            } else {
                format!("rg_udiv((uint64_t){left},(uint64_t){right},{index},failure)")
            }
        }
        crate::BinaryOp::FloorDiv => {
            if matches!(dtype.category(), crate::DTypeCategory::Signed) {
                format!("rg_sfdiv((int64_t){a}[l],(int64_t){b}[l],{index},failure)")
            } else {
                format!("rg_udiv((uint64_t){left},(uint64_t){right},{index},failure)")
            }
        }
        crate::BinaryOp::Mod | crate::BinaryOp::FMod => {
            if matches!(dtype.category(), crate::DTypeCategory::Signed) {
                let helper = if matches!(op, crate::BinaryOp::Mod) {
                    "rg_smod"
                } else {
                    "rg_srem"
                };
                format!("{helper}((int64_t){a}[l],(int64_t){b}[l],{index},failure)")
            } else {
                format!("rg_umod((uint64_t){left},(uint64_t){right},{index},failure)")
            }
        }
        crate::BinaryOp::Shl => format!(
            "rg_shl((uint64_t){left},(int64_t){b}[l],{}, {index},failure)",
            dtype.bits()
        ),
        crate::BinaryOp::Shr => {
            if matches!(dtype.category(), crate::DTypeCategory::Signed) {
                format!(
                    "rg_sshr((uint64_t){left},(int64_t){b}[l],{}, {index},failure)",
                    dtype.bits()
                )
            } else {
                format!(
                    "rg_ushr((uint64_t){left},(int64_t){b}[l],{}, {index},failure)",
                    dtype.bits()
                )
            }
        }
        _ => return Err(JitError::Unsupported(format!("portable binary {op:?}"))),
    };
    wrap_expr(dtype, value)
}
fn emit_vector_insts(
    lines: &mut Vec<String>,
    program: &crate::VectorProgram,
    abi: &KernelAbi,
    ids: &BTreeMap<u64, usize>,
    base: &str,
    active: usize,
) -> Result<(), JitError> {
    let mut names = BTreeMap::new();
    for inst in &program.instructions {
        let dst = inst
            .dst
            .as_ref()
            .map(|op| match op {
                crate::VectorOperand::Register {
                    physical, dtype, ..
                } => Ok(format!("r{}_{}_{}", physical, ctype(*dtype), inst.index)),
                crate::VectorOperand::Global { .. } => {
                    Err(JitError::Unsupported("global destination".into()))
                }
            })
            .transpose()?;
        let dst_ty = inst.dst.as_ref().map(vector_dtype).transpose()?;
        let input = |n: usize| {
            inst.inputs
                .get(n)
                .ok_or_else(|| {
                    JitError::Unsupported(format!("B1 instruction {} missing operand", inst.index))
                })
                .and_then(|op| vector_reg(op, &names))
        };
        match &inst.kind {
            crate::VectorInstKind::Splat => {
                let d = dst
                    .clone()
                    .ok_or_else(|| JitError::Unsupported("B1 splat without destination".into()))?;
                let (ty, literal) = match inst.payload.arg {
                    crate::UArg::Scalar { dtype, bits } => {
                        (dst_ty.unwrap_or(dtype), literal_expr(dtype, bits))
                    }
                    crate::UArg::Int(value) => (
                        dst_ty.ok_or_else(|| {
                            JitError::Unsupported("portable constant type".into())
                        })?,
                        format!("{value}LL"),
                    ),
                    _ => return Err(JitError::Unsupported("B1 constant payload".into())),
                };
                lines.push(format!(
                    "    {} {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]={};",
                    ctype(ty),
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    literal
                ));
            }
            crate::VectorInstKind::Address | crate::VectorInstKind::Index => {
                if let Some(d) = dst.clone() {
                    lines.push(format!(
                        "    size_t {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]={}+l;",
                        d,
                        usize::from(program.lanes),
                        active,
                        d,
                        base
                    ));
                }
            }
            crate::VectorInstKind::Load { buffer } => {
                let d = dst
                    .clone()
                    .ok_or_else(|| JitError::Unsupported("B1 load without destination".into()))?;
                let ty =
                    dst_ty.ok_or_else(|| JitError::Unsupported("portable untyped load".into()))?;
                let slot = ids
                    .get(buffer)
                    .ok_or_else(|| JitError::Unsupported("B1 load unknown buffer".into()))?;
                let scalar = abi
                    .buffers
                    .iter()
                    .find(|b| b.id == *buffer)
                    .is_some_and(|b| b.elements == 1);
                let index = if scalar {
                    "0".to_owned()
                } else {
                    format!("{base}+l")
                };
                let storage = abi
                    .buffers
                    .iter()
                    .find(|b| b.id == *buffer)
                    .ok_or_else(|| JitError::Unsupported("portable load ABI".into()))?
                    .dtype;
                let load = match storage {
                    DType::F16 => "rg_f16_to_f32",
                    DType::BF16 => "rg_bf16_to_f32",
                    _ => "",
                };
                let rhs = if load.is_empty() {
                    format!("(({}*)buffers[{}])[{}]", ctype(storage), slot, index)
                } else {
                    format!("{load}(((uint16_t*)buffers[{}])[{}])", slot, index)
                };
                lines.push(format!(
                    "    {} {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]={};",
                    ctype(ty),
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    rhs
                ));
            }
            crate::VectorInstKind::Unary => {
                let d = dst
                    .clone()
                    .ok_or_else(|| JitError::Unsupported("B1 unary destination".into()))?;
                let a = input(0)?;
                let ty =
                    dst_ty.ok_or_else(|| JitError::Unsupported("portable unary type".into()))?;
                let expr = match inst.payload.uop_kind {
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Neg) if ty.is_float() => {
                        format!("-{}[l]", a)
                    }
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Abs) if ty.is_float() => {
                        format!("fabs({}[l])", a)
                    }
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Neg) => {
                        wrap_expr(ty, format!("0-({}){}[l]", unsigned_ctype(ty)?, a))?
                    }
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Abs)
                        if matches!(ty.category(), crate::DTypeCategory::Signed) =>
                    {
                        wrap_expr(
                            ty,
                            format!(
                                "{}[l]<0 ? 0-({}){}[l] : ({}){}[l]",
                                a,
                                unsigned_ctype(ty)?,
                                a,
                                unsigned_ctype(ty)?,
                                a
                            ),
                        )?
                    }
                    _ => return Err(JitError::Unsupported("portable unary opcode".into())),
                };
                lines.push(format!(
                    "    {} {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]={};",
                    ctype(ty),
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    expr
                ));
            }
            crate::VectorInstKind::Binary => {
                let d = dst
                    .clone()
                    .ok_or_else(|| JitError::Unsupported("B1 binary destination".into()))?;
                let (a, b) = (input(0)?, input(1)?);
                let ty =
                    dst_ty.ok_or_else(|| JitError::Unsupported("portable binary type".into()))?;
                let crate::UOpKind::GraphBinary(op) = inst.payload.uop_kind else {
                    return Err(JitError::Unsupported("portable binary opcode".into()));
                };
                let expr = vector_binary_expr(ty, op, &a, &b, &format!("{base}+l"))?;
                lines.push(format!(
                    "    {} {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]={};",
                    ctype(ty),
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    expr
                ));
            }
            crate::VectorInstKind::Compare => {
                let d = dst
                    .clone()
                    .ok_or_else(|| JitError::Unsupported("B1 compare destination".into()))?;
                let (a, b) = (input(0)?, input(1)?);
                let op = match inst.payload.uop_kind {
                    crate::UOpKind::GraphCompare(crate::CompareOp::Eq) => "==",
                    crate::UOpKind::GraphCompare(crate::CompareOp::Ne) => "!=",
                    crate::UOpKind::GraphCompare(crate::CompareOp::Lt) => "<",
                    crate::UOpKind::GraphCompare(crate::CompareOp::Le) => "<=",
                    crate::UOpKind::GraphCompare(crate::CompareOp::Gt) => ">",
                    crate::UOpKind::GraphCompare(crate::CompareOp::Ge) => ">=",
                    _ => return Err(JitError::Unsupported("portable compare opcode".into())),
                };
                lines.push(format!(
                    "    uint8_t {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]=({}[l]{}{}[l]);",
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    a,
                    op,
                    b
                ));
            }
            crate::VectorInstKind::Select => {
                let d = dst
                    .clone()
                    .ok_or_else(|| JitError::Unsupported("B1 select destination".into()))?;
                let (c, a, b) = (input(0)?, input(1)?, input(2)?);
                lines.push(format!(
                    "    {} {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]={}[l]?{}[l]:{}[l];",
                    ctype(
                        dst_ty
                            .ok_or_else(|| JitError::Unsupported("portable select type".into()))?
                    ),
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    c,
                    a,
                    b
                ));
            }
            crate::VectorInstKind::Cast => {
                let d = dst
                    .clone()
                    .ok_or_else(|| JitError::Unsupported("B1 cast destination".into()))?;
                let a = input(0)?;
                let ty =
                    dst_ty.ok_or_else(|| JitError::Unsupported("portable cast type".into()))?;
                lines.push(format!(
                    "    {} {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]=({}){}[l];",
                    ctype(ty),
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    ctype(ty),
                    a
                ));
            }
            crate::VectorInstKind::Store { buffer } => {
                let value = input(1).or_else(|_| input(0))?;
                let slot = ids
                    .get(buffer)
                    .ok_or_else(|| JitError::Unsupported("B1 store unknown buffer".into()))?;
                let ty = abi
                    .buffers
                    .iter()
                    .find(|b| b.id == *buffer)
                    .ok_or_else(|| JitError::Unsupported("portable store ABI".into()))?
                    .dtype;
                let stored = match ty {
                    DType::F16 => format!("rg_f32_to_f16({value}[l])"),
                    DType::BF16 => format!("rg_f32_to_bf16({value}[l])"),
                    _ => format!("{value}[l]"),
                };
                lines.push(format!(
                    "    for(size_t l=0;l<{}u;l++) (({}*)buffers[{}])[{}+l]={};",
                    active,
                    ctype(ty),
                    slot,
                    base,
                    stored
                ));
            }
            crate::VectorInstKind::Control => {
                if let Some(d) = dst.clone() {
                    lines.push(format!(
                        "    size_t {d}[{}]; for(size_t l=0;l<{}u;l++) {d}[l]={base}+l;",
                        usize::from(program.lanes),
                        active
                    ));
                }
            }
        }
        if let (
            Some(crate::VectorOperand::Register {
                physical, dtype, ..
            }),
            Some(name),
        ) = (&inst.dst, dst)
        {
            names.insert((*physical, *dtype), name);
        }
    }
    Ok(())
}
fn render_reduction(
    store: &UOp,
    _abi: &KernelAbi,
    ids: &BTreeMap<u64, usize>,
    out: &BufferAbi,
    lines: &mut Vec<String>,
) -> Result<Option<String>, JitError> {
    let Some(finalize) = store
        .sources()
        .get(1)
        .filter(|n| matches!(n.kind(), UOpKind::ReduceFinalize))
    else {
        return Ok(None);
    };
    let update = finalize
        .sources()
        .first()
        .ok_or_else(|| JitError::Unsupported("reduction finalize".into()))?;
    let init = update
        .sources()
        .first()
        .ok_or_else(|| JitError::Unsupported("reduction init".into()))?;
    let UArg::Reduction {
        input_shape,
        output_shape,
        axes,
        keepdim,
        kind,
        mean,
    } = init.arg()
    else {
        return Err(JitError::Unsupported("reduction metadata".into()));
    };
    if !matches!(
        kind,
        crate::ReduceKind::Sum
            | crate::ReduceKind::Mean
            | crate::ReduceKind::Max
            | crate::ReduceKind::Min
    ) {
        return Err(JitError::Unsupported(
            "native C reduction kind is not implemented".into(),
        ));
    }
    if matches!(kind, crate::ReduceKind::Max | crate::ReduceKind::Min) && !out.dtype.is_float() {
        return Err(JitError::Unsupported(
            "native extrema reduction currently requires floating point".into(),
        ));
    }
    let value_node = update
        .sources()
        .get(1)
        .ok_or_else(|| JitError::Unsupported("reduction producer".into()))?;
    let reduce_dims: Vec<usize> = axes.iter().map(|a| input_shape.dims()[*a]).collect();
    let reduce_len = reduce_dims.iter().product::<usize>();
    let out_len = output_shape
        .numel()
        .map_err(|_| JitError::Unsupported("reduction output overflow".into()))?;
    // Replace the elementwise loop opened by the shared prologue.
    lines.pop();
    lines.push(format!(
        "  for (size_t rg_out = 0; rg_out < {out_len}u; ++rg_out) {{"
    ));
    let acc = accumulator_type(out.dtype);
    let initial = if matches!(kind, crate::ReduceKind::Max) {
        "-INFINITY"
    } else if matches!(kind, crate::ReduceKind::Min) {
        "INFINITY"
    } else {
        "0"
    };
    lines.push(format!("    {acc} rg_acc = {initial};"));
    if reduce_len != 0 {
        lines.push(format!(
            "    for (size_t rg_r = 0; rg_r < {reduce_len}u; ++rg_r) {{"
        ));
        lines.push(format!(
            "      size_t rg_i = {};",
            reduction_index_expr(input_shape, output_shape, axes, *keepdim)
        ));
        let mut map = BTreeMap::new();
        let value = emit(value_node, ids, &mut map, lines)?;
        if matches!(kind, crate::ReduceKind::Max) {
            lines.push(format!(
                "      if (!isnan(({acc})({value})) && ({acc})({value}) > rg_acc) rg_acc = ({acc})({value});"
            ));
        } else if matches!(kind, crate::ReduceKind::Min) {
            lines.push(format!(
                "      if (!isnan(({acc})({value})) && ({acc})({value}) < rg_acc) rg_acc = ({acc})({value});"
            ));
        } else if out.dtype == DType::Bool {
            lines.push(format!("      rg_acc = (uint8_t)(rg_acc || ({value}));"));
        } else {
            lines.push(format!(
                "      rg_acc = ({acc})(rg_acc + ({acc})({value}));"
            ));
        }
        lines.push("    }".into());
    }
    let store_value: String = if *mean && reduce_len == 0 {
        match out.dtype.category() {
            crate::DTypeCategory::Float => "NAN".into(),
            _ => "0".into(),
        }
    } else if *mean {
        // CpuBackend turns the finalized scalar into f64 before mean, including
        // its intentionally lossy U64 conversion, then quantizes to dtype.
        format!("((double)rg_acc / (double){reduce_len})")
    } else {
        "rg_acc".into()
    };
    let store_value = match out.dtype {
        DType::F16 => format!("rg_f32_to_f16((float)({store_value}))"),
        DType::BF16 => format!("rg_f32_to_bf16((float)({store_value}))"),
        _ => store_value,
    };
    lines.push(format!(
        "    (({}*)buffers[{}])[rg_out] = ({});",
        ctype(out.dtype),
        ids[&out.id],
        store_value
    ));
    lines.push("  }".into());
    lines.push("  return failure[1] ? (int)failure[1] : 0;".into());
    lines.push("}".into());
    Ok(Some(lines.join("\n") + "\n"))
}
fn accumulator_type(dtype: DType) -> &'static str {
    match dtype.category() {
        crate::DTypeCategory::Bool => "uint8_t",
        crate::DTypeCategory::Signed => "int64_t",
        crate::DTypeCategory::Unsigned => "uint64_t",
        crate::DTypeCategory::Float => "double",
    }
}
fn reduction_index_expr(
    input: &crate::Shape,
    output: &crate::Shape,
    axes: &[usize],
    keepdim: bool,
) -> String {
    let mut terms = Vec::new();
    let mut out_axis = 0usize;
    let mut red_axis = 0usize;
    for axis in 0..input.rank() {
        let dim = input.dims()[axis];
        let coord = if axes.contains(&axis) {
            let div = axes[red_axis + 1..]
                .iter()
                .map(|a| input.dims()[*a])
                .product::<usize>();
            red_axis += 1;
            if dim == 0 || div == 0 {
                "0u".into()
            } else {
                format!("((rg_r / {div}u) % {dim}u)")
            }
        } else {
            let oa = if keepdim {
                axis
            } else {
                let x = out_axis;
                out_axis += 1;
                x
            };
            let div = output.dims()[oa + 1..].iter().product::<usize>();
            if keepdim {
                out_axis += 1;
            }
            if dim == 0 || div == 0 {
                "0u".into()
            } else {
                format!("((rg_out / {div}u) % {dim}u)")
            }
        };
        terms.push(coord);
    }
    let mut result = terms.remove(0);
    for (coord, dim) in terms.into_iter().zip(input.dims().iter().skip(1)) {
        result = format!("(({result})*{dim}u+{coord})");
    }
    result
}
fn emit(
    n: &UOp,
    ids: &BTreeMap<u64, usize>,
    map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
) -> Result<String, JitError> {
    let id = map.len();
    map.insert(id, lines.len() + 1);
    let ty = n
        .ty()
        .ok_or_else(|| JitError::Unsupported(format!("untyped {:?}", n.kind())))?
        .scalar;
    let mut s = |i: usize| emit(&n.sources()[i], ids, map, lines);
    match n.kind() {
        UOpKind::Const => match n.arg() {
            UArg::Int(v) => Ok(format!("(({}){}LL)", expr_ctype(ty), v)),
            UArg::Scalar { dtype, bits } if *dtype == ty => Ok(literal_expr(*dtype, *bits)),
            UArg::Scalar { .. } => {
                Err(JitError::Unsupported("scalar literal/type mismatch".into()))
            }
            _ => Err(JitError::Unsupported("non-integer const".into())),
        },
        UOpKind::Load => {
            let ix = n
                .sources()
                .first()
                .ok_or_else(|| JitError::Unsupported("load no index".into()))?;
            let (buffer, off) = match ix.arg() {
                UArg::BufferIndex {
                    buffer,
                    input_shape,
                    output_shape,
                    ..
                } => (*buffer, broadcast_offset(input_shape, output_shape)),
                UArg::ViewBufferIndex {
                    buffer,
                    input_shape,
                    output_shape,
                    view,
                    ..
                } => {
                    let logical = broadcast_offset(input_shape, output_shape);
                    (*buffer, view_offset(view, &logical))
                }
                _ => return Err(JitError::Unsupported("load index".into())),
            };
            let load = match ty {
                DType::F16 => "rg_f16_to_f32",
                DType::BF16 => "rg_bf16_to_f32",
                _ => "",
            };
            if load.is_empty() {
                Ok(format!(
                    "(({}*)buffers[{}])[{}]",
                    ctype(ty),
                    ids[&buffer],
                    off
                ))
            } else {
                Ok(format!(
                    "{load}(((uint16_t*)buffers[{}])[{}])",
                    ids[&buffer], off
                ))
            }
        }
        UOpKind::Cast => Ok(format!("(({})({}))", expr_ctype(ty), s(0)?)),
        UOpKind::GraphUnary(op) => {
            let a = s(0)?;
            let x = match op {
                crate::UnaryOp::Neg => format!("-({a})"),
                crate::UnaryOp::Abs => format!("fabs({a})"),
                crate::UnaryOp::Square => format!("({a})*({a})"),
                crate::UnaryOp::Relu => format!("(({a})>0?({a}):0)"),
                crate::UnaryOp::Sqrt => format!("sqrt({a})"),
                crate::UnaryOp::Rsqrt => format!("(1.0/sqrt({a}))"),
                crate::UnaryOp::Exp => format!("exp({a})"),
                crate::UnaryOp::Reciprocal => format!("(1.0/({a}))"),
                _ => return Err(JitError::Unsupported(format!("unary {op:?}"))),
            };
            Ok(x)
        }
        UOpKind::GraphBinary(op) => {
            let (a, b) = (s(0)?, s(1)?);
            let x = match op {
                crate::BinaryOp::Add => "+",
                crate::BinaryOp::Sub => "-",
                crate::BinaryOp::Mul => "*",
                crate::BinaryOp::Div | crate::BinaryOp::TruncDiv if !ty.is_float() => {
                    return Ok(int_call("div", ty, &a, &b));
                }
                crate::BinaryOp::Mod | crate::BinaryOp::FMod if !ty.is_float() => {
                    return Ok(int_call("mod", ty, &a, &b));
                }
                crate::BinaryOp::Shl if !ty.is_float() => {
                    return Ok(format!(
                        "(({})rg_shl((uint64_t)({a}),(int64_t)({b}),{},rg_i,failure)",
                        ctype(ty),
                        ty.bits()
                    ));
                }
                crate::BinaryOp::Shr if !ty.is_float() => {
                    return Ok(format!(
                        "(({})rg_shr((uint64_t)({a}),(int64_t)({b}),{},rg_i,failure)",
                        ctype(ty),
                        ty.bits()
                    ));
                }
                crate::BinaryOp::Div => "/",
                crate::BinaryOp::BitAnd => "&",
                crate::BinaryOp::BitOr => "|",
                crate::BinaryOp::BitXor => "^",
                crate::BinaryOp::Maximum => return Ok(format!("(({a})>({b})?({a}):({b}))")),
                crate::BinaryOp::Minimum => return Ok(format!("(({a})<({b})?({a}):({b}))")),
                _ => return Err(JitError::Unsupported(format!("binary {op:?}"))),
            };
            Ok(format!("(({a}) {x} ({b}))"))
        }
        UOpKind::GraphCompare(op) => {
            let (a, b) = (s(0)?, s(1)?);
            let x = match op {
                crate::CompareOp::Eq => "==",
                crate::CompareOp::Ne => "!=",
                crate::CompareOp::Lt => "<",
                crate::CompareOp::Le => "<=",
                crate::CompareOp::Gt => ">",
                crate::CompareOp::Ge => ">=",
            };
            Ok(format!("(({a}) {x} ({b}))"))
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            Ok(format!("(({})?({}):({}))", s(0)?, s(1)?, s(2)?))
        }
        _ => Err(JitError::Unsupported(format!("{:?}", n.kind()))),
    }
}
fn int_call(op: &str, ty: DType, a: &str, b: &str) -> String {
    let signed = matches!(ty.category(), crate::DTypeCategory::Signed);
    let helper = match (op, signed) {
        ("div", true) => "rg_sdiv",
        ("div", false) => "rg_udiv",
        ("mod", true) => "rg_smod",
        ("mod", false) => "rg_umod",
        _ => unreachable!(),
    };
    format!(
        "(({}){helper}(({})({a}),({})({b}),rg_i,failure))",
        ctype(ty),
        if signed { "int64_t" } else { "uint64_t" },
        if signed { "int64_t" } else { "uint64_t" }
    )
}
fn literal_expr(dtype: DType, bits: u64) -> String {
    match dtype {
        DType::Bool => format!("((uint8_t){})", u8::from(bits != 0)),
        DType::F16 => format!("rg_f16_to_f32((uint16_t)0x{:04x}u)", bits as u16),
        DType::BF16 => format!("rg_bf16_to_f32((uint16_t)0x{:04x}u)", bits as u16),
        DType::F32 => format!(
            "((union{{uint32_t u;float f;}}){{.u=0x{:08x}u}}.f)",
            bits as u32
        ),
        DType::F64 => format!(
            "((union{{uint64_t u;double f;}}){{.u=UINT64_C(0x{:016x})}}.f)",
            bits
        ),
        _ => format!("(({})UINT64_C(0x{:016x}))", ctype(dtype), bits),
    }
}
fn broadcast_offset(input: &crate::Shape, output: &crate::Shape) -> String {
    if input == output {
        return "rg_i".into();
    }
    let pad = output.rank() - input.rank();
    let mut parts = Vec::new();
    for (a, d) in input.dims().iter().enumerate() {
        if *d != 1 {
            let divisor = output.dims()[pad + a + 1..].iter().product::<usize>();
            parts.push(if *d == 0 || divisor == 0 {
                "0u".into()
            } else {
                format!("((rg_i / {divisor}u) % {d}u)")
            });
        }
    }
    if parts.is_empty() {
        "0".into()
    } else {
        let mut x = parts.remove(0);
        for (p, d) in parts
            .into_iter()
            .zip(input.dims().iter().filter(|d| **d != 1).skip(1))
        {
            x = format!("(({x})*{d}u+{p})")
        }
        x
    }
}
fn view_offset(view: &crate::ViewMap, logical: &str) -> String {
    let mut terms = vec![format!("{}u", view.offset)];
    for (axis, (&dimension, &stride)) in view
        .logical_shape
        .dims()
        .iter()
        .zip(&view.strides)
        .enumerate()
    {
        if dimension == 0 || stride == 0 {
            continue;
        }
        let divisor = view.logical_shape.dims()[axis + 1..]
            .iter()
            .product::<usize>();
        if divisor == 0 {
            terms.push("0u".into());
        } else {
            terms.push(format!("((({logical})/{divisor}u)%{dimension}u)*{stride}u"));
        }
    }
    format!("({})", terms.join("+"))
}
fn ctype(d: DType) -> &'static str {
    match d {
        DType::Bool => "uint8_t",
        DType::I8 => "int8_t",
        DType::U8 => "uint8_t",
        DType::I16 => "int16_t",
        DType::U16 | DType::F16 | DType::BF16 => "uint16_t",
        DType::I32 => "int32_t",
        DType::U32 => "uint32_t",
        DType::I64 => "int64_t",
        DType::U64 => "uint64_t",
        DType::F32 => "float",
        DType::F64 => "double",
    }
}
fn expr_ctype(d: DType) -> &'static str {
    match d {
        DType::F16 | DType::BF16 => "float",
        _ => ctype(d),
    }
}
fn key(s: &str) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("rustgrad-cpu-jit-v1")
}
static COMPILE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
fn compile_cached(r: &RenderedC) -> Result<PathBuf, JitError> {
    let _guard = COMPILE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| JitError::Io("compile lock poisoned".into()))?;
    let d = cache_dir();
    fs::create_dir_all(&d).map_err(|e| JitError::Io(e.to_string()))?;
    let lib = d.join(format!(
        "{}.{}",
        r.cache_key,
        if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        }
    ));
    if lib.exists() {
        return Ok(lib);
    }
    let c = d.join(format!("{}.c", r.cache_key));
    fs::write(&c, &r.source).map_err(|e| JitError::Io(e.to_string()))?;
    let tmp = d.join(format!("{}.tmp", r.cache_key));
    let out = Command::new("cc")
        .args([
            "-std=c11",
            "-O2",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            "-Werror",
            "-o",
        ])
        .arg(&tmp)
        .arg(&c)
        .output()
        .map_err(|e| JitError::Compiler {
            status: None,
            stderr: e.to_string(),
        })?;
    if !out.status.success() {
        return Err(JitError::Compiler {
            status: out.status.code(),
            stderr: String::from_utf8_lossy(&out.stderr)
                .chars()
                .take(8192)
                .collect(),
        });
    }
    fs::rename(&tmp, &lib).map_err(|e| JitError::Io(e.to_string()))?;
    Ok(lib)
}
struct Library(*mut c_void);
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
impl Library {
    fn open(p: &Path) -> Result<Self, JitError> {
        let s = CString::new(p.to_string_lossy().as_bytes())
            .map_err(|e| JitError::Loader(e.to_string()))?;
        let h = unsafe { dlopen(s.as_ptr(), 2) };
        if h.is_null() {
            return Err(JitError::Loader(last_error()));
        }
        Ok(Self(h))
    }
    unsafe fn symbol<T: Copy>(&self, n: &[u8]) -> Result<T, JitError> {
        let p = unsafe { dlsym(self.0, n.as_ptr().cast()) };
        if p.is_null() {
            return Err(JitError::Loader(last_error()));
        }
        Ok(unsafe { std::mem::transmute_copy(&p) })
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.0);
        }
    }
}
unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, n: *const c_char) -> *mut c_void;
    fn dlclose(h: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}
fn last_error() -> String {
    unsafe {
        let p = dlerror();
        if p.is_null() {
            "unknown loader error".into()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Graph, Scalar, Shape, SymbolicExpr, TensorData};
    use std::collections::{BTreeMap, HashMap};
    #[test]
    fn source_is_deterministic_and_native_call_works() {
        let mut g = Graph::new();
        let a = g.input("a", Shape::from([3]));
        let b = g.input("b", Shape::from([3]));
        let o = g.add(a, b).unwrap();
        let u = crate::lower_graph_elementwise(&g, o).unwrap();
        let r = CpuJit::render(&u).unwrap();
        assert_eq!(r.source, CpuJit::render(&u).unwrap().source);
        let k = CpuJit::compile(&u).unwrap();
        let mut x = JitBuffer::zeroed(DType::F32, 3, false);
        let mut y = JitBuffer::zeroed(DType::F32, 3, false);
        let z = JitBuffer::zeroed(DType::F32, 3, true);
        for (b, v) in [(&mut x, 1f32), (&mut y, 2f32)] {
            for q in b.bytes_mut().chunks_exact_mut(4) {
                q.copy_from_slice(&v.to_ne_bytes())
            }
        }
        let mut buffers = [x, y, z];
        for _ in 0..3 {
            k.call(&mut buffers, &[]).unwrap();
        }
        for q in buffers[2].bytes().chunks_exact(4) {
            assert_eq!(f32::from_ne_bytes(q.try_into().unwrap()), 3.0);
        }
        let mut malformed = [
            JitBuffer::zeroed(DType::F32, 2, false),
            JitBuffer::zeroed(DType::F32, 3, false),
            JitBuffer::zeroed(DType::F32, 3, true),
        ];
        assert!(matches!(
            k.call(&mut malformed, &[]),
            Err(JitError::InvalidBuffer(_))
        ));
    }

    #[test]
    fn exact_native_arithmetic_wraps_and_reports_division_failure() {
        let mut g = Graph::new();
        let a = g.input_dtype("a", Shape::from([2]), DType::U64);
        let b = g.input_dtype("b", Shape::from([2]), DType::U64);
        let out = g.add(a, b).unwrap();
        let k = CpuJit::compile(&crate::lower_graph_elementwise(&g, out).unwrap()).unwrap();
        let mut left = JitBuffer::zeroed(DType::U64, 2, false);
        let mut right = JitBuffer::zeroed(DType::U64, 2, false);
        for (dst, values) in [(&mut left, [u64::MAX, 7]), (&mut right, [1, 9])] {
            for (bytes, value) in dst.bytes_mut().chunks_exact_mut(8).zip(values) {
                bytes.copy_from_slice(&value.to_ne_bytes());
            }
        }
        let result = JitBuffer::zeroed(DType::U64, 2, true);
        let mut buffers = [left, right, result];
        k.call(&mut buffers, &[]).unwrap();
        assert_eq!(
            u64::from_ne_bytes(buffers[2].bytes()[..8].try_into().unwrap()),
            0
        );
        assert_eq!(
            u64::from_ne_bytes(buffers[2].bytes()[8..].try_into().unwrap()),
            16
        );

        let mut div_graph = Graph::new();
        let n = div_graph.input_dtype("n", Shape::from([1]), DType::I64);
        let d = div_graph.input_dtype("d", Shape::from([1]), DType::I64);
        let quotient = div_graph.div(n, d).unwrap();
        let div = CpuJit::compile(&crate::lower_graph_elementwise(&div_graph, quotient).unwrap())
            .unwrap();
        let mut numerator = JitBuffer::zeroed(DType::I64, 1, false);
        numerator.bytes_mut().copy_from_slice(&42i64.to_ne_bytes());
        let denominator = JitBuffer::zeroed(DType::I64, 1, false);
        let output = JitBuffer::zeroed(DType::I64, 1, true);
        assert_eq!(
            div.call(&mut [numerator, denominator, output], &[]),
            Err(JitError::DivisionByZero { index: 0 })
        );
    }

    #[test]
    fn narrow_float_raw_storage_executes_natively() {
        let mut g = Graph::new();
        let a = g.input_dtype("a", Shape::from([1]), DType::F16);
        let b = g.input_dtype("b", Shape::from([1]), DType::F16);
        let out = g.add(a, b).unwrap();
        let k = CpuJit::compile(&crate::lower_graph_elementwise(&g, out).unwrap()).unwrap();
        let mut left = JitBuffer::zeroed(DType::F16, 1, false);
        let mut right = JitBuffer::zeroed(DType::F16, 1, false);
        left.bytes_mut().copy_from_slice(&0x3c00u16.to_ne_bytes());
        right.bytes_mut().copy_from_slice(&0x3c00u16.to_ne_bytes());
        let output = JitBuffer::zeroed(DType::F16, 1, true);
        let mut buffers = [left, right, output];
        k.call(&mut buffers, &[]).unwrap();
        assert_eq!(
            u16::from_ne_bytes(buffers[2].bytes().try_into().unwrap()),
            0x4000
        );

        let mut bf = Graph::new();
        let x = bf.input_dtype("x", Shape::from([1]), DType::BF16);
        let z = bf.neg(x).unwrap();
        let kernel = CpuJit::compile(&crate::lower_graph_elementwise(&bf, z).unwrap()).unwrap();
        let mut input = JitBuffer::zeroed(DType::BF16, 1, false);
        input.bytes_mut().copy_from_slice(&0x0001u16.to_ne_bytes());
        let output = JitBuffer::zeroed(DType::BF16, 1, true);
        let mut buffers = [input, output];
        kernel.call(&mut buffers, &[]).unwrap();
        assert_eq!(
            u16::from_ne_bytes(buffers[1].bytes().try_into().unwrap()),
            0x8001
        );
    }

    #[test]
    fn static_sum_and_mean_execute_natively() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::from([2, 3]));
        let sum = g
            .reduce(x, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let kernel = crate::lower_graph_reduction(&g, sum).unwrap();
        let rendered = CpuJit::render(&kernel).unwrap();
        assert!(rendered.source.contains("rg_acc"));
        let jit = CpuJit::compile(&kernel).unwrap();
        let mut input = JitBuffer::zeroed(DType::F32, 6, false);
        for (bytes, value) in input
            .bytes_mut()
            .chunks_exact_mut(4)
            .zip([1f32, 2., 3., 4., 5., 6.])
        {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        let output = JitBuffer::zeroed(DType::F32, 2, true);
        let mut buffers = [input, output];
        jit.call(&mut buffers, &[]).unwrap();
        assert_eq!(
            f32::from_ne_bytes(buffers[1].bytes()[..4].try_into().unwrap()),
            6.0
        );
        assert_eq!(
            f32::from_ne_bytes(buffers[1].bytes()[4..].try_into().unwrap()),
            15.0
        );

        let mean = g.reduce(x, crate::ReduceKind::Mean, None, false).unwrap();
        let mean_jit = CpuJit::compile(&crate::lower_graph_reduction(&g, mean).unwrap()).unwrap();
        let mut input = JitBuffer::zeroed(DType::F32, 6, false);
        for (bytes, value) in input
            .bytes_mut()
            .chunks_exact_mut(4)
            .zip([1f32, 2., 3., 4., 5., 6.])
        {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        let output = JitBuffer::zeroed(DType::F32, 1, true);
        let mut buffers = [input, output];
        mean_jit.call(&mut buffers, &[]).unwrap();
        assert_eq!(
            f32::from_ne_bytes(buffers[1].bytes().try_into().unwrap()),
            3.5
        );
    }

    #[test]
    fn reduction_dtype_native_matches_interpreter_and_cpu() {
        for (dtype, values) in [
            (
                DType::Bool,
                vec![
                    Scalar::Bool(false),
                    Scalar::Bool(true),
                    Scalar::Bool(true),
                    Scalar::Bool(false),
                ],
            ),
            (
                DType::I8,
                vec![Scalar::I(120), Scalar::I(10), Scalar::I(-3), Scalar::I(1)],
            ),
            (
                DType::I32,
                vec![Scalar::I(1), Scalar::I(-2), Scalar::I(3), Scalar::I(4)],
            ),
            (
                DType::I64,
                vec![Scalar::I(1), Scalar::I(-2), Scalar::I(3), Scalar::I(4)],
            ),
            (
                DType::U8,
                vec![Scalar::U(250), Scalar::U(10), Scalar::U(3), Scalar::U(1)],
            ),
            (
                DType::U32,
                vec![Scalar::U(1), Scalar::U(2), Scalar::U(3), Scalar::U(4)],
            ),
            (
                DType::U64,
                vec![Scalar::U(1), Scalar::U(2), Scalar::U(3), Scalar::U(4)],
            ),
            (
                DType::F16,
                vec![Scalar::F(1.), Scalar::F(-2.), Scalar::F(3.), Scalar::F(4.)],
            ),
            (
                DType::BF16,
                vec![Scalar::F(1.), Scalar::F(-2.), Scalar::F(3.), Scalar::F(4.)],
            ),
            (
                DType::F32,
                vec![Scalar::F(1.), Scalar::F(-2.), Scalar::F(3.), Scalar::F(4.)],
            ),
            (
                DType::F64,
                vec![Scalar::F(1.), Scalar::F(-2.), Scalar::F(3.), Scalar::F(4.)],
            ),
        ] {
            for kind in [crate::ReduceKind::Sum, crate::ReduceKind::Mean] {
                let mut graph = Graph::new();
                let x = graph.input_dtype("x", Shape::from([2, 2]), dtype);
                let output = graph.reduce(x, kind, Some(vec![1]), true).unwrap();
                let input = TensorData::from_scalars([2, 2], dtype, values.clone()).unwrap();
                let inputs = HashMap::from([("x".into(), input.clone())]);
                let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
                let interpreted = crate::execute_elementwise(&graph, output, &inputs).unwrap();
                let jit = CpuJit::compile(&crate::lower_graph_reduction(&graph, output).unwrap())
                    .unwrap();
                let mut buffers = [
                    JitBuffer::from_tensor(&input, false),
                    JitBuffer::zeroed(expected.dtype(), expected.len(), true),
                ];
                jit.call(&mut buffers, &[]).unwrap();
                let native = buffers
                    .into_iter()
                    .nth(1)
                    .unwrap()
                    .into_tensor(expected.shape().clone())
                    .unwrap();
                assert_eq!(native.storage(), expected.storage(), "{dtype:?} {kind:?}");
                assert_eq!(
                    native.storage(),
                    interpreted.storage(),
                    "{dtype:?} {kind:?}"
                );
            }
        }
    }

    #[test]
    fn empty_reduction_domains_have_defined_native_results() {
        for kind in [crate::ReduceKind::Sum, crate::ReduceKind::Mean] {
            let mut graph = Graph::new();
            let x = graph.input("x", Shape::from([2, 0]));
            let output = graph.reduce(x, kind, Some(vec![1]), false).unwrap();
            let input = TensorData::new([2, 0], vec![]).unwrap();
            let inputs = HashMap::from([("x".into(), input.clone())]);
            let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
            let jit =
                CpuJit::compile(&crate::lower_graph_reduction(&graph, output).unwrap()).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&input, false),
                JitBuffer::zeroed(DType::F32, 2, true),
            ];
            jit.call(&mut buffers, &[]).unwrap();
            let native = buffers
                .into_iter()
                .nth(1)
                .unwrap()
                .into_tensor(Shape::from([2]))
                .unwrap();
            if matches!(kind, crate::ReduceKind::Sum) {
                assert_eq!(native.to_vec_f64(), expected.to_vec_f64());
            } else {
                assert!(native.to_vec_f64().iter().all(|v| v.is_nan()));
                assert!(expected.to_vec_f64().iter().all(|v| v.is_nan()));
            }
        }
    }

    #[test]
    fn symbolic_specialization_validates_and_executes_two_bindings() {
        let expr = SymbolicExpr::variable("n", 0, 4).unwrap();
        let var = expr.variables().into_iter().next().unwrap();
        let symbolic = SymbolicShape::new(vec![expr.into()]);
        for n in [0usize, 3] {
            let bindings = BTreeMap::from([(var.clone(), n as i64)]);
            let shape = symbolic.bind(&bindings).unwrap();
            let mut graph = Graph::new();
            let x = graph.input_symbolic("x", &symbolic, &bindings).unwrap();
            let out = graph.square(x).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, out).unwrap();
            let jit = CpuJit::compile_specialized(&uop, std::slice::from_ref(&symbolic), &bindings)
                .unwrap();
            let input = TensorData::from_scalars(
                shape.clone(),
                DType::F32,
                (0..n).map(|x| Scalar::F(x as f64 + 1.)),
            )
            .unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&input, false),
                JitBuffer::zeroed(DType::F32, n, true),
            ];
            jit.call(&mut buffers, &[]).unwrap();
            let native = buffers
                .into_iter()
                .nth(1)
                .unwrap()
                .into_tensor(shape.clone())
                .unwrap();
            let expected = CpuBackend
                .execute(&graph, out, &HashMap::from([("x".into(), input)]))
                .unwrap();
            assert_eq!(native.storage(), expected.storage());
        }
        assert!(matches!(
            CpuJit::compile_specialized(
                &UOp::sink(vec![]),
                std::slice::from_ref(&symbolic),
                &BTreeMap::new()
            ),
            Err(JitError::Symbolic(_))
        ));
        let other = SymbolicExpr::variable("other", 0, 1)
            .unwrap()
            .variables()
            .into_iter()
            .next()
            .unwrap();
        assert!(matches!(
            CpuJit::compile_specialized(
                &UOp::sink(vec![]),
                std::slice::from_ref(&symbolic),
                &BTreeMap::from([(var, 1), (other, 1)])
            ),
            Err(JitError::Symbolic(_))
        ));
    }

    #[test]
    fn portable_vector_main_and_tail_match_scalar_and_cpu() {
        for len in [0usize, 1, 3, 4, 5, 8] {
            let mut graph = Graph::new();
            let x = graph.input("x", Shape::from([len]));
            let output = graph.neg(x).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let plan = CpuJit::vector_plan(&uop).unwrap();
            assert_eq!(plan.lanes, 4);
            assert!(plan.enabled);
            let vector_source = CpuJit::render_vectorized(&uop).unwrap();
            let linear = CpuJit::linearize(&uop).unwrap();
            let spaces = crate::MemorySpacePlan::from_linear(&linear).unwrap();
            let program = crate::VectorProgram::from_linear(&linear, &spaces).unwrap();
            assert!(
                program.b1_eligibility().is_ok(),
                "{:?}",
                program.b1_eligibility()
            );
            assert!(vector_source.source.contains("rg_base"));
            assert!(vector_source.source.contains("VectorProgram key"));
            let vector = CpuJit::compile_vectorized(&uop).unwrap();
            let scalar = CpuJit::compile(&uop).unwrap();
            let input = TensorData::from_scalars(
                [len],
                DType::F32,
                (0..len).map(|v| Scalar::F(v as f64 - 2.)),
            )
            .unwrap();
            let mut vb = [
                JitBuffer::from_tensor(&input, false),
                JitBuffer::zeroed(DType::F32, len, true),
            ];
            let mut sb = [
                JitBuffer::from_tensor(&input, false),
                JitBuffer::zeroed(DType::F32, len, true),
            ];
            vector.call(&mut vb, &[]).unwrap();
            scalar.call(&mut sb, &[]).unwrap();
            let native = vb
                .into_iter()
                .nth(1)
                .unwrap()
                .into_tensor(Shape::from([len]))
                .unwrap();
            let scalar_native = sb
                .into_iter()
                .nth(1)
                .unwrap()
                .into_tensor(Shape::from([len]))
                .unwrap();
            let expected = CpuBackend
                .execute(&graph, output, &HashMap::from([("x".into(), input)]))
                .unwrap();
            assert_eq!(native.storage(), scalar_native.storage());
            assert_eq!(native.storage(), expected.storage());
        }
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2, 1]));
        let y = graph.input("y", Shape::from([1, 3]));
        let output = graph.add(x, y).unwrap();
        assert!(
            !CpuJit::vector_plan(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap()
                .enabled
        );
    }

    #[test]
    fn portable_b2_exact_and_narrow_float_vectors_execute() {
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::F16,
            DType::BF16,
        ] {
            for len in [0usize, 1, 3, 4, 5, 8, 17] {
                let mut graph = Graph::new();
                let x = graph.input_dtype("x", Shape::from([len]), dtype);
                let y = graph.input_dtype("y", Shape::from([len]), dtype);
                let out = graph.add(x, y).unwrap();
                let uop = crate::lower_graph_elementwise(&graph, out).unwrap();
                let rendered = CpuJit::render_vectorized(&uop).unwrap();
                assert!(rendered.source.contains("B2 VectorProgram"), "{dtype:?}");
                let vector = CpuJit::compile_vectorized(&uop).unwrap();
                let scalar = CpuJit::compile(&uop).unwrap();
                let values = (0..len).map(|i| {
                    if dtype == DType::Bool {
                        Scalar::Bool(i % 2 == 0)
                    } else if dtype.is_float() {
                        Scalar::F(i as f64 - 3.25)
                    } else if matches!(dtype.category(), crate::DTypeCategory::Signed) {
                        Scalar::I(i as i64 - 5)
                    } else {
                        Scalar::U((i as u64).wrapping_mul(37))
                    }
                });
                let input = TensorData::from_scalars([len], dtype, values).unwrap();
                let other = TensorData::from_scalars(
                    [len],
                    dtype,
                    (0..len).map(|i| {
                        if dtype == DType::Bool {
                            Scalar::Bool(i % 3 == 0)
                        } else if dtype.is_float() {
                            Scalar::F(0.5)
                        } else if matches!(dtype.category(), crate::DTypeCategory::Signed) {
                            Scalar::I(i64::MAX)
                        } else {
                            Scalar::U(u64::MAX)
                        }
                    }),
                )
                .unwrap();
                let mut native_buffers = [
                    JitBuffer::from_tensor(&input, false),
                    JitBuffer::from_tensor(&other, false),
                    JitBuffer::zeroed(dtype, len, true),
                ];
                let mut scalar_buffers = native_buffers.clone();
                vector.call(&mut native_buffers, &[]).unwrap();
                scalar.call(&mut scalar_buffers, &[]).unwrap();
                let native = native_buffers[2]
                    .clone()
                    .into_tensor(Shape::from([len]))
                    .unwrap();
                let scalar_native = scalar_buffers[2]
                    .clone()
                    .into_tensor(Shape::from([len]))
                    .unwrap();
                let expected = CpuBackend
                    .execute(
                        &graph,
                        out,
                        &HashMap::from([("x".into(), input), ("y".into(), other)]),
                    )
                    .unwrap();
                assert_eq!(
                    native.storage(),
                    scalar_native.storage(),
                    "{dtype:?} len={len}"
                );
                assert_eq!(native.storage(), expected.storage(), "{dtype:?} len={len}");
            }
        }
    }

    #[test]
    fn portable_b2_reports_first_division_and_shift_failure() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([5]), DType::I32);
        let y = graph.input_dtype("y", Shape::from([5]), DType::I32);
        let out = graph.div(x, y).unwrap();
        let jit = CpuJit::compile_vectorized(&crate::lower_graph_elementwise(&graph, out).unwrap())
            .unwrap();
        let lhs = TensorData::from_scalars([5], DType::I32, [Scalar::I(1); 5]).unwrap();
        let rhs = TensorData::from_scalars(
            [5],
            DType::I32,
            [
                Scalar::I(1),
                Scalar::I(0),
                Scalar::I(0),
                Scalar::I(1),
                Scalar::I(0),
            ],
        )
        .unwrap();
        assert_eq!(
            jit.call(
                &mut [
                    JitBuffer::from_tensor(&lhs, false),
                    JitBuffer::from_tensor(&rhs, false),
                    JitBuffer::zeroed(DType::I32, 5, true)
                ],
                &[]
            ),
            Err(JitError::DivisionByZero { index: 1 })
        );

        let mut shift = Graph::new();
        let a = shift.input_dtype("a", Shape::from([5]), DType::U8);
        let b = shift.input_dtype("b", Shape::from([5]), DType::U8);
        let out = shift.shl(a, b).unwrap();
        let jit = CpuJit::compile_vectorized(&crate::lower_graph_elementwise(&shift, out).unwrap())
            .unwrap();
        let count = TensorData::from_scalars(
            [5],
            DType::U8,
            [
                Scalar::U(1),
                Scalar::U(8),
                Scalar::U(9),
                Scalar::U(1),
                Scalar::U(2),
            ],
        )
        .unwrap();
        assert_eq!(
            jit.call(
                &mut [
                    JitBuffer::from_tensor(&count, false),
                    JitBuffer::from_tensor(&count, false),
                    JitBuffer::zeroed(DType::U8, 5, true)
                ],
                &[]
            ),
            Err(JitError::InvalidShift { index: 1 })
        );
    }

    #[test]
    fn b1_vector_program_compare_select_executes() {
        let mut graph = Graph::new();
        let a = graph.input("a", Shape::from([5]));
        let b = graph.input("b", Shape::from([5]));
        let predicate = graph.gt(a, b).unwrap();
        let output = graph.select(predicate, a, b).unwrap();
        let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
        assert!(
            CpuJit::render_vectorized(&uop)
                .unwrap()
                .source
                .contains("VectorProgram key")
        );
        let jit = CpuJit::compile_vectorized(&uop).unwrap();
        let left = TensorData::from_scalars(
            [5],
            DType::F32,
            [
                Scalar::F(1.),
                Scalar::F(5.),
                Scalar::F(-1.),
                Scalar::F(0.),
                Scalar::F(9.),
            ],
        )
        .unwrap();
        let right = TensorData::from_scalars(
            [5],
            DType::F32,
            [
                Scalar::F(2.),
                Scalar::F(4.),
                Scalar::F(-2.),
                Scalar::F(0.),
                Scalar::F(8.),
            ],
        )
        .unwrap();
        let mut buffers = [
            JitBuffer::from_tensor(&left, false),
            JitBuffer::from_tensor(&right, false),
            JitBuffer::zeroed(DType::F32, 5, true),
        ];
        jit.call(&mut buffers, &[]).unwrap();
        let native = buffers
            .into_iter()
            .nth(2)
            .unwrap()
            .into_tensor(Shape::from([5]))
            .unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("a".into(), left), ("b".into(), right)]),
            )
            .unwrap();
        assert_eq!(native.storage(), expected.storage());
    }
}
