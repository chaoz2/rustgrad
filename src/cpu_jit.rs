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

pub const RENDERER_VERSION: &str = "rustgrad-c11-scalar-v2";
pub const ABI_VERSION: u32 = 2;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum JitError {
    Unsupported(String),
    InvalidBuffer(String),
    DivisionByZero { index: usize },
    InvalidShift { index: usize },
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
            _ => return Err(JitError::Loader(format!("unknown native status {status}"))),
        }
        Ok(())
    }
}

fn render(root: &UOp) -> Result<RenderedC, JitError> {
    render_with_policy(root, false)
}
fn vector_plan(root: &UOp) -> Result<VectorPlan, JitError> {
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
    let nodes = root
        .topological()
        .map_err(|e| JitError::Unsupported(e.to_string()))?;
    let mut buffers: BTreeMap<u64, BufferAbi> = BTreeMap::new();
    for n in &nodes {
        if let UArg::BufferIndex {
            buffer, elements, ..
        } = n.arg()
        {
            let ty = n
                .ty()
                .ok_or_else(|| JitError::Unsupported("untyped buffer index".into()))?
                .scalar;
            buffers.entry(*buffer).or_insert(BufferAbi {
                id: *buffer,
                dtype: ty,
                elements: *elements,
                mutable: false,
            });
        }
    }
    for n in &nodes {
        if matches!(n.kind(), UOpKind::Store)
            && let Some(i) = n.sources().first()
            && let UArg::BufferIndex { buffer, .. } = i.arg()
            && let Some(b) = buffers.get_mut(buffer)
        {
            b.mutable = true;
        }
    }
    let abi = KernelAbi {
        version: ABI_VERSION,
        buffers: buffers.into_values().collect(),
        symbol_count: 0,
    };
    let mut ids = BTreeMap::new();
    for (i, b) in abi.buffers.iter().enumerate() {
        ids.insert(b.id, i);
    }
    let store = root
        .sources()
        .iter()
        .find(|x| matches!(x.kind(), UOpKind::Store))
        .ok_or_else(|| JitError::Unsupported("Sink without Store".into()))?;
    let (out_id, extent) = match store.sources().first().and_then(|x| match x.arg() {
        UArg::BufferIndex {
            buffer, elements, ..
        } => Some((*buffer, *elements)),
        _ => None,
    }) {
        Some(x) => x,
        None => return Err(JitError::Unsupported("Store needs BufferIndex".into())),
    };
    let out = abi.buffers.iter().find(|b| b.id == out_id).unwrap();
    let (plan, linear_key) = if request_vector {
        let linear = CpuJit::linearize(root)?;
        linear
            .validate()
            .map_err(|error| JitError::Unsupported(error.to_string()))?;
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
            )),
        )
    } else {
        (
            VectorPlan {
                lanes: 1,
                enabled: false,
                reason: "disabled".into(),
            },
            None,
        )
    };
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
        mean,
    } = init.arg()
    else {
        return Err(JitError::Unsupported("reduction metadata".into()));
    };
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
    lines.push(format!("    {acc} rg_acc = 0;"));
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
        if out.dtype == DType::Bool {
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
            format!("((rg_r / {div}u) % {dim}u)")
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
            format!("((rg_out / {div}u) % {dim}u)")
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
            _ => Err(JitError::Unsupported("non-integer const".into())),
        },
        UOpKind::Load => {
            let ix = n
                .sources()
                .first()
                .ok_or_else(|| JitError::Unsupported("load no index".into()))?;
            let UArg::BufferIndex {
                buffer,
                input_shape,
                output_shape,
                ..
            } = ix.arg()
            else {
                return Err(JitError::Unsupported("load index".into()));
            };
            let off = broadcast_offset(input_shape, output_shape);
            let load = match ty {
                DType::F16 => "rg_f16_to_f32",
                DType::BF16 => "rg_bf16_to_f32",
                _ => "",
            };
            if load.is_empty() {
                Ok(format!(
                    "(({}*)buffers[{}])[{}]",
                    ctype(ty),
                    ids[buffer],
                    off
                ))
            } else {
                Ok(format!(
                    "{load}(((uint16_t*)buffers[{}])[{}])",
                    ids[buffer], off
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
fn broadcast_offset(input: &crate::Shape, output: &crate::Shape) -> String {
    if input == output {
        return "rg_i".into();
    }
    let pad = output.rank() - input.rank();
    let mut parts = Vec::new();
    for (a, d) in input.dims().iter().enumerate() {
        if *d != 1 {
            let divisor = output.dims()[pad + a + 1..].iter().product::<usize>();
            parts.push(format!("((rg_i / {divisor}u) % {d}u)"));
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
        .args(["-std=c11", "-O2", "-fPIC", "-shared", "-Werror", "-o"])
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
            let output = graph.square(x).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let plan = CpuJit::vector_plan(&uop).unwrap();
            assert_eq!(plan.lanes, 4);
            assert!(plan.enabled);
            let vector_source = CpuJit::render_vectorized(&uop).unwrap();
            assert!(vector_source.source.contains("rg_base"));
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
}
