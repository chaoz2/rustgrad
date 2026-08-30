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
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
#[path = "cpu_jit_random.rs"]
mod random;

// Bump whenever the scalar expression surface changes: mixed captures include
// this identity before they can reuse a native-renderer admission decision.
pub const RENDERER_VERSION: &str = "rustgrad-c11-scalar-v27";
pub const ABI_VERSION: u32 = 2;
const C11_COMPILER_COMMAND: &str = "cc";
const C11_COMPILER_FLAGS: &[&str] = &[
    "-std=c11",
    "-O2",
    "-ffp-contract=off",
    "-fPIC",
    "-shared",
    "-Werror",
];

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
    pub quantized_buffers: Vec<QuantizedBufferAbi>,
    pub pointer_order: Vec<KernelPointerAbi>,
    pub symbol_count: usize,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferAbi {
    pub id: u64,
    pub dtype: DType,
    pub elements: usize,
    pub mutable: bool,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedBufferAbi {
    pub id: u64,
    pub desc: crate::QuantizedBufferDesc,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelPointerAbi {
    Dense(usize),
    Quantized(usize),
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
            crate::Storage::Float8(values) => out.bytes.copy_from_slice(values.as_raw()),
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
        if let Some(format) = self.dtype.float8_format() {
            return crate::TensorData::from_storage(
                shape,
                crate::Storage::Float8(crate::Float8Storage::from_raw(format, self.bytes)),
            );
        }
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
        let (lib, call) = match load_library_call(&path) {
            Ok(loaded) => loaded,
            Err(_) => {
                // A durable cache entry is untrusted until the loader and its
                // exact stable entry symbol accept it. Evict a damaged entry
                // before rebuilding so a truncated or stale artifact cannot
                // poison every later compile for this source identity.
                evict_cached_library(&path)?;
                let rebuilt = compile_cached(r)?;
                match load_library_call(&rebuilt) {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        let _ = evict_cached_library(&rebuilt);
                        return Err(error);
                    }
                }
            }
        };
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
        if !self.abi.quantized_buffers.is_empty() {
            return Err(JitError::InvalidBuffer(
                "packed resources require the mixed native ABI".into(),
            ));
        }
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
        let mut ptrs: Vec<*mut c_void> = self
            .abi
            .pointer_order
            .iter()
            .map(|entry| match entry {
                KernelPointerAbi::Dense(index) => buffers[*index].bytes.as_mut_ptr().cast(),
                KernelPointerAbi::Quantized(_) => unreachable!("validated dense ABI"),
            })
            .collect();
        self.invoke_transactional(buffers, &mut ptrs, symbols)
    }

    pub(crate) fn call_with_quantized(
        &self,
        buffers: &mut [JitBuffer],
        quantized: &[&crate::QuantizedTensorData],
        symbols: &[i64],
    ) -> Result<(), JitError> {
        if buffers.len() != self.abi.buffers.len()
            || quantized.len() != self.abi.quantized_buffers.len()
        {
            return Err(JitError::InvalidBuffer(
                "mixed native resource count mismatch".into(),
            ));
        }
        if symbols.len() != self.abi.symbol_count {
            return Err(JitError::InvalidBuffer(
                "mixed native symbol count mismatch".into(),
            ));
        }
        for (buffer, want) in buffers.iter().zip(&self.abi.buffers) {
            buffer.validate(want)?;
        }
        for (value, want) in quantized.iter().zip(&self.abi.quantized_buffers) {
            value
                .validate()
                .map_err(|error| JitError::InvalidBuffer(error.to_string()))?;
            if value.descriptor() != &want.desc {
                return Err(JitError::InvalidBuffer(format!(
                    "quantized buffer {} descriptor mismatch",
                    want.id
                )));
            }
        }
        let mut ptrs = self
            .abi
            .pointer_order
            .iter()
            .map(|entry| match entry {
                KernelPointerAbi::Dense(index) => buffers[*index].bytes.as_mut_ptr().cast(),
                KernelPointerAbi::Quantized(index) => {
                    quantized[*index].bytes().as_ptr().cast_mut().cast()
                }
            })
            .collect::<Vec<_>>();
        self.invoke_transactional(buffers, &mut ptrs, symbols)
    }

    /// Native kernels may detect a domain failure after earlier loop iterations
    /// have stored results. Keep every ABI-declared mutable buffer private to
    /// the call until native completion succeeds, including intentional
    /// in-place input/output buffers.
    fn invoke_transactional(
        &self,
        buffers: &mut [JitBuffer],
        ptrs: &mut [*mut c_void],
        symbols: &[i64],
    ) -> Result<(), JitError> {
        let backups = buffers
            .iter()
            .zip(&self.abi.buffers)
            .enumerate()
            .filter(|(_, (_, abi))| abi.mutable)
            .map(|(index, (buffer, _))| (index, buffer.bytes.clone()))
            .collect::<Vec<_>>();
        let result = self.invoke(ptrs, symbols);
        if result.is_err() {
            for (index, bytes) in backups {
                buffers[index].bytes.copy_from_slice(&bytes);
            }
        }
        result
    }

    fn invoke(&self, ptrs: &mut [*mut c_void], symbols: &[i64]) -> Result<(), JitError> {
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
    if matches!(
        root.kind(),
        UOpKind::Matmul | UOpKind::Conv2d | UOpKind::Movement | UOpKind::Random
    ) {
        return Ok(VectorPlan {
            lanes: 1,
            enabled: false,
            reason: "static contraction uses scalar lanes".into(),
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
    if let (UOpKind::Random, UArg::Random(plan)) = (root.kind(), root.arg()) {
        return random::render(plan);
    }
    if matches!(root.kind(), UOpKind::Matmul)
        && let Some(plan) = root.arg().quantized_matmul_plan()
    {
        return render_quantized_matmul(plan);
    }
    if matches!(root.kind(), UOpKind::Matmul)
        && let Some(plan) = root.arg().matmul_plan()
    {
        return render_matmul(plan);
    }
    if matches!(root.kind(), UOpKind::Conv2d)
        && let Some(plan) = root.arg().static_conv2d_plan()
    {
        return render_static_conv2d(plan);
    }
    if matches!(root.kind(), UOpKind::Movement)
        && let Some(plan) = root.arg().quantized_row_gather_plan()
    {
        return render_quantized_row_gather(plan);
    }
    if let (UOpKind::Movement, UArg::Movement(plan)) = (root.kind(), root.arg()) {
        return render_movement(plan);
    }
    let nodes = root
        .topological()
        .map_err(|e| JitError::Unsupported(e.to_string()))?;
    let needs_erf = nodes.iter().any(|node| {
        matches!(
            node.kind(),
            UOpKind::GraphUnary(crate::UnaryOp::Erf | crate::UnaryOp::Erfc)
        )
    });
    let needs_f8_encode = nodes.iter().any(|node| {
        matches!(node.kind(), UOpKind::Cast | UOpKind::ReduceFinalize)
            && node.ty().is_some_and(|ty| ty.scalar.is_float8())
    });
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
        pointer_order: (0..buffers.len()).map(KernelPointerAbi::Dense).collect(),
        buffers,
        quantized_buffers: Vec::new(),
        symbol_count: 0,
    };
    let mut ids = BTreeMap::new();
    for (i, b) in abi.buffers.iter().enumerate() {
        ids.insert(b.id, i);
    }
    let out_id = *out_id;
    let extent = *extent;
    let out = abi.buffers.iter().find(|b| b.id == out_id).unwrap();
    // Float8 storage is a tagged byte encoding, never an ordered integer.
    // Numeric comparisons have one complete decode-only contract. WHERE has
    // a separate storage contract: a Bool condition chooses one homogeneous
    // raw Float8 byte, so neither branch is decoded or re-encoded. CAST uses
    // the exact host codec in both directions, preserving same-format bytes.
    // Validate those cases node-by-node and fail closed for every other
    // Float8 ALU graph rather than treating payload bytes as numeric lanes.
    for node in &nodes {
        if !matches!(
            node.kind(),
            UOpKind::GraphUnary(_)
                | UOpKind::GraphBinary(_)
                | UOpKind::GraphCompare(_)
                | UOpKind::GraphLogical(_)
                | UOpKind::Cast
                | UOpKind::Ternary(_)
        ) {
            continue;
        }
        let node_dtype = node.ty().map(|ty| ty.scalar);
        let source_types = node
            .sources()
            .iter()
            .map(|source| source.ty().map(|ty| ty.scalar))
            .collect::<Vec<_>>();
        let touches_float8 = node_dtype.is_some_and(DType::is_float8)
            || source_types
                .iter()
                .copied()
                .any(|dtype| dtype.is_some_and(DType::is_float8));
        if !touches_float8 {
            continue;
        }
        match node.kind() {
            UOpKind::GraphCompare(_)
                if node_dtype == Some(DType::Bool)
                    && source_types.len() == 2
                    && source_types[0] == source_types[1]
                    && source_types[0].is_some_and(DType::is_float8) => {}
            UOpKind::Ternary(crate::uop::Ternary::Where)
                if node_dtype.is_some_and(DType::is_float8)
                    && source_types.len() == 3
                    && source_types[0] == Some(DType::Bool)
                    && source_types[1] == node_dtype
                    && source_types[2] == node_dtype => {}
            UOpKind::Cast if source_types.len() == 1 && source_types[0].is_some() => {}
            UOpKind::GraphCompare(_) => {
                return Err(JitError::Unsupported(
                    "native Float8 comparison requires one homogeneous format".into(),
                ));
            }
            UOpKind::Ternary(crate::uop::Ternary::Where) => {
                return Err(JitError::Unsupported(
                    "native Float8 selection requires homogeneous branches".into(),
                ));
            }
            _ => {
                return Err(JitError::Unsupported(
                    "native Float8 elementwise supports comparisons, raw selection, and typed casts only"
                        .into(),
                ));
            }
        }
    }
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
        "#include <string.h>".into(),
        "#include <limits.h>".into(),
        format!("/* {RENDERER_VERSION} C11 ABI v2; vector lanes={} ({}) linear={linear_key:?} */", plan.lanes, plan.reason),
        "static float rg_f16_to_f32(uint16_t h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,o;if(!e)o=m? s|((uint32_t)(113-__builtin_clz(m))<<23)|((uint32_t)(m<<(126-__builtin_clz(m)))<<13):s;else o=e==31?s|0x7f800000|(m<<13):s|((e+112)<<23)|(m<<13);union{uint32_t u;float f;}v={o};return v.f;}".into(),
        "static uint16_t rg_f32_to_f16(float x){union{float f;uint32_t u;}v={x};uint32_t b=v.u,s=(b>>16)&0x8000,e=(b>>23)&255,m=b&0x7fffff;if(e==255)return(uint16_t)(s|0x7c00|(m?((m>>13)|1):0));int q=(int)e-112;if(q<=0){if(q<-10)return(uint16_t)s;uint32_t z=m|0x800000,sh=(uint32_t)(14-q),r=z>>sh,rem=z&((1u<<sh)-1),half=1u<<(sh-1);return(uint16_t)(s+r+(rem>half||(rem==half&&(r&1))));}if(q>=31)return(uint16_t)(s|0x7c00);uint32_t r=m>>13,rem=m&0x1fff; r+=rem>0x1000||(rem==0x1000&&(r&1));if(r==0x400){if(q==30)return(uint16_t)(s|0x7c00);q++;r=0;}return(uint16_t)(s|((uint32_t)q<<10)|r);}".into(),
        "static float rg_bf16_to_f32(uint16_t b){union{uint32_t u;float f;}v={(uint32_t)b<<16};return v.f;}".into(),
        "static uint16_t rg_f32_to_bf16(float x){union{float f;uint32_t u;}v={x};uint32_t b=v.u,hi=b>>16;if((b&0x7f800000)==0x7f800000&&(b&0x007fffff))return(uint16_t)((hi&0x7f)?hi:(hi|1));return(uint16_t)((b+0x7fff+((b>>16)&1))>>16);}".into(),
        // mode: 0=E4M3, 1=E5M2, 2=FNUZ. This is the exact inverse of
        // Float8Format::decode: FNUZ reserves 0x80 as NaN, E5M2 reserves the
        // terminal exponent for infinity/NaN, and E4M3 reserves only its
        // terminal mantissa. ldexp keeps subnormal and normal powers exact.
        "static double rg_f8_decode(uint8_t x,int bias,unsigned mb,unsigned mode){unsigned em=(1u<<(7u-mb))-1u,mm=(1u<<mb)-1u,e=(x>>mb)&em,m=x&mm,s=x>>7;if(mode==2u&&x==0x80u)return NAN;if((x&0x7fu)==0u)return s?-0.0:0.0;if(mode!=2u&&e==em){if(mode==1u){double v=m?NAN:INFINITY;return s?-v:v;}if(m==mm)return NAN;}double v=e?ldexp(1.0+(double)m/(double)(mm+1u),(int)e-bias):ldexp((double)m/(double)(mm+1u),1-bias);return s?-v:v;}".into(),
        "static double rg_round_ties_even(double x){double lo,frac,out;if(!isfinite(x)||x==0.0)return x;lo=floor(x);frac=x-lo;if(frac<0.5)out=lo;else if(frac>0.5)out=lo+1.0;else out=fmod(lo,2.0)==0.0?lo:lo+1.0;return out==0.0?copysign(0.0,x):out;}".into(),
        "static int8_t rg_i8(uint8_t x){int8_t r;memcpy(&r,&x,1);return r;} static int16_t rg_i16(uint16_t x){int16_t r;memcpy(&r,&x,2);return r;} static int32_t rg_i32(uint32_t x){int32_t r;memcpy(&r,&x,4);return r;} static int64_t rg_i64(uint64_t x){int64_t r;memcpy(&r,&x,8);return r;}".into(),
        "static int64_t rg_sdiv(int64_t a,int64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return(a==INT64_MIN&&b==-1)?INT64_MIN:a/b;}".into(),
        "static uint64_t rg_udiv(uint64_t a,uint64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return a/b;}".into(),
        "static int64_t rg_sfdiv(int64_t a,int64_t b,uint64_t i,uint64_t *f){int64_t q=rg_sdiv(a,b,i,f),r;if(!b||(a==INT64_MIN&&b==-1))return q;r=a%b;return r<0?q-(b>0?1:-1):q;}".into(),
        "static int64_t rg_srem(int64_t a,int64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return(a==INT64_MIN&&b==-1)?0:a%b;}".into(),
        "static int64_t rg_smod(int64_t a,int64_t b,uint64_t i,uint64_t *f){int64_t r=rg_srem(a,b,i,f);if(!b||r>=0)return r;return b>0?r+b:r-b;}".into(),
        "static uint64_t rg_umod(uint64_t a,uint64_t b,uint64_t i,uint64_t *f){if(!b){if(!f[1]){f[0]=i;f[1]=1;}return 0;}return a%b;}".into(),
        "static uint64_t rg_shl(uint64_t a,int64_t b,unsigned bits,uint64_t i,uint64_t *f){if(b<0||(uint64_t)b>=bits){if(!f[1]){f[0]=i;f[1]=2;}return 0;}return a<<b;}".into(),
        "static uint64_t rg_shr(uint64_t a,int64_t b,unsigned bits,uint64_t i,uint64_t *f){if(b<0||(uint64_t)b>=bits){if(!f[1]){f[0]=i;f[1]=2;}return 0;}return a>>b;}".into(),
        "static int64_t rg_sshr(uint64_t a,int64_t b,unsigned bits,uint64_t i,uint64_t *f){uint64_t mask,r,mag;if(b<0||(uint64_t)b>=bits){if(!f[1]){f[0]=i;f[1]=2;}return 0;}mask=bits==64?UINT64_MAX:((UINT64_C(1)<<bits)-1);r=(a&mask)>>(unsigned)b;if(!((a>>(bits-1))&1))return(int64_t)r;if(b)r|=mask^(mask>>((unsigned)b));mag=(~r+1)&mask;if(bits==64&&mag==(UINT64_C(1)<<63))return INT64_MIN;return-(int64_t)mag;}".into(),
        "int rustgrad_kernel(void **buffers, const int64_t *symbols, uint64_t *failure) { (void)symbols; failure[0]=UINT64_MAX; failure[1]=0;".into(),
        if plan.enabled { format!("  for (size_t rg_base = 0; rg_base + {}u <= {extent}u; rg_base += {}u) {{ for (size_t rg_lane = 0; rg_lane < {}u; ++rg_lane) {{ size_t rg_i = rg_base + rg_lane;", plan.lanes, plan.lanes, plan.lanes) } else { format!("  for (size_t rg_i = 0; rg_i < {extent}u; ++rg_i) {{") },
    ];
    if needs_f8_encode {
        let kernel_index = lines
            .iter()
            .position(|line| line.starts_with("int rustgrad_kernel"))
            .expect("renderer always emits the kernel declaration");
        lines.insert(
            kernel_index,
            // Exact Float8Format::encode mirror. The thresholds are f64
            // payload bits, so conversion is independent of long-double ABI.
            "static uint8_t rg_f8_encode(double x,int bias,unsigned sb,unsigned mode,uint64_t min_half,uint64_t overflow,uint8_t max_normal,uint64_t min_normal){if(mode==2u&&!isfinite(x))return 0x80u;if(mode==2u&&x==0.0)return 0u;uint8_t sign=signbit(x)?0x80u:0u;if(mode==0u&&!isfinite(x))return sign?0xffu:0x7fu;if(mode==1u&&!isfinite(x))return(uint8_t)(sign|(isinf(x)?0x7cu:0x7fu));union{double f;uint64_t u;}v={x};uint64_t bits=v.u,abs=bits&UINT64_C(0x7fffffffffffffff),mask=(UINT64_C(1)<<(sb-1u))-1u,mantissa=(bits>>(53u-sb))&mask,half=UINT64_C(1)<<(52u-sb),result;int exponent=(int)((bits>>52)&0x7ffu)-1023+bias;if(abs<=min_half)result=0;else if(abs>overflow)result=max_normal;else if(abs>=min_normal){result=((uint64_t)exponent<<(sb-1u))|mantissa;uint64_t round_bits=bits&((half<<1u)-1u);if(round_bits>half||(round_bits==half&&(mantissa&1u)))result++;}else{unsigned shift=(unsigned)(1-exponent);mantissa|=UINT64_C(1)<<(sb-1u);result=mantissa>>shift;uint64_t h=half<<shift,round_bits=(bits|(UINT64_C(1)<<52))&((h<<1u)-1u);if(round_bits>h||(round_bits==h&&(result&1u)))result++;}if(mode==2u&&result==0)return 0;return(uint8_t)(result|sign);}".into(),
        );
    }
    if needs_erf {
        let kernel_index = lines
            .iter()
            .position(|line| line.starts_with("int rustgrad_kernel"))
            .expect("renderer always emits the kernel declaration");
        lines.insert(
            kernel_index,
            "static double rg_erf(double x){double t,p;if(isnan(x))return x;t=1.0/(1.0+0.3275911*fabs(x));p=((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t;return copysign(1.0,x)*(1.0-p*exp((-x)*x));}".into(),
        );
    }
    if let Some(rendered) = render_reduction(store, &abi, &ids, out, &mut lines)? {
        let source = rendered;
        let cache_key = native_cache_key("", &source);
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
    let cache_key = native_cache_key("", &source);
    Ok(RenderedC {
        source,
        source_map: map,
        abi,
        cache_key,
    })
}

fn render_static_conv2d(plan: &crate::StaticConv2dPlan) -> Result<RenderedC, JitError> {
    plan.validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let elements = |shape: &crate::Shape| {
        shape
            .numel()
            .map_err(|_| JitError::Unsupported("static conv shape overflow".into()))
    };
    let mut buffers = vec![
        BufferAbi {
            id: plan.input.index() as u64,
            dtype: DType::F32,
            elements: elements(&plan.input_shape)?,
            mutable: false,
        },
        BufferAbi {
            id: plan.weight.index() as u64,
            dtype: DType::F32,
            elements: elements(&plan.weight_shape)?,
            mutable: false,
        },
    ];
    if let Some(bias) = plan.bias {
        buffers.push(BufferAbi {
            id: bias.index() as u64,
            dtype: DType::F32,
            elements: elements(plan.bias_shape.as_ref().expect("validated bias shape"))?,
            mutable: false,
        });
    }
    buffers.push(BufferAbi {
        id: plan.output.index() as u64,
        dtype: DType::F32,
        elements: elements(&plan.output_shape)?,
        mutable: true,
    });
    let abi = KernelAbi {
        version: ABI_VERSION,
        pointer_order: (0..buffers.len()).map(KernelPointerAbi::Dense).collect(),
        buffers,
        quantized_buffers: Vec::new(),
        symbol_count: 0,
    };
    let ids = abi
        .buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| (buffer.id, index))
        .collect::<BTreeMap<_, _>>();
    let bias_decl = plan.bias.map(|bias| {
        format!(
            "  const float *rg_bias=(const float*)buffers[{}];",
            ids[&(bias.index() as u64)]
        )
    });
    let bias_value = if plan.bias.is_some() {
        "rg_bias[rg_oc]"
    } else {
        "0.0f"
    };
    let output_elements = elements(&plan.output_shape)?;
    let mut lines = vec![
        "#include <stdint.h>".into(),
        "#include <stddef.h>".into(),
        format!(
            "/* {RENDERER_VERSION} static-conv1x1 plan={} N={} Cin={} Cout={} H={} W={} */",
            plan.cache_key,
            plan.batch,
            plan.input_channels,
            plan.output_channels,
            plan.height,
            plan.width
        ),
        "int rustgrad_kernel(void **buffers, const int64_t *symbols, uint64_t *failure) { (void)symbols; failure[0]=UINT64_MAX; failure[1]=0;".into(),
        format!(
            "  const float *rg_input=(const float*)buffers[{}];",
            ids[&(plan.input.index() as u64)]
        ),
        format!(
            "  const float *rg_weight=(const float*)buffers[{}];",
            ids[&(plan.weight.index() as u64)]
        ),
    ];
    if let Some(decl) = bias_decl {
        lines.push(decl);
    }
    lines.extend([
        format!(
            "  float *rg_output=(float*)buffers[{}];",
            ids[&(plan.output.index() as u64)]
        ),
        format!("  for (size_t rg_i=0; rg_i<{output_elements}u; ++rg_i) {{"),
        "    size_t rg_q=rg_i;".into(),
        format!(
            "    size_t rg_x=rg_q%{}u; rg_q/={}u;",
            plan.width, plan.width
        ),
        format!(
            "    size_t rg_y=rg_q%{}u; rg_q/={}u;",
            plan.height, plan.height
        ),
        format!(
            "    size_t rg_oc=rg_q%{}u; rg_q/={}u;",
            plan.output_channels, plan.output_channels
        ),
        "    size_t rg_n=rg_q;".into(),
        format!("    float rg_acc={bias_value};"),
        format!(
            "    for (size_t rg_ic=0; rg_ic<{}u; ++rg_ic) {{",
            plan.input_channels
        ),
        format!(
            "      size_t rg_input_offset=((rg_n*{}u+rg_ic)*{}u+rg_y)*{}u+rg_x;",
            plan.input_channels, plan.height, plan.width
        ),
        format!(
            "      size_t rg_weight_offset=rg_oc*{}u+rg_ic;",
            plan.input_channels
        ),
        "      rg_acc += rg_input[rg_input_offset]*rg_weight[rg_weight_offset];".into(),
        "    }".into(),
        "    rg_output[rg_i]=rg_acc;".into(),
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
        pointer_order: (0..buffers.len()).map(KernelPointerAbi::Dense).collect(),
        buffers,
        quantized_buffers: Vec::new(),
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
    ]);
    if plan.dtype == DType::F32 {
        // The CPU oracle commits both the product and running sum through
        // binary_scalar at F32 storage width on every contraction step.
        // Keep the C temporaries explicit so a compiler cannot retain a wider
        // accumulator across the loop.
        lines.extend([
            "    float rg_acc=0.0f;".into(),
            format!("    for (size_t rg_k=0; rg_k<{}u; ++rg_k) {{ float rg_product=(float)(rg_lhs[{lhs_offset}]*rg_rhs[{rhs_offset}]); rg_acc=(float)(rg_acc+rg_product); }}", plan.k),
            "    rg_out[rg_i]=rg_acc;".into(),
        ]);
    } else {
        lines.extend([
            "    double rg_acc=0.0;".into(),
            format!("    for (size_t rg_k=0; rg_k<{}u; ++rg_k) rg_acc += rg_lhs[{lhs_offset}] * rg_rhs[{rhs_offset}];", plan.k),
            "    rg_out[rg_i]=rg_acc;".into(),
        ]);
    }
    lines.extend(["  }".into(), "  return 0;".into(), "}".into()]);
    let source = lines.join("\n") + "\n";
    let cache_key = native_cache_key(&plan.cache_key.to_string(), &source);
    Ok(RenderedC {
        source,
        source_map: BTreeMap::from([(0, 1)]),
        abi,
        cache_key,
    })
}

fn quantized_decode_snippet(kind: crate::GgmlType) -> Result<&'static str, JitError> {
    match kind {
        crate::GgmlType::Q4_0 => Ok(
            "size_t rg_lane=rg_k&31u; const uint8_t *rg_b=rg_w+rg_block*18u; float rg_d=rg_half(rg_b); uint8_t rg_p=rg_b[2u+(rg_lane&15u)]; int rg_q=(rg_lane<16u?(rg_p&15u):(rg_p>>4))-8; float rg_v=rg_d*(float)rg_q;",
        ),
        crate::GgmlType::Q8_0 => Ok(
            "size_t rg_lane=rg_k&31u; const uint8_t *rg_b=rg_w+rg_block*34u; float rg_d=rg_half(rg_b); float rg_v=rg_d*(float)(int8_t)rg_b[2u+rg_lane];",
        ),
        crate::GgmlType::Q4K => Ok(
            "size_t rg_lane=rg_k&255u,rg_g=rg_lane/32u,rg_l=rg_lane&31u; const uint8_t *rg_b=rg_w+rg_block*144u; float rg_d=rg_half(rg_b),rg_dm=rg_half(rg_b+2u); unsigned rg_s,rg_m;if(rg_g<4u){rg_s=rg_b[4u+rg_g]&63u;rg_m=rg_b[8u+rg_g]&63u;}else{size_t rg_x=rg_g-4u;rg_s=(rg_b[12u+rg_x]&15u)|((rg_b[4u+rg_x]>>6)<<4);rg_m=(rg_b[12u+rg_x]>>4)|((rg_b[8u+rg_x]>>6)<<4);}unsigned rg_q=(rg_b[16u+(rg_g/2u)*32u+rg_l]>>((rg_g&1u)*4u))&15u;float rg_v=rg_d*(float)rg_s*(float)rg_q-rg_dm*(float)rg_m;",
        ),
        crate::GgmlType::Q6K => Ok(
            "size_t rg_lane=rg_k&255u,rg_h=rg_lane/128u,rg_x=rg_lane&127u; const uint8_t *rg_b=rg_w+rg_block*210u; unsigned rg_low=(rg_b[rg_h*64u+(rg_x&63u)]>>((rg_x/64u)*4u))&15u;unsigned rg_hi=((rg_b[128u+rg_h*32u+(rg_x&31u)]>>((rg_x/32u)*2u))&3u)<<4;int rg_q=(int)(rg_low|rg_hi)-32;int rg_s=(int)(int8_t)rg_b[192u+rg_lane/16u];float rg_v=rg_half(rg_b+208u)*(float)(rg_q*rg_s);",
        ),
        _ => Err(JitError::Unsupported(
            "unsupported GGML quantized kernel format".into(),
        )),
    }
}

fn render_quantized_matmul(plan: &crate::QuantizedMatmulPlan) -> Result<RenderedC, JitError> {
    plan.validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let activation_elements = plan
        .activation_shape
        .numel()
        .map_err(|_| JitError::Unsupported("quantized activation shape overflow".into()))?;
    let output_elements = plan
        .output_shape
        .numel()
        .map_err(|_| JitError::Unsupported("quantized output shape overflow".into()))?;
    let buffers = vec![
        BufferAbi {
            id: plan.activation.index() as u64,
            dtype: DType::F32,
            elements: activation_elements,
            mutable: false,
        },
        BufferAbi {
            id: plan.output.index() as u64,
            dtype: DType::F32,
            elements: output_elements,
            mutable: true,
        },
    ];
    let abi = KernelAbi {
        version: ABI_VERSION,
        buffers,
        quantized_buffers: vec![QuantizedBufferAbi {
            id: plan.weight.index() as u64,
            desc: plan.weight_desc.clone(),
        }],
        pointer_order: vec![
            KernelPointerAbi::Dense(0),
            KernelPointerAbi::Quantized(0),
            KernelPointerAbi::Dense(1),
        ],
        symbol_count: 0,
    };
    let block_value = quantized_decode_snippet(plan.weight_desc.ggml_type)?;
    let block_elements = plan.weight_desc.block_elements;
    let blocks_per_row = if plan.k == 0 {
        0
    } else {
        plan.k / block_elements
    };
    let source = [
        "#include <stdint.h>\n#include <stddef.h>\n",
        "static float rg_half(const uint8_t *p){uint16_t h=(uint16_t)p[0]|((uint16_t)p[1]<<8);uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,o;if(!e){if(!m)o=s;else{unsigned sh=0;while(!(m&0x400)){m<<=1;sh++;}m&=0x3ff;o=s|((uint32_t)(113-sh)<<23)|(m<<13);}}else o=e==31?s|0x7f800000|(m<<13):s|((e+112)<<23)|(m<<13);union{uint32_t u;float f;}v={o};return v.f;}\n",
        &format!(
            "/* {RENDERER_VERSION} quantized-matmul plan={} type={} bytes={} */\n",
            plan.cache_key,
            plan.weight_desc.ggml_type.raw(),
            plan.weight_desc.bytes
        ),
        "int rustgrad_kernel(void **buffers,const int64_t *symbols,uint64_t *failure){(void)symbols;failure[0]=UINT64_MAX;failure[1]=0;const float *rg_a=(const float*)buffers[0];const uint8_t *rg_w=(const uint8_t*)buffers[1];float *rg_o=(float*)buffers[2];",
        &format!(
            "for(size_t rg_i=0;rg_i<{output_elements}u;++rg_i){{size_t rg_col=rg_i%{}u,rg_row=rg_i/{}u;double rg_acc=0.0;for(size_t rg_k=0;rg_k<{}u;++rg_k){{size_t rg_block=rg_col*{}u+rg_k/{}u;{}rg_acc+=(double)rg_a[rg_row*{}u+rg_k]*(double)rg_v;}}rg_o[rg_i]=(float)rg_acc;}}return 0;}}\n",
            plan.n.max(1),
            plan.n.max(1),
            plan.k,
            blocks_per_row,
            block_elements,
            block_value,
            plan.k,
        ),
    ]
    .concat();
    let cache_key = native_cache_key(&plan.cache_key.to_string(), &source);
    Ok(RenderedC {
        source,
        source_map: BTreeMap::from([(0, 1)]),
        abi,
        cache_key,
    })
}

fn render_quantized_row_gather(
    plan: &crate::QuantizedRowGatherPlan,
) -> Result<RenderedC, JitError> {
    plan.validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let index_elements = plan
        .indices_shape
        .numel()
        .map_err(|_| JitError::Unsupported("quantized gather indices overflow".into()))?;
    let output_elements = plan
        .output_shape
        .numel()
        .map_err(|_| JitError::Unsupported("quantized gather output overflow".into()))?;
    let rows = plan.weight_desc.logical_shape.dims()[0];
    let columns = plan.weight_desc.logical_shape.dims()[1];
    let blocks_per_row = columns / plan.weight_desc.block_elements;
    let signed = matches!(
        plan.indices_dtype,
        DType::I8 | DType::I16 | DType::I32 | DType::I64
    );
    let negative = if signed {
        "if(rg_idx[rg_i]<0){failure[0]=rg_i;return 3;}"
    } else {
        ""
    };
    let index_type = ctype(plan.indices_dtype);
    let block_value = quantized_decode_snippet(plan.weight_desc.ggml_type)?;
    let abi = KernelAbi {
        version: ABI_VERSION,
        buffers: vec![
            BufferAbi {
                id: plan.indices.index() as u64,
                dtype: plan.indices_dtype,
                elements: index_elements,
                mutable: false,
            },
            BufferAbi {
                id: plan.output.index() as u64,
                dtype: DType::F32,
                elements: output_elements,
                mutable: true,
            },
        ],
        quantized_buffers: vec![QuantizedBufferAbi {
            id: plan.weight.index() as u64,
            desc: plan.weight_desc.clone(),
        }],
        pointer_order: vec![
            KernelPointerAbi::Dense(0),
            KernelPointerAbi::Quantized(0),
            KernelPointerAbi::Dense(1),
        ],
        symbol_count: 0,
    };
    let source = [
        "#include <stdint.h>\n#include <stddef.h>\n",
        "static float rg_half(const uint8_t *p){uint16_t h=(uint16_t)p[0]|((uint16_t)p[1]<<8);uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,o;if(!e){if(!m)o=s;else{unsigned sh=0;while(!(m&0x400)){m<<=1;sh++;}m&=0x3ff;o=s|((uint32_t)(113-sh)<<23)|(m<<13);}}else o=e==31?s|0x7f800000|(m<<13):s|((e+112)<<23)|(m<<13);union{uint32_t u;float f;}v={o};return v.f;}\n",
        &format!(
            "/* {RENDERER_VERSION} quantized-row-gather plan={} type={} bytes={} */\n",
            plan.cache_key,
            plan.weight_desc.ggml_type.raw(),
            plan.weight_desc.bytes
        ),
        &format!(
            "int rustgrad_kernel(void **buffers,const int64_t *symbols,uint64_t *failure){{(void)symbols;failure[0]=UINT64_MAX;failure[1]=0;const {index_type} *rg_idx=(const {index_type}*)buffers[0];const uint8_t *rg_w=(const uint8_t*)buffers[1];float *rg_o=(float*)buffers[2];for(size_t rg_i=0;rg_i<{index_elements}u;++rg_i){{{negative}uint64_t rg_row=(uint64_t)rg_idx[rg_i];if(rg_row>={rows}u){{failure[0]=rg_i;return 3;}}}}for(size_t rg_i=0;rg_i<{index_elements}u;++rg_i){{size_t rg_row=(size_t)rg_idx[rg_i];for(size_t rg_k=0;rg_k<{columns}u;++rg_k){{size_t rg_block=rg_row*{blocks_per_row}u+rg_k/{}u;{block_value}rg_o[rg_i*{columns}u+rg_k]=rg_v;}}}}return 0;}}\n",
            plan.weight_desc.block_elements,
        ),
    ]
    .concat();
    let cache_key = native_cache_key(&plan.cache_key.to_string(), &source);
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
    let homogeneous_data = match &plan.kind {
        crate::MovementKernelKind::AffineCopy { input, .. } => input.dtype == plan.dtype,
        crate::MovementKernelKind::Pad { input, .. } => input.dtype == plan.dtype,
        crate::MovementKernelKind::Concat { inputs, .. } => {
            inputs.iter().all(|operand| operand.dtype == plan.dtype)
        }
        crate::MovementKernelKind::Gather { input, .. } => input.dtype == plan.dtype,
        crate::MovementKernelKind::Scatter { base, updates, .. } => {
            base.dtype == plan.dtype && updates.dtype == plan.dtype
        }
        crate::MovementKernelKind::Bitcast { .. } => true,
        crate::MovementKernelKind::Contiguous { input } => input.dtype == plan.dtype,
    };
    if !homogeneous_data {
        return Err(JitError::Unsupported(
            "native movement requires homogeneous operand and output dtypes".into(),
        ));
    }
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
        pointer_order: (0..buffers.len()).map(KernelPointerAbi::Dense).collect(),
        buffers,
        quantized_buffers: Vec::new(),
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
        crate::MovementKernelKind::AffineCopy { input, view } => {
            let output_len = elements(&plan.output_shape)?;
            if output_len == 0 {
                lines.push("  /* empty affine-copy domain */".into());
            } else {
                lines.push(format!(
                    "  for (size_t rg_i=0; rg_i<{output_len}u; ++rg_i) {{ size_t rg_q=rg_i, rg_offset={}u;",
                    usize::try_from(view.offset).map_err(|_| JitError::Unsupported(
                        "affine-copy offset must be nonnegative".into()
                    ))?
                ));
                for axis in (0..view.logical_shape.rank()).rev() {
                    let dim = view.logical_shape.dims()[axis];
                    let stride = usize::try_from(view.strides[axis]).map_err(|_| {
                        JitError::Unsupported("affine-copy stride must be nonnegative".into())
                    })?;
                    if dim == 0 {
                        return Err(JitError::Unsupported(
                            "nonempty affine-copy cannot have a zero dimension".into(),
                        ));
                    }
                    lines.push(format!(
                        "    size_t rg_c{axis}=rg_q%{dim}u; rg_q/={dim}u; rg_offset+=rg_c{axis}*{stride}u;"
                    ));
                }
                lines.push(format!(
                    "    (({output_ty}*)buffers[{output_slot}])[rg_i] = ((const {output_ty}*)buffers[{}])[rg_offset];",
                    ids[&(input.node.index() as u64)]
                ));
                lines.push("  }".into());
            }
        }
        crate::MovementKernelKind::Pad {
            input,
            padding,
            fill_bits,
        } => {
            let output_len = elements(&plan.output_shape)?;
            if output_len == 0 {
                lines.push("  /* empty pad domain */".into());
            } else {
                lines.push(format!(
                    "  for (size_t rg_i=0; rg_i<{output_len}u; ++rg_i) {{"
                ));
                let mut guards = Vec::new();
                let mut offset = Vec::new();
                for (axis, ((&out_dim, &in_dim), &(before, _))) in plan
                    .output_shape
                    .dims()
                    .iter()
                    .zip(input.shape.dims())
                    .zip(padding)
                    .enumerate()
                {
                    let out_stride = plan.output_shape.dims()[axis + 1..]
                        .iter()
                        .product::<usize>();
                    let in_stride = input.shape.dims()[axis + 1..].iter().product::<usize>();
                    let coord = format!("((rg_i/{out_stride}u)%{out_dim}u)");
                    guards.push(format!("{coord}>={before}u && {coord}-{before}u<{in_dim}u"));
                    if in_dim != 0 {
                        offset.push(format!("({coord}-{before}u)*{in_stride}u"));
                    }
                }
                let guard = if guards.is_empty() {
                    "1".into()
                } else {
                    guards.join(" && ")
                };
                let source = if offset.is_empty() {
                    "0".into()
                } else {
                    offset.join("+")
                };
                lines.push(format!("    if ({guard}) (({output_ty}*)buffers[{output_slot}])[rg_i] = ((const {output_ty}*)buffers[{}])[{source}]; else (({output_ty}*)buffers[{output_slot}])[rg_i] = {};", ids[&(input.node.index() as u64)], movement_fill_literal(plan.dtype, *fill_bits)));
                lines.push("  }".into());
            }
        }
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
        crate::MovementKernelKind::Bitcast { input } => {
            let output_bytes = elements(&plan.output_shape)?
                .checked_mul(plan.dtype.itemsize())
                .ok_or_else(|| JitError::Unsupported("bitcast byte overflow".into()))?;
            if output_bytes == 0 {
                lines.push("  /* empty bitcast domain */".into());
            } else if plan.dtype == DType::Bool {
                lines.push(format!(
                    "  for (size_t rg_i=0; rg_i<{output_bytes}u; ++rg_i) ((uint8_t*)buffers[{output_slot}])[rg_i] = ((const uint8_t*)buffers[{}])[rg_i] != 0;",
                    ids[&(input.node.index() as u64)]
                ));
            } else {
                lines.push(format!(
                    "  memcpy(buffers[{output_slot}], buffers[{}], {output_bytes}u);",
                    ids[&(input.node.index() as u64)]
                ));
            }
        }
        crate::MovementKernelKind::Contiguous { input } => {
            let output_bytes = elements(&plan.output_shape)?
                .checked_mul(plan.dtype.itemsize())
                .ok_or_else(|| JitError::Unsupported("contiguous byte overflow".into()))?;
            if output_bytes == 0 {
                lines.push("  /* empty contiguous domain */".into());
            } else {
                lines.push(format!(
                    "  memcpy(buffers[{output_slot}], buffers[{}], {output_bytes}u);",
                    ids[&(input.node.index() as u64)]
                ));
            }
        }
    }
    lines.push("  return failure[1] ? (int)failure[1] : 0;".into());
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = native_cache_key(&format!("movement-{}", plan.cache_key), &source);
    Ok(RenderedC {
        source,
        source_map: BTreeMap::from([(0, 1)]),
        abi,
        cache_key,
    })
}

fn movement_fill_literal(dtype: DType, bits: u64) -> String {
    match dtype {
        // Movement kernels write narrow storage directly and intentionally do
        // not widen the fill through a floating arithmetic expression.
        DType::F16 | DType::BF16 => format!("((uint16_t)0x{:04x}u)", bits as u16),
        _ => literal_expr(dtype, bits),
    }
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
    if lanes == 0
        || program.main_elements % lanes != 0
        || program.tail_elements >= lanes
        || program.main_elements.checked_add(program.tail_elements) != Some(elements)
    {
        return Err(JitError::Unsupported(
            "invalid portable lane/tail control".into(),
        ));
    }
    let mut lines = vec![
        "#include <stdint.h>".into(), "#include <stddef.h>".into(), "#include <math.h>".into(), "#include <string.h>".into(), "#include <limits.h>".into(),
        format!("/* {RENDERER_VERSION} B2 VectorProgram key={} lanes={} */", program.cache_key, lanes),
        "static int8_t rg_i8(uint8_t x){int8_t r;memcpy(&r,&x,1);return r;} static int16_t rg_i16(uint16_t x){int16_t r;memcpy(&r,&x,2);return r;} static int32_t rg_i32(uint32_t x){int32_t r;memcpy(&r,&x,4);return r;} static int64_t rg_i64(uint64_t x){int64_t r;memcpy(&r,&x,8);return r;}".into(),
        "static void rg_fail(uint64_t*f,uint64_t i,uint64_t c){if(!f[1]||i<f[0]){f[0]=i;f[1]=c;}}".into(),
        "static uint64_t rg_udiv(uint64_t a,uint64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}return a/b;} static uint64_t rg_umod(uint64_t a,uint64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}return a%b;}".into(),
        "static int64_t rg_sdiv(int64_t a,int64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}if(a==INT64_MIN&&b==-1)return INT64_MIN;return a/b;} static int64_t rg_sfdiv(int64_t a,int64_t b,uint64_t i,uint64_t*f){int64_t q=rg_sdiv(a,b,i,f),r;if(!b|| (a==INT64_MIN&&b==-1))return q;r=a%b;return r<0?q-(b>0?1:-1):q;} static int64_t rg_srem(int64_t a,int64_t b,uint64_t i,uint64_t*f){if(!b){rg_fail(f,i,1);return 0;}return(a==INT64_MIN&&b==-1)?0:a%b;} static int64_t rg_smod(int64_t a,int64_t b,uint64_t i,uint64_t*f){int64_t r=rg_srem(a,b,i,f);if(!b||r>=0)return r;return b>0?r+b:r-b;}".into(),
        "static uint64_t rg_shl(uint64_t a,int64_t b,unsigned n,uint64_t i,uint64_t*f){if(b<0||(uint64_t)b>=n){rg_fail(f,i,2);return 0;}return a<<(unsigned)b;} static uint64_t rg_ushr(uint64_t a,int64_t b,unsigned n,uint64_t i,uint64_t*f){if(b<0||(uint64_t)b>=n){rg_fail(f,i,2);return 0;}return a>>(unsigned)b;} static uint64_t rg_sshr(uint64_t a,int64_t b,unsigned n,uint64_t i,uint64_t*f){uint64_t mask=n==64?UINT64_MAX:((UINT64_C(1)<<n)-1),r;if(b<0||(uint64_t)b>=n){rg_fail(f,i,2);return 0;}r=(a&mask)>>(unsigned)b;if(b&&((a>>(n-1))&1))r|=mask^(mask>>((unsigned)b));return r;}".into(),
        "static float rg_f16_to_f32(uint16_t h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,o;if(!e)o=m? s|((uint32_t)(113-__builtin_clz(m))<<23)|((uint32_t)(m<<(126-__builtin_clz(m)))<<13):s;else o=e==31?s|0x7f800000|(m<<13):s|((e+112)<<23)|(m<<13);union{uint32_t u;float f;}v={o};return v.f;} static uint16_t rg_f32_to_f16(float x){union{float f;uint32_t u;}v={x};uint32_t b=v.u,s=(b>>16)&0x8000,e=(b>>23)&255,m=b&0x7fffff;if(e==255)return(uint16_t)(s|0x7c00|(m?((m>>13)|1):0));int q=(int)e-112;if(q<=0){if(q<-10)return(uint16_t)s;uint32_t z=m|0x800000,sh=(uint32_t)(14-q),r=z>>sh,rem=z&((1u<<sh)-1),half=1u<<(sh-1);return(uint16_t)(s+r+(rem>half||(rem==half&&(r&1))));}if(q>=31)return(uint16_t)(s|0x7c00);uint32_t r=m>>13,rem=m&0x1fff;r+=rem>0x1000||(rem==0x1000&&(r&1));if(r==0x400){if(q==30)return(uint16_t)(s|0x7c00);q++;r=0;}return(uint16_t)(s|((uint32_t)q<<10)|r);} static float rg_bf16_to_f32(uint16_t b){union{uint32_t u;float f;}v={(uint32_t)b<<16};return v.f;} static uint16_t rg_f32_to_bf16(float x){union{float f;uint32_t u;}v={x};uint32_t b=v.u,hi=b>>16;if((b&0x7f800000)==0x7f800000&&(b&0x007fffff))return(uint16_t)((hi&0x7f)?hi:(hi|1));return(uint16_t)((b+0x7fff+((b>>16)&1))>>16);}".into(),
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
    let cache_key = native_cache_key(&format!("b1-{}", program.cache_key), &source);
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
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Neg) if ty == DType::Bool => {
                        format!("!{}[l]", a)
                    }
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Neg) if ty.is_float() => {
                        format!("-{}[l]", a)
                    }
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Neg) if ty == DType::Bool => {
                        format!("!{}[l]", a)
                    }
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Abs) if ty.is_float() => {
                        format!("fabs({}[l])", a)
                    }
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Abs)
                        if ty == DType::Bool
                            || matches!(ty.category(), crate::DTypeCategory::Unsigned) =>
                    {
                        format!("{}[l]", a)
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
                let value = if ty == DType::Bool {
                    format!("((uint8_t)(({}[l])!=0))", a)
                } else {
                    format!("(({}){}[l])", ctype(ty), a)
                };
                lines.push(format!(
                    "    {} {}[{}]; for(size_t l=0;l<{}u;l++) {}[l]={};",
                    ctype(ty),
                    d,
                    usize::from(program.lanes),
                    active,
                    d,
                    value
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
            | crate::ReduceKind::Product
            | crate::ReduceKind::Max
            | crate::ReduceKind::Min
    ) {
        return Err(JitError::Unsupported(
            "native C reduction kind is not implemented".into(),
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
    let initial = if matches!(kind, crate::ReduceKind::Max) && out.dtype.is_float() {
        "-INFINITY"
    } else if matches!(kind, crate::ReduceKind::Min) && out.dtype.is_float() {
        "INFINITY"
    } else if matches!(kind, crate::ReduceKind::Product) {
        "1"
    } else {
        "0"
    };
    lines.push(format!("    {acc} rg_acc = {initial};"));
    if matches!(kind, crate::ReduceKind::Max | crate::ReduceKind::Min) {
        // CpuBackend accepts the first stored lane unconditionally. This is
        // observable for a leading NaN and for equal signed-zero lanes.
        lines.push("    int rg_seen = 0;".into());
    }
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
        let value = if value_node.ty().is_some_and(|ty| ty.scalar.is_float8()) {
            float8_decode_expr(
                value_node
                    .ty()
                    .expect("guarded typed reduction lane")
                    .scalar,
                &value,
            )
            .expect("guarded Float8 reduction lane")
        } else {
            value
        };
        if matches!(kind, crate::ReduceKind::Max) && out.dtype.is_float() {
            lines.push(format!(
                "      if (!rg_seen) {{ rg_acc = ({acc})({value}); rg_seen = 1; }} else if (!isnan(({acc})({value})) && !isnan(rg_acc) && ({acc})({value}) > rg_acc) rg_acc = ({acc})({value});"
            ));
        } else if matches!(kind, crate::ReduceKind::Min) && out.dtype.is_float() {
            lines.push(format!(
                "      if (!rg_seen) {{ rg_acc = ({acc})({value}); rg_seen = 1; }} else if (!isnan(({acc})({value})) && !isnan(rg_acc) && ({acc})({value}) < rg_acc) rg_acc = ({acc})({value});"
            ));
        } else if matches!(kind, crate::ReduceKind::Max) {
            lines.push(format!(
                "      if (!rg_seen) {{ rg_acc = ({acc})({value}); rg_seen = 1; }} else if (({acc})({value}) > rg_acc) rg_acc = ({acc})({value});"
            ));
        } else if matches!(kind, crate::ReduceKind::Min) {
            lines.push(format!(
                "      if (!rg_seen) {{ rg_acc = ({acc})({value}); rg_seen = 1; }} else if (({acc})({value}) < rg_acc) rg_acc = ({acc})({value});"
            ));
        } else if out.dtype == DType::Bool {
            let operator = if matches!(kind, crate::ReduceKind::Product) {
                "&&"
            } else {
                "||"
            };
            lines.push(format!(
                "      rg_acc = (uint8_t)(rg_acc {operator} ({value}));"
            ));
        } else if matches!(kind, crate::ReduceKind::Product) {
            lines.push(format!(
                "      rg_acc = ({acc})(rg_acc * ({acc})({value}));"
            ));
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
        dtype if dtype.is_float8() => {
            float8_encode_expr(dtype, &store_value).expect("guarded Float8 reduction output")
        }
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
            UArg::Scalar { dtype, bits } if *dtype == ty && dtype.is_float8() => {
                Ok(format!("((uint8_t)0x{:02x}u)", *bits as u8))
            }
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
            let raw = format!("((uint8_t*)buffers[{}])[{}]", ids[&buffer], off);
            if ty.is_float8() {
                Ok(raw)
            } else if matches!(ty, DType::F16 | DType::BF16) {
                let load = if ty == DType::F16 {
                    "rg_f16_to_f32"
                } else {
                    "rg_bf16_to_f32"
                };
                Ok(format!(
                    "{load}(((uint16_t*)buffers[{}])[{}])",
                    ids[&buffer], off
                ))
            } else {
                Ok(format!(
                    "(({}*)buffers[{}])[{}]",
                    ctype(ty),
                    ids[&buffer],
                    off
                ))
            }
        }
        UOpKind::Cast => {
            let source_dtype = n
                .sources()
                .first()
                .and_then(|source| source.ty())
                .ok_or_else(|| JitError::Unsupported("untyped cast input".into()))?
                .scalar;
            Ok(cast_expression(source_dtype, ty, s(0)?))
        }
        UOpKind::GraphUnary(op) => {
            let input_ty = n
                .sources()
                .first()
                .and_then(|source| source.ty())
                .ok_or_else(|| JitError::Unsupported("untyped unary input".into()))?
                .scalar;
            let a = s(0)?;
            let x = match op {
                crate::UnaryOp::Neg if input_ty.is_float() => format!("-({a})"),
                // Raw GraphUnary Bool negation is storage-level logical-not.
                // Public Graph::neg deliberately uses its own source-literal
                // logical_not composition and does not depend on this arm.
                crate::UnaryOp::Neg if input_ty == DType::Bool => {
                    format!("((uint8_t)!({a}))")
                }
                // Negating exact integer storage must never ask C to negate a
                // signed minimum. Subtract from zero in the corresponding
                // unsigned width, then restore the source storage lane.
                crate::UnaryOp::Neg => {
                    wrap_expr(input_ty, format!("0-({})({a})", unsigned_ctype(input_ty)?))?
                }
                crate::UnaryOp::Abs if input_ty.is_float() => format!("fabs({a})"),
                crate::UnaryOp::Abs if input_ty == DType::Bool => a,
                crate::UnaryOp::Abs
                    if matches!(input_ty.category(), crate::DTypeCategory::Unsigned) =>
                {
                    a
                }
                // Match B2's exact-width signed formula: test the signed
                // lane, do any negation in unsigned storage, and wrap it back
                // so every signed minimum follows wrapping_abs rather than an
                // f64/fabs conversion or signed-overflow UB.
                crate::UnaryOp::Abs
                    if matches!(input_ty.category(), crate::DTypeCategory::Signed) =>
                {
                    let unsigned = unsigned_ctype(input_ty)?;
                    wrap_expr(
                        input_ty,
                        format!("({a})<0 ? 0-({unsigned})({a}) : ({unsigned})({a})"),
                    )?
                }
                // CPU/generic evaluate arithmetic after widening a typed
                // storage lane to f64. Keep that working-value contract for
                // the direct multiply instead of letting C evaluate two
                // float operands before the typed output store.
                crate::UnaryOp::Square if ty.is_float() => {
                    format!("((double)({a}))*((double)({a}))")
                }
                crate::UnaryOp::Square if input_ty == DType::Bool => a,
                // Multiply exact integer lanes in defined unsigned arithmetic,
                // then restore the original storage width. This avoids C's
                // signed-overflow UB and the narrow unsigned integer promotions
                // that would otherwise make U16 multiplication overflow `int`.
                crate::UnaryOp::Square => {
                    wrap_expr(input_ty, format!("((uint64_t)({a}))*((uint64_t)({a}))"))?
                }
                crate::UnaryOp::Relu
                    if input_ty == DType::Bool
                        || matches!(input_ty.category(), crate::DTypeCategory::Unsigned) =>
                {
                    a
                }
                crate::UnaryOp::Relu => format!("(({a})>0?({a}):0)"),
                crate::UnaryOp::Step if input_ty == DType::Bool => a,
                crate::UnaryOp::Step => format!("(({a})>0?1:0)"),
                crate::UnaryOp::Sqrt => format!("sqrt({a})"),
                crate::UnaryOp::Rsqrt => format!("(1.0/sqrt({a}))"),
                crate::UnaryOp::Exp => format!("exp({a})"),
                // The CPU and generic evaluators both promote a storage lane
                // to f64 before evaluating Exp2, then quantize only at the
                // result boundary.  C11's `exp2` has that same double input
                // contract; the existing narrow-float store helpers below
                // perform the F16/BF16 rounding.  Keep raw exact-dtype Exp2
                // fail-closed by admitting only the floating UOp contract.
                crate::UnaryOp::Exp2 if ty.is_float() => format!("exp2({a})"),
                // As for Exp2, the CPU and generic evaluators perform Log2
                // after widening the stored lane to f64. C11 `log2` keeps
                // that double evaluation, while the established stores below
                // make the F16/BF16/F32 result rounding explicit. The public
                // non-float path has this floating result contract, matching
                // CPU/generic's scalar-to-f64 evaluation.
                crate::UnaryOp::Log2 if ty.is_float() => format!("log2({a})"),
                // CPU and generic evaluate Sin on the widened f64 scalar.
                // C11 `sin` preserves that operation boundary; narrow floats
                // are quantized solely by the established storage stores.
                crate::UnaryOp::Sin if ty.is_float() => format!("sin({a})"),
                crate::UnaryOp::Tan if ty.is_float() => format!("tan({a})"),
                crate::UnaryOp::Cos if ty.is_float() => format!("cos({a})"),
                crate::UnaryOp::Log if ty.is_float() => format!("log({a})"),
                crate::UnaryOp::Sinh if ty.is_float() => format!("sinh({a})"),
                crate::UnaryOp::Cosh if ty.is_float() => format!("cosh({a})"),
                crate::UnaryOp::Tanh if ty.is_float() => format!("tanh({a})"),
                // CpuBackend and the captured interpreter deliberately use
                // tinygrad's A&S 7.1.26 polynomial rather than host libc erf.
                // Keep that same operation order here so F64 native replay is
                // not weakened to a broad approximation tolerance.
                crate::UnaryOp::Erf if ty.is_float() => format!("rg_erf({a})"),
                crate::UnaryOp::Erfc if ty.is_float() => format!("(1.0-rg_erf({a}))"),
                crate::UnaryOp::Asin if ty.is_float() => format!("asin({a})"),
                crate::UnaryOp::Acos if ty.is_float() => format!("acos({a})"),
                crate::UnaryOp::Atan if ty.is_float() => format!("atan({a})"),
                crate::UnaryOp::Asinh if ty.is_float() => format!("asinh({a})"),
                crate::UnaryOp::Acosh if ty.is_float() => format!("acosh({a})"),
                crate::UnaryOp::Atanh if ty.is_float() => format!("atanh({a})"),
                // CPU/generic evaluate floating Trunc after widening the
                // storage lane to f64, then narrow only at the output store.
                // C11's `trunc` has the same double contract and preserves
                // signed zero, NaN, and infinities. Exact Bool/integer Trunc
                // is an identity in the shared evaluator; calling C `trunc`
                // there would introduce a lossy double round-trip for wide
                // integers, so preserve the source storage lane directly.
                crate::UnaryOp::Trunc if ty.is_float() => format!("trunc({a})"),
                crate::UnaryOp::Trunc => a,
                crate::UnaryOp::Floor if input_ty.is_float() => format!("floor({a})"),
                crate::UnaryOp::Floor => a,
                crate::UnaryOp::Ceil if input_ty.is_float() => format!("ceil({a})"),
                crate::UnaryOp::Ceil => a,
                crate::UnaryOp::Round if input_ty.is_float() => {
                    format!("rg_round_ties_even({a})")
                }
                crate::UnaryOp::Round => a,
                crate::UnaryOp::IsNan if ty == DType::Bool && input_ty.is_float() => {
                    format!("((uint8_t)isnan({a}))")
                }
                crate::UnaryOp::IsNan if ty == DType::Bool => "((uint8_t)0)".into(),
                // The raw IsInf public helper has Bool output. C11's
                // type-generic predicate precisely recognizes both floating
                // infinities and excludes NaN/finite/signed-zero lanes. Do
                // not pass exact Bool/integer storage through it: source and
                // CPU/generic semantics make those lanes deterministically
                // false without a lossy wide-integer conversion.
                crate::UnaryOp::IsInf if ty == DType::Bool && input_ty.is_float() => {
                    format!("((uint8_t)isinf({a}))")
                }
                crate::UnaryOp::IsInf if ty == DType::Bool => "((uint8_t)0)".into(),
                crate::UnaryOp::IsFinite if ty == DType::Bool && input_ty.is_float() => {
                    format!("((uint8_t)isfinite({a}))")
                }
                crate::UnaryOp::IsFinite if ty == DType::Bool => "((uint8_t)1)".into(),
                // tinygrad Sign is `ne(0).where(lt(0).where(-1, 1), 0)`.
                // Keep its ordered comparisons: NaN is nonzero but unordered
                // and therefore +1, while either signed zero takes the
                // canonical positive zero branch. Integer branches avoid
                // arithmetic so signed minima never overflow.
                crate::UnaryOp::Sign if ty == DType::Bool => {
                    format!("((uint8_t)(({a})!=0))")
                }
                crate::UnaryOp::Sign if matches!(ty.category(), crate::DTypeCategory::Unsigned) => {
                    format!("(({a})==0?0:1)")
                }
                crate::UnaryOp::Sign if matches!(ty.category(), crate::DTypeCategory::Signed) => {
                    format!("(({a})<0?-1:(({a})>0?1:0))")
                }
                crate::UnaryOp::Sign if ty.is_float() => {
                    format!("(({a})==0.0?0.0:(({a})<0.0?-1.0:1.0))")
                }
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
                crate::BinaryOp::Div
                | crate::BinaryOp::FloorDiv
                | crate::BinaryOp::TruncDiv
                | crate::BinaryOp::Mod
                | crate::BinaryOp::FMod
                    if !ty.is_float() =>
                {
                    return Ok(int_call(*op, ty, &a, &b));
                }
                crate::BinaryOp::Shl if !ty.is_float() => {
                    return Ok(format!(
                        "(({})rg_shl((uint64_t)({a}),(int64_t)({b}),{},rg_i,failure)",
                        ctype(ty),
                        ty.bits()
                    ));
                }
                crate::BinaryOp::Shr if !ty.is_float() => {
                    if matches!(ty.category(), crate::DTypeCategory::Signed) {
                        return Ok(format!(
                            "(({})rg_sshr((uint64_t)({a}),(int64_t)({b}),{},rg_i,failure))",
                            ctype(ty),
                            ty.bits()
                        ));
                    }
                    return Ok(format!(
                        "(({})rg_shr((uint64_t)({a}),(int64_t)({b}),{},rg_i,failure)",
                        ctype(ty),
                        ty.bits()
                    ));
                }
                crate::BinaryOp::Div => "/",
                crate::BinaryOp::FloorDiv if ty.is_float() => {
                    return Ok(format!("floor(((double)({a}))/((double)({b})))"));
                }
                crate::BinaryOp::TruncDiv if ty.is_float() => {
                    return Ok(format!("trunc(((double)({a}))/((double)({b})))"));
                }
                crate::BinaryOp::Mod if ty.is_float() => {
                    return Ok(format!(
                        "(((double)({a}))-floor(((double)({a}))/((double)({b})))*((double)({b})))"
                    ));
                }
                crate::BinaryOp::FMod if ty.is_float() => {
                    return Ok(format!("fmod((double)({a}),(double)({b}))"));
                }
                crate::BinaryOp::BitAnd => "&",
                crate::BinaryOp::BitOr => "|",
                crate::BinaryOp::BitXor => "^",
                crate::BinaryOp::Maximum => return Ok(format!("(({a})<({b})?({b}):({a}))")),
                crate::BinaryOp::Minimum => return Ok(format!("(({a})>({b})?({b}):({a}))")),
                _ => return Err(JitError::Unsupported(format!("binary {op:?}"))),
            };
            if ty.is_float() {
                // A typed Cast rounds its source value before this operation,
                // but CPU/generic then evaluate the ALU at f64 and narrow
                // only at this result's storage boundary.
                Ok(format!("(((double)({a})) {x} ((double)({b})))"))
            } else {
                Ok(format!("(({a}) {x} ({b}))"))
            }
        }
        UOpKind::GraphLogical(op) => {
            if ty != DType::Bool {
                return Err(JitError::Unsupported("logical output is not Bool".into()));
            }
            let a = s(0)?;
            match op {
                crate::LogicalOp::Not => Ok(format!("((uint8_t)!({a}))")),
                crate::LogicalOp::And => {
                    let b = s(1)?;
                    Ok(format!("((uint8_t)(({a}) && ({b})))"))
                }
                crate::LogicalOp::Or => {
                    let b = s(1)?;
                    Ok(format!("((uint8_t)(({a}) || ({b})))"))
                }
            }
        }
        UOpKind::GraphCompare(op) => {
            let (mut a, mut b) = (s(0)?, s(1)?);
            let source_dtype = n.sources().first().and_then(|source| source.ty());
            if source_dtype.is_some_and(|ty| ty.scalar.is_float8()) {
                let dtype = source_dtype.expect("guarded Float8 source").scalar;
                a = float8_decode_expr(dtype, &a).expect("guarded Float8 dtype");
                b = float8_decode_expr(dtype, &b).expect("guarded Float8 dtype");
                return Ok(match op {
                    crate::CompareOp::Eq => format!("(({a}) == ({b}))"),
                    crate::CompareOp::Ne => format!("(({a}) != ({b}))"),
                    crate::CompareOp::Lt => format!("(({a}) < ({b}))"),
                    crate::CompareOp::Le => format!("(!(({b}) < ({a})))"),
                    crate::CompareOp::Gt => format!("(({b}) < ({a}))"),
                    crate::CompareOp::Ge => format!("(!(({a}) < ({b})))"),
                });
            }
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
fn int_call(op: crate::BinaryOp, ty: DType, a: &str, b: &str) -> String {
    let signed = matches!(ty.category(), crate::DTypeCategory::Signed);
    let helper = match (op, signed) {
        (crate::BinaryOp::Div | crate::BinaryOp::TruncDiv, true) => "rg_sdiv",
        (crate::BinaryOp::Div | crate::BinaryOp::FloorDiv | crate::BinaryOp::TruncDiv, false) => {
            "rg_udiv"
        }
        (crate::BinaryOp::FloorDiv, true) => "rg_sfdiv",
        (crate::BinaryOp::Mod, true) => "rg_smod",
        (crate::BinaryOp::FMod, true) => "rg_srem",
        (crate::BinaryOp::Mod | crate::BinaryOp::FMod, false) => "rg_umod",
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
fn float8_decode_expr(dtype: DType, raw: &str) -> Option<String> {
    let (bias, mantissa_bits, mode) = match dtype {
        DType::F8E4M3 => (7, 3, 0),
        DType::F8E5M2 => (15, 2, 1),
        DType::F8E4M3FNUZ => (8, 3, 2),
        DType::F8E5M2FNUZ => (16, 2, 2),
        _ => return None,
    };
    Some(format!(
        "rg_f8_decode(({raw}),{bias},{mantissa_bits}u,{mode}u)"
    ))
}
fn float8_encode_expr(dtype: DType, value: &str) -> Option<String> {
    let (bias, significand_bits, mode, min_half, overflow, max_normal, min_normal) = match dtype {
        DType::F8E4M3 => (
            7,
            4,
            0,
            0x3F50_0000_0000_0000u64,
            0x407D_0000_0000_0000u64,
            0x7eu8,
            0x3F90_0000_0000_0000u64,
        ),
        DType::F8E5M2 => (
            15,
            3,
            1,
            0x3EE0_0000_0000_0000u64,
            0x40ED_FFFF_FFFF_FFFFu64,
            0x7bu8,
            0x3F10_0000_0000_0000u64,
        ),
        DType::F8E4M3FNUZ => (
            8,
            4,
            2,
            0x3F40_0000_0000_0000u64,
            0x406E_FFFF_FFFF_FFFFu64,
            0x7fu8,
            0x3F80_0000_0000_0000u64,
        ),
        DType::F8E5M2FNUZ => (
            16,
            3,
            2,
            0x3ED0_0000_0000_0000u64,
            0x40ED_FFFF_FFFF_FFFFu64,
            0x7fu8,
            0x3F00_0000_0000_0000u64,
        ),
        _ => return None,
    };
    Some(format!(
        "rg_f8_encode((double)({value}),{bias},{significand_bits}u,{mode}u,UINT64_C(0x{min_half:016x}),UINT64_C(0x{overflow:016x}),0x{max_normal:02x}u,UINT64_C(0x{min_normal:016x}))"
    ))
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
fn view_offset(view: &crate::AffineView, logical: &str) -> String {
    let mut terms = vec![format!("(int64_t){}", view.offset)];
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
            terms.push(format!(
                "((int64_t)((({logical})/{divisor}u)%{dimension}u)*{} )",
                stride
            ));
        }
    }
    format!("({})", terms.join("+"))
}
fn ctype(d: DType) -> &'static str {
    if d.is_float8() {
        return "uint8_t";
    }
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
        DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ => "uint8_t",
    }
}
fn expr_ctype(d: DType) -> &'static str {
    match d {
        DType::F16 | DType::BF16 => "float",
        _ => ctype(d),
    }
}
/// A fused narrow CAST has a typed storage boundary. C `float` alone is not
/// that boundary, so explicitly encode then decode F16/BF16 before a later
/// expression consumes the value. The helpers use raw payload-aware encoding.
fn cast_expression(source_dtype: DType, dtype: DType, value: String) -> String {
    if source_dtype == dtype && dtype.is_float8() {
        return value;
    }
    let value = if source_dtype.is_float8() {
        float8_decode_expr(source_dtype, &value).expect("guarded Float8 source dtype")
    } else {
        value
    };
    match dtype {
        dtype if dtype.is_float8() => {
            float8_encode_expr(dtype, &value).expect("guarded Float8 target dtype")
        }
        DType::Bool => format!("((uint8_t)(({value})!=0))"),
        DType::F16 => format!("rg_f16_to_f32(rg_f32_to_f16((float)({value})))"),
        DType::BF16 => format!("rg_bf16_to_f32(rg_f32_to_bf16((float)({value})))"),
        _ => format!("(({})({value}))", expr_ctype(dtype)),
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
/// Stable identity for one durable C11 shared-library artifact. The compiler
/// command and every fixed flag are part of the key rather than an unstated
/// property of the temporary cache directory.
fn native_cache_key(discriminator: &str, source: &str) -> String {
    let flags = C11_COMPILER_FLAGS.join("\u{1f}");
    key(&format!(
        "{RENDERER_VERSION}\u{1f}{ABI_VERSION}\u{1f}{C11_COMPILER_COMMAND}\u{1f}{flags}\u{1f}{}\u{1f}{}\u{1f}{discriminator}\u{1f}{source}",
        std::env::consts::ARCH,
        std::env::consts::OS,
    ))
}
fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("rustgrad-cpu-jit-v1")
}
static COMPILE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static COMPILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
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
    match fs::symlink_metadata(&lib) {
        Ok(metadata) if metadata.file_type().is_file() => return Ok(lib),
        Ok(_) => {
            return Err(JitError::Io(format!(
                "CPU JIT cache entry is not a regular file: {}",
                lib.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(JitError::Io(error.to_string())),
    }
    let sequence = COMPILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let stem = format!(".{}-{}-{sequence}", r.cache_key, std::process::id());
    let source = d.join(format!("{stem}.c"));
    let temp = d.join(format!("{stem}.tmp"));
    let result = (|| {
        fs::write(&source, &r.source).map_err(|e| JitError::Io(e.to_string()))?;
        let out = Command::new(C11_COMPILER_COMMAND)
            .args(C11_COMPILER_FLAGS)
            .arg("-o")
            .arg(&temp)
            .arg(&source)
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
        fs::File::open(&temp)
            .and_then(|file| file.sync_all())
            .map_err(|e| JitError::Io(e.to_string()))?;
        match fs::rename(&temp, &lib) {
            Ok(()) => Ok(lib),
            Err(error) => match fs::symlink_metadata(&lib) {
                // Another process may have compiled this exact
                // content-addressed key while this temporary artifact was
                // being built. Its completed regular file is a valid hit.
                Ok(metadata) if metadata.file_type().is_file() => Ok(lib),
                _ => Err(JitError::Io(error.to_string())),
            },
        }
    })();
    let _ = fs::remove_file(&source);
    let _ = fs::remove_file(&temp);
    result
}
fn evict_cached_library(path: &Path) -> Result<(), JitError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| JitError::Io(error.to_string()))?;
    if !metadata.file_type().is_file() {
        return Err(JitError::Io(format!(
            "CPU JIT cache entry is not a regular file: {}",
            path.display()
        )));
    }
    fs::remove_file(path).map_err(|error| JitError::Io(error.to_string()))
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
fn load_library_call(
    path: &Path,
) -> Result<
    (
        Arc<Library>,
        unsafe extern "C" fn(*mut *mut c_void, *const i64, *mut u64) -> c_int,
    ),
    JitError,
> {
    let lib = Arc::new(Library::open(path)?);
    let call = unsafe { lib.symbol(b"rustgrad_kernel\0")? };
    Ok((lib, call))
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
    use crate::{
        Backend, CompareOp, CpuBackend, Graph, Op, Scalar, Shape, Storage, SymbolicExpr, TensorData,
    };
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn jit_buffer_round_trips_raw_float8_storage() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let format = dtype.float8_format().unwrap();
            let raw = vec![0x00, 0x80, 0x7f, 0xff];
            let input = TensorData::from_storage(
                [2, 2],
                Storage::Float8(crate::Float8Storage::from_raw(format, raw.clone())),
            )
            .unwrap();
            let output = JitBuffer::from_tensor(&input, true)
                .into_tensor(Shape::from([2, 2]))
                .unwrap();
            let Storage::Float8(output) = output.storage() else {
                panic!("expected Float8 storage for {dtype:?}");
            };
            assert_eq!(output.format(), format);
            assert_eq!(output.as_raw(), raw, "{dtype:?}");
        }
    }

    #[test]
    fn computed_affine_copy_renderer_closes_loop_and_kernel() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let producer = graph.relu(input).unwrap();
        let viewed = graph.reshape(producer, [1, 4]).unwrap();
        let plan = crate::MovementKernelPlan::from_computed_affine_view(&graph, viewed).unwrap();
        let rendered = render_movement(&plan).unwrap();

        assert!(rendered.source.contains("for (size_t rg_i=0;"));
        assert_eq!(
            rendered.source.bytes().filter(|byte| *byte == b'{').count(),
            rendered.source.bytes().filter(|byte| *byte == b'}').count()
        );
        assert!(
            rendered
                .source
                .ends_with("  return failure[1] ? (int)failure[1] : 0;\n}\n")
        );
    }

    #[test]
    fn bitcast_renderer_copies_raw_bytes_and_normalizes_bool_storage() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 8], DType::U8);
        let output = graph.bitcast(input, DType::U32).unwrap();
        let plan = crate::MovementKernelPlan::from_graph(&graph, output).unwrap();
        let rendered = render_movement(&plan).unwrap();
        assert!(
            rendered
                .source
                .contains("memcpy(buffers[1], buffers[0], 16u);")
        );
        assert!(!rendered.source.contains("!= 0"));
        assert_eq!(rendered.abi.buffers[0].dtype, DType::U8);
        assert_eq!(rendered.abi.buffers[0].elements, 16);
        assert_eq!(rendered.abi.buffers[1].dtype, DType::U32);
        assert_eq!(rendered.abi.buffers[1].elements, 4);

        let mut bool_graph = Graph::new();
        let input = bool_graph.input_dtype("input", [4], DType::U8);
        let output = bool_graph.bitcast(input, DType::Bool).unwrap();
        let plan = crate::MovementKernelPlan::from_graph(&bool_graph, output).unwrap();
        let rendered = render_movement(&plan).unwrap();
        assert!(rendered.source.contains("[rg_i] != 0;"));
        assert!(!rendered.source.contains("memcpy("));

        let mut empty_graph = Graph::new();
        let input = empty_graph.input_dtype("input", [3, 0], DType::F16);
        let output = empty_graph.bitcast(input, DType::U8).unwrap();
        let plan = crate::MovementKernelPlan::from_graph(&empty_graph, output).unwrap();
        assert!(
            render_movement(&plan)
                .unwrap()
                .source
                .contains("empty bitcast domain")
        );
    }

    #[test]
    fn contiguous_renderer_emits_owned_raw_copy_and_empty_noop() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [4], DType::F32);
        let computed = graph.cast(input, DType::F32).unwrap();
        let output = graph.contiguous(computed).unwrap();
        let plan = crate::MovementKernelPlan::from_graph(&graph, output).unwrap();
        let rendered = render_movement(&plan).unwrap();
        assert!(matches!(
            &plan.kind,
            crate::MovementKernelKind::Contiguous { input: operand }
                if operand.node == computed && operand.dtype == DType::F32
        ));
        assert!(
            rendered
                .source
                .contains("memcpy(buffers[1], buffers[0], 16u);")
        );
        assert_eq!(rendered.abi.buffers[0].dtype, DType::F32);
        assert_eq!(rendered.abi.buffers[1].dtype, DType::F32);

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [3, 0], DType::BF16);
        let computed = empty.cast(input, DType::BF16).unwrap();
        let output = empty.contiguous(computed).unwrap();
        let plan = crate::MovementKernelPlan::from_graph(&empty, output).unwrap();
        assert!(
            render_movement(&plan)
                .unwrap()
                .source
                .contains("empty contiguous domain")
        );
    }

    #[test]
    fn extrema_render_as_ordered_selects_not_host_intrinsics() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", Shape::from([1]), DType::F32);
        let rhs = graph.input_dtype("rhs", Shape::from([1]), DType::F32);
        let maximum = graph.maximum(lhs, rhs).unwrap();
        let minimum = graph.minimum(lhs, rhs).unwrap();
        let maximum = CpuJit::render(&crate::lower_graph_elementwise(&graph, maximum).unwrap())
            .unwrap()
            .source;
        let minimum = CpuJit::render(&crate::lower_graph_elementwise(&graph, minimum).unwrap())
            .unwrap()
            .source;
        assert!(maximum.contains("?"));
        assert!(maximum.contains("<"));
        assert!(minimum.contains("?"));
        assert!(minimum.contains(">"));
        assert!(!maximum.contains("fmax"));
        assert!(!minimum.contains("fmin"));
    }

    #[test]
    fn float8_compare_decodes_select_preserves_raw_and_other_alu_fails_closed() {
        let operations = [
            CompareOp::Eq,
            CompareOp::Ne,
            CompareOp::Lt,
            CompareOp::Le,
            CompareOp::Gt,
            CompareOp::Ge,
        ];
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let format = dtype.float8_format().unwrap();
            let nan = if matches!(dtype, DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) {
                0x80
            } else {
                0x7f
            };
            let lhs_values = TensorData::from_storage(
                [6],
                Storage::Float8(crate::Float8Storage::from_raw(
                    format,
                    vec![
                        format.encode(-2.0),
                        format.encode(-0.0),
                        format.encode(0.5),
                        format.encode(2.0),
                        nan,
                        format.encode(4.0),
                    ],
                )),
            )
            .unwrap();
            let rhs_values = TensorData::from_storage(
                [6],
                Storage::Float8(crate::Float8Storage::from_raw(
                    format,
                    vec![
                        format.encode(-1.0),
                        format.encode(0.0),
                        format.encode(1.0),
                        format.encode(1.0),
                        format.encode(0.0),
                        format.encode(4.0),
                    ],
                )),
            )
            .unwrap();

            for op in operations {
                let mut graph = Graph::new();
                let lhs = graph.input_dtype("lhs", [6], dtype);
                let rhs = graph.input_dtype("rhs", [6], dtype);
                let output = graph.compare(op, lhs, rhs).unwrap();
                let bindings = HashMap::from([
                    ("lhs".into(), lhs_values.clone()),
                    ("rhs".into(), rhs_values.clone()),
                ]);
                let expected = CpuBackend.execute(&graph, output, &bindings).unwrap();
                let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
                let scalar_rendered = CpuJit::render(&uop).unwrap();
                let vector_rendered = CpuJit::render_vectorized(&uop).unwrap();
                assert!(scalar_rendered.source.matches("rg_f8_decode(").count() > 1);
                assert!(vector_rendered.source.matches("rg_f8_decode(").count() > 1);
                assert!(!vector_rendered.source.contains("B2 VectorProgram"));

                for kernel in [
                    CpuJit::compile(&uop).unwrap(),
                    CpuJit::compile_vectorized(&uop).unwrap(),
                ] {
                    let mut buffers = [
                        JitBuffer::from_tensor(&lhs_values, false),
                        JitBuffer::from_tensor(&rhs_values, false),
                        JitBuffer::zeroed(DType::Bool, expected.len(), true),
                    ];
                    kernel.call(&mut buffers, &[]).unwrap();
                    let actual = buffers[2]
                        .clone()
                        .into_tensor(expected.shape().clone())
                        .unwrap();
                    assert_eq!(actual.storage(), expected.storage(), "{dtype:?} {op:?}");
                }
            }

            let condition_values = TensorData::from_storage(
                [6],
                Storage::Bool(vec![true, false, true, false, true, false]),
            )
            .unwrap();
            let mut graph = Graph::new();
            let condition = graph.input_dtype("condition", [6], DType::Bool);
            let on_true = graph.input_dtype("on_true", [6], dtype);
            let on_false = graph.input_dtype("on_false", [], dtype);
            let selected = graph.select(condition, on_true, on_false).unwrap();
            let false_values = TensorData::from_storage(
                [],
                Storage::Float8(crate::Float8Storage::from_raw(format, vec![0xa5])),
            )
            .unwrap();
            let bindings = HashMap::from([
                ("condition".into(), condition_values.clone()),
                ("on_true".into(), lhs_values.clone()),
                ("on_false".into(), false_values.clone()),
            ]);
            let expected = CpuBackend.execute(&graph, selected, &bindings).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, selected).unwrap();
            let scalar_rendered = CpuJit::render(&uop).unwrap();
            let vector_rendered = CpuJit::render_vectorized(&uop).unwrap();
            // The shared helper definition is present in every elementwise C
            // unit; raw Select must not emit an invocation in the kernel.
            assert_eq!(scalar_rendered.source.matches("rg_f8_decode(").count(), 1);
            assert_eq!(vector_rendered.source.matches("rg_f8_decode(").count(), 1);
            assert!(!vector_rendered.source.contains("B2 VectorProgram"));
            for kernel in [
                CpuJit::compile(&uop).unwrap(),
                CpuJit::compile_vectorized(&uop).unwrap(),
            ] {
                let mut buffers = [
                    JitBuffer::from_tensor(&condition_values, false),
                    JitBuffer::from_tensor(&lhs_values, false),
                    JitBuffer::from_tensor(&false_values, false),
                    JitBuffer::zeroed(dtype, expected.len(), true),
                ];
                kernel.call(&mut buffers, &[]).unwrap();
                let actual = buffers[3]
                    .clone()
                    .into_tensor(expected.shape().clone())
                    .unwrap();
                assert_eq!(actual.storage(), expected.storage(), "{dtype:?} raw Select");
            }

            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1], dtype);
            let rhs = graph.input_dtype("rhs", [1], dtype);
            let output = graph.binary(crate::BinaryOp::Add, lhs, rhs).unwrap();
            let error = CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap_err();
            assert!(
                matches!(&error, JitError::Unsupported(reason)
                    if reason == "native Float8 elementwise supports comparisons, raw selection, and typed casts only"),
                "{dtype:?}: {error:?}"
            );

            let mut mixed = Graph::new();
            let lhs = mixed.input_dtype("lhs", [1], dtype);
            let rhs = mixed.input_dtype("rhs", [1], DType::F16);
            let output = mixed.compare(CompareOp::Eq, lhs, rhs).unwrap();
            let error = CpuJit::render(&crate::lower_graph_elementwise(&mixed, output).unwrap())
                .unwrap_err();
            assert!(
                matches!(&error, JitError::Unsupported(reason)
                    if reason == "native Float8 comparison requires one homogeneous format"),
                "{dtype:?}: {error:?}"
            );
        }
    }

    #[test]
    fn float8_casts_use_exact_native_codecs_and_preserve_same_format_bytes() {
        assert_eq!(RENDERER_VERSION, "rustgrad-c11-scalar-v27");

        let execute = |graph: &Graph,
                       output,
                       input: &TensorData,
                       expected: &TensorData,
                       source: DType,
                       target: DType| {
            let uop = crate::lower_graph_elementwise(graph, output).unwrap();
            let scalar = CpuJit::render(&uop).unwrap();
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(!vector.source.contains("B2 VectorProgram"));
            if source.is_float8() && source != target {
                assert!(scalar.source.matches("rg_f8_decode(").count() > 1);
            }
            if target.is_float8() && source != target {
                assert!(scalar.source.matches("rg_f8_encode(").count() > 1);
            }
            for kernel in [
                CpuJit::compile(&uop).unwrap(),
                CpuJit::compile_vectorized(&uop).unwrap(),
            ] {
                let mut buffers = [
                    JitBuffer::from_tensor(input, false),
                    JitBuffer::zeroed(target, expected.len(), true),
                ];
                kernel.call(&mut buffers, &[]).unwrap();
                let actual = buffers[1]
                    .clone()
                    .into_tensor(expected.shape().clone())
                    .unwrap();
                assert_eq!(
                    actual.storage(),
                    expected.storage(),
                    "{source:?} -> {target:?}"
                );
            }
        };

        let codec_values = TensorData::from_storage(
            [12],
            Storage::F64(vec![
                0.0,
                -0.0,
                0.000_976_562_5,
                0.001_953_125,
                1.0625,
                1.1875,
                240.0,
                448.0,
                57_344.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ]),
        )
        .unwrap();

        for (index, dtype) in DType::FP8S.into_iter().enumerate() {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", codec_values.shape().clone(), DType::F64);
            let output = graph.cast(input, dtype).unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), codec_values.clone())]),
                )
                .unwrap();
            execute(&graph, output, &codec_values, &expected, DType::F64, dtype);

            let format = dtype.float8_format().unwrap();
            let raw = TensorData::from_storage(
                [8],
                Storage::Float8(crate::Float8Storage::from_raw(
                    format,
                    vec![0x00, 0x80, 0x01, 0x38, 0x7e, 0x7f, 0xa5, 0xff],
                )),
            )
            .unwrap();
            let mut identity_graph = Graph::new();
            let input = identity_graph.input_dtype("input", [8], dtype);
            let output = identity_graph.cast(input, dtype).unwrap();
            let expected = CpuBackend
                .execute(
                    &identity_graph,
                    output,
                    &HashMap::from([("input".into(), raw.clone())]),
                )
                .unwrap();
            assert_eq!(expected.storage(), raw.storage());
            execute(&identity_graph, output, &raw, &expected, dtype, dtype);

            let finite = TensorData::from_scalars(
                [6],
                dtype,
                [
                    Scalar::F(0.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.001_953_125),
                    Scalar::F(1.0),
                    Scalar::F(2.0),
                    Scalar::F(16.0),
                ],
            )
            .unwrap();
            for target in [DType::F64, DType::FP8S[(index + 1) % DType::FP8S.len()]] {
                let mut graph = Graph::new();
                let input = graph.input_dtype("input", [6], dtype);
                let output = graph.cast(input, target).unwrap();
                let expected = CpuBackend
                    .execute(
                        &graph,
                        output,
                        &HashMap::from([("input".into(), finite.clone())]),
                    )
                    .unwrap();
                execute(&graph, output, &finite, &expected, dtype, target);
            }
        }
    }

    #[test]
    fn public_logical_not_keeps_cast_then_ne_and_renders_bool_truthiness() {
        for dtype in [DType::F32, DType::I32] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([4]), dtype);
            let output = graph.logical_not(input).unwrap();
            let Op::Compare {
                op: CompareOp::Ne,
                lhs,
                ..
            } = graph.op(output).unwrap()
            else {
                panic!("logical_not must retain its source Ne root");
            };
            assert!(matches!(
                graph.op(*lhs).unwrap(),
                Op::Cast {
                    input: cast_input,
                    dtype: DType::Bool,
                } if *cast_input == input
            ));

            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let topological = uop.topological().unwrap();
            assert!(topological.iter().any(|node| {
                matches!(node.kind(), UOpKind::Cast)
                    && node.ty().is_some_and(|ty| ty.scalar == DType::Bool)
            }));
            assert!(
                topological
                    .iter()
                    .any(|node| { matches!(node.kind(), UOpKind::GraphCompare(CompareOp::Ne)) })
            );

            // `!=0` is the source truthiness predicate: it keeps fractional
            // nonzero F32 values, NaNs, and infinities true while both zeros
            // are false, unlike a numeric C cast to uint8_t.
            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains("!=0"), "{dtype:?}");
            assert!(scalar.source.contains("uint8_t"), "{dtype:?}");
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert!(vector.source.contains("!=0"), "{dtype:?}");
        }
    }

    #[test]
    fn raw_graph_unary_neg_abs_keep_exact_integer_storage_and_bool_semantics() {
        assert_eq!(RENDERER_VERSION, "rustgrad-c11-scalar-v27");

        let signed = [
            (DType::I8, "uint8_t", "rg_i8"),
            (DType::I16, "uint16_t", "rg_i16"),
            (DType::I32, "uint32_t", "rg_i32"),
            (DType::I64, "uint64_t", "rg_i64"),
        ];
        for (dtype, unsigned, helper) in signed {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([2]), dtype);
            let neg = graph.unary(crate::UnaryOp::Neg, input).unwrap();
            let abs = graph.unary(crate::UnaryOp::Abs, input).unwrap();
            assert!(matches!(
                graph.op(neg).unwrap(),
                Op::Unary {
                    op: crate::UnaryOp::Neg,
                    ..
                }
            ));
            assert!(matches!(
                graph.op(abs).unwrap(),
                Op::Unary {
                    op: crate::UnaryOp::Abs,
                    ..
                }
            ));

            let neg_uop = crate::lower_graph_elementwise(&graph, neg).unwrap();
            let abs_uop = crate::lower_graph_elementwise(&graph, abs).unwrap();
            let neg_source = CpuJit::render(&neg_uop).unwrap();
            let abs_source = CpuJit::render(&abs_uop).unwrap();
            assert!(neg_source.source.contains(RENDERER_VERSION));
            assert!(
                neg_source.source.contains(&format!("0-({unsigned})")),
                "{dtype:?}"
            );
            assert!(neg_source.source.contains(helper), "{dtype:?}");
            assert!(
                abs_source.source.contains(&format!("<0 ? 0-({unsigned})")),
                "{dtype:?}"
            );
            assert!(abs_source.source.contains(helper), "{dtype:?}");
            assert!(!abs_source.source.contains("fabs("), "{dtype:?}");
            assert_eq!(
                neg_source.cache_key,
                CpuJit::render(&neg_uop).unwrap().cache_key
            );
            let neg_vector = CpuJit::render_vectorized(&neg_uop).unwrap();
            let abs_vector = CpuJit::render_vectorized(&abs_uop).unwrap();
            assert!(neg_vector.source.contains("B2 VectorProgram"));
            assert!(abs_vector.source.contains("B2 VectorProgram"));
            assert!(neg_vector.source.contains("rg_i"), "{dtype:?}");
            assert!(
                abs_vector.source.contains(&format!("<0 ? 0-({unsigned})")),
                "{dtype:?}"
            );
        }

        for dtype in [DType::U8, DType::U16, DType::U32, DType::U64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([2]), dtype);
            let neg = graph.unary(crate::UnaryOp::Neg, input).unwrap();
            let abs = graph.unary(crate::UnaryOp::Abs, input).unwrap();
            let neg_source =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, neg).unwrap()).unwrap();
            let abs_source =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, abs).unwrap()).unwrap();
            assert!(neg_source.source.contains("0-(uint"), "{dtype:?}");
            assert!(!abs_source.source.contains("fabs("), "{dtype:?}");
            let neg_vector =
                CpuJit::render_vectorized(&crate::lower_graph_elementwise(&graph, neg).unwrap())
                    .unwrap();
            let abs_vector =
                CpuJit::render_vectorized(&crate::lower_graph_elementwise(&graph, abs).unwrap())
                    .unwrap();
            assert!(neg_vector.source.contains("B2 VectorProgram"));
            assert!(abs_vector.source.contains("B2 VectorProgram"));
            assert!(neg_vector.source.contains("0-(uint"), "{dtype:?}");
        }

        let mut bool_graph = Graph::new();
        let bool_input = bool_graph.input_dtype("input", Shape::from([2]), DType::Bool);
        let bool_neg = bool_graph.unary(crate::UnaryOp::Neg, bool_input).unwrap();
        let bool_abs = bool_graph.unary(crate::UnaryOp::Abs, bool_input).unwrap();
        let bool_neg_source =
            CpuJit::render(&crate::lower_graph_elementwise(&bool_graph, bool_neg).unwrap())
                .unwrap();
        let bool_abs_source =
            CpuJit::render(&crate::lower_graph_elementwise(&bool_graph, bool_abs).unwrap())
                .unwrap();
        assert!(bool_neg_source.source.contains("((uint8_t)!("));
        assert!(!bool_abs_source.source.contains("fabs("));
        assert!(
            CpuJit::render_vectorized(
                &crate::lower_graph_elementwise(&bool_graph, bool_neg).unwrap()
            )
            .unwrap()
            .source
            .contains("B2 VectorProgram")
        );
        assert!(
            CpuJit::render_vectorized(
                &crate::lower_graph_elementwise(&bool_graph, bool_abs).unwrap()
            )
            .unwrap()
            .source
            .contains("B2 VectorProgram")
        );
        let bool_bindings = HashMap::from([(
            "input".to_string(),
            TensorData::from_storage([2], Storage::Bool(vec![true, false])).unwrap(),
        )]);
        assert_eq!(
            CpuBackend
                .execute(&bool_graph, bool_neg, &bool_bindings)
                .unwrap(),
            TensorData::from_storage([2], Storage::Bool(vec![false, true])).unwrap()
        );
        assert_eq!(
            CpuBackend
                .execute(&bool_graph, bool_abs, &bool_bindings)
                .unwrap(),
            TensorData::from_storage([2], Storage::Bool(vec![true, false])).unwrap()
        );

        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([2]), dtype);
            let neg = graph.unary(crate::UnaryOp::Neg, input).unwrap();
            let abs = graph.unary(crate::UnaryOp::Abs, input).unwrap();
            let neg_uop = crate::lower_graph_elementwise(&graph, neg).unwrap();
            let abs_uop = crate::lower_graph_elementwise(&graph, abs).unwrap();
            assert!(
                CpuJit::render(&neg_uop).unwrap().source.contains("-("),
                "{dtype:?}"
            );
            assert!(
                CpuJit::render(&abs_uop).unwrap().source.contains("fabs("),
                "{dtype:?}"
            );
            let neg_vector = CpuJit::render_vectorized(&neg_uop).unwrap();
            let abs_vector = CpuJit::render_vectorized(&abs_uop).unwrap();
            if matches!(dtype, DType::F16 | DType::BF16) {
                assert!(!neg_vector.source.contains("B2 VectorProgram"));
                assert!(!abs_vector.source.contains("B2 VectorProgram"));
            } else {
                assert!(neg_vector.source.contains("B2 VectorProgram"));
                assert!(abs_vector.source.contains("B2 VectorProgram"));
            }
        }

        // Raw U64 lanes stay integer-width: no f64/fabs path is permitted
        // for values above 2^53, and signed minima retain wrapping behavior.
        for (dtype, input, negated, absolute) in [
            (
                DType::I8,
                TensorData::from_storage([1], Storage::I8(vec![i8::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I8(vec![i8::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I8(vec![i8::MIN])).unwrap(),
            ),
            (
                DType::I16,
                TensorData::from_storage([1], Storage::I16(vec![i16::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I16(vec![i16::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I16(vec![i16::MIN])).unwrap(),
            ),
            (
                DType::I32,
                TensorData::from_storage([1], Storage::I32(vec![i32::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I32(vec![i32::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I32(vec![i32::MIN])).unwrap(),
            ),
            (
                DType::I64,
                TensorData::from_storage([1], Storage::I64(vec![i64::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I64(vec![i64::MIN])).unwrap(),
                TensorData::from_storage([1], Storage::I64(vec![i64::MIN])).unwrap(),
            ),
            (
                DType::U64,
                TensorData::from_storage([1], Storage::U64(vec![(1u64 << 53) + 1])).unwrap(),
                TensorData::from_storage([1], Storage::U64(vec![u64::MAX - (1u64 << 53)])).unwrap(),
                TensorData::from_storage([1], Storage::U64(vec![(1u64 << 53) + 1])).unwrap(),
            ),
        ] {
            let mut graph = Graph::new();
            let source = graph.input_dtype("input", Shape::from([1]), dtype);
            let neg = graph.unary(crate::UnaryOp::Neg, source).unwrap();
            let abs = graph.unary(crate::UnaryOp::Abs, source).unwrap();
            let bindings = HashMap::from([("input".to_string(), input)]);
            assert_eq!(CpuBackend.execute(&graph, neg, &bindings).unwrap(), negated);
            assert_eq!(
                CpuBackend.execute(&graph, abs, &bindings).unwrap(),
                absolute
            );
        }

        // Maxima take the ordinary exact-width path while minima wrap.  In
        // particular this also proves the scalar renderer never needs an
        // f64 detour to represent I64 arithmetic.
        for (dtype, input, negated) in [
            (
                DType::I8,
                TensorData::from_storage([1], Storage::I8(vec![i8::MAX])).unwrap(),
                TensorData::from_storage([1], Storage::I8(vec![-i8::MAX])).unwrap(),
            ),
            (
                DType::I16,
                TensorData::from_storage([1], Storage::I16(vec![i16::MAX])).unwrap(),
                TensorData::from_storage([1], Storage::I16(vec![-i16::MAX])).unwrap(),
            ),
            (
                DType::I32,
                TensorData::from_storage([1], Storage::I32(vec![i32::MAX])).unwrap(),
                TensorData::from_storage([1], Storage::I32(vec![-i32::MAX])).unwrap(),
            ),
            (
                DType::I64,
                TensorData::from_storage([1], Storage::I64(vec![i64::MAX])).unwrap(),
                TensorData::from_storage([1], Storage::I64(vec![-i64::MAX])).unwrap(),
            ),
        ] {
            let mut graph = Graph::new();
            let source = graph.input_dtype("input", Shape::from([1]), dtype);
            let neg = graph.unary(crate::UnaryOp::Neg, source).unwrap();
            let abs = graph.unary(crate::UnaryOp::Abs, source).unwrap();
            let bindings = HashMap::from([("input".to_string(), input.clone())]);
            assert_eq!(CpuBackend.execute(&graph, neg, &bindings).unwrap(), negated);
            assert_eq!(CpuBackend.execute(&graph, abs, &bindings).unwrap(), input);
        }

        for (dtype, input) in [
            (
                DType::F32,
                TensorData::from_storage([3], Storage::F32(vec![-0.0, f32::NAN, f32::INFINITY]))
                    .unwrap(),
            ),
            (
                DType::F64,
                TensorData::from_storage([3], Storage::F64(vec![-0.0, f64::NAN, f64::INFINITY]))
                    .unwrap(),
            ),
        ] {
            let mut graph = Graph::new();
            let source = graph.input_dtype("input", Shape::from([3]), dtype);
            let neg = graph.unary(crate::UnaryOp::Neg, source).unwrap();
            let abs = graph.unary(crate::UnaryOp::Abs, source).unwrap();
            let bindings = HashMap::from([("input".to_string(), input)]);
            let negated = CpuBackend.execute(&graph, neg, &bindings).unwrap();
            let absolute = CpuBackend.execute(&graph, abs, &bindings).unwrap();
            assert!(negated.scalar_at(0).as_f64().is_sign_positive());
            assert!(absolute.scalar_at(0).as_f64().is_sign_positive());
            assert!(negated.scalar_at(1).as_f64().is_nan());
            assert!(absolute.scalar_at(1).as_f64().is_nan());
            assert!(negated.scalar_at(2).as_f64().is_infinite());
            assert!(absolute.scalar_at(2).as_f64().is_infinite());
        }
    }

    #[test]
    fn storage_unaries_use_defined_integer_and_rounding_predicate_paths() {
        let mut exact = Graph::new();
        let i64_input = exact.input_dtype("i64", Shape::from([2]), DType::I64);
        let u16_input = exact.input_dtype("u16", Shape::from([2]), DType::U16);
        for output in [
            exact.unary(crate::UnaryOp::Square, i64_input).unwrap(),
            exact.unary(crate::UnaryOp::Square, u16_input).unwrap(),
        ] {
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&exact, output).unwrap()).unwrap();
            assert!(rendered.source.contains("((uint64_t)("));
            assert!(rendered.source.contains("*((uint64_t)("));
            assert!(!rendered.source.contains("fabs("));
        }

        let mut floating = Graph::new();
        let input = floating.input_dtype("input", Shape::from([7]), DType::F64);
        for (op, token) in [
            (crate::UnaryOp::Floor, "floor("),
            (crate::UnaryOp::Ceil, "ceil("),
            (crate::UnaryOp::Round, "rg_round_ties_even("),
            (crate::UnaryOp::IsNan, "isnan("),
            (crate::UnaryOp::IsInf, "isinf("),
            (crate::UnaryOp::IsFinite, "isfinite("),
        ] {
            let output = floating.unary(op, input).unwrap();
            let uop = crate::lower_graph_elementwise(&floating, output).unwrap();
            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains(token), "{op:?}");
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(vector.source.contains(token), "{op:?}");
            assert!(!vector.source.contains("B2 VectorProgram"), "{op:?}");
        }

        let round = floating.unary(crate::UnaryOp::Round, input).unwrap();
        let round_source =
            CpuJit::render(&crate::lower_graph_elementwise(&floating, round).unwrap())
                .unwrap()
                .source;
        assert!(round_source.contains("frac<0.5"));
        assert!(round_source.contains("fmod(lo,2.0)==0.0"));
        assert!(round_source.contains("copysign(0.0,x)"));

        for (dtype, op, expected) in [
            (DType::I64, crate::UnaryOp::IsNan, "((uint8_t)0)"),
            (DType::U64, crate::UnaryOp::IsInf, "((uint8_t)0)"),
            (DType::Bool, crate::UnaryOp::IsFinite, "((uint8_t)1)"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.unary(op, input).unwrap();
            let source = CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap()
                .source;
            assert!(source.contains(expected), "{dtype:?} {op:?}");
        }
    }

    #[test]
    fn extended_float_unaries_use_c11_math_and_source_erf_polynomial() {
        let operations = [
            (crate::UnaryOp::Sinh, "sinh("),
            (crate::UnaryOp::Cosh, "cosh("),
            (crate::UnaryOp::Tanh, "tanh("),
            (crate::UnaryOp::Erf, "rg_erf("),
            (crate::UnaryOp::Erfc, "1.0-rg_erf("),
            (crate::UnaryOp::Asin, "asin("),
            (crate::UnaryOp::Acos, "acos("),
            (crate::UnaryOp::Atan, "atan("),
            (crate::UnaryOp::Asinh, "asinh("),
            (crate::UnaryOp::Acosh, "acosh("),
            (crate::UnaryOp::Atanh, "atanh("),
        ];
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            for (op, token) in operations {
                let mut graph = Graph::new();
                let input = graph.input_dtype("input", Shape::from([3]), dtype);
                let output = graph.unary(op, input).unwrap();
                let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
                let scalar = CpuJit::render(&uop).unwrap();
                assert!(scalar.source.contains(token), "{dtype:?} {op:?}");
                let vector = CpuJit::render_vectorized(&uop).unwrap();
                assert!(vector.source.contains(token), "{dtype:?} {op:?}");
                assert!(
                    !vector.source.contains("B2 VectorProgram"),
                    "{dtype:?} {op:?}"
                );
            }
        }

        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([1]), DType::F64);
        let output = graph.unary(crate::UnaryOp::Erf, input).unwrap();
        let source = CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap())
            .unwrap()
            .source;
        for token in [
            "0.3275911",
            "1.061405429",
            "-1.453152027",
            "1.421413741",
            "-0.284496736",
            "0.254829592",
            "copysign(1.0,x)",
            "exp((-x)*x)",
        ] {
            assert!(source.contains(token), "missing source erf term {token}");
        }
        assert!(!source.contains("return erf(x)"));
    }

    #[test]
    fn exp2_emits_the_cpu_oracle_double_path_and_preserves_vector_fallback() {
        // Public Exp2 lifts exact storage to F32 and preserves each float
        // storage dtype. The scalar renderer evaluates the same f64 Exp2 as
        // CpuBackend/Kernel, then its existing stores perform narrow rounding.
        for dtype in [
            DType::Bool,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([5]), dtype);
            let output = graph.exp2(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();

            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains("exp2("), "{dtype:?}");
            assert_eq!(scalar.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            // B1 deliberately admits only Neg/Abs unary operations. An Exp2
            // vector request must therefore use the legacy per-lane emitter,
            // which shares the scalar expression and remains deterministic.
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(vector.source.contains("exp2("), "{dtype:?}");
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert_eq!(
                vector.cache_key,
                CpuJit::render_vectorized(&uop).unwrap().cache_key
            );

            // IsFinite remains its source-literal IsInf/IsNaN/logical-not
            // composition for every input storage dtype.
            let finite = graph.isfinite(input).unwrap();
            let finite =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, finite).unwrap()).unwrap();
            assert_eq!(finite.abi.buffers.last().unwrap().dtype, DType::Bool);
            assert!(finite.source.contains("||"), "{dtype:?}");
            assert!(finite.source.contains("!="), "{dtype:?}");
        }

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", Shape::from([1]), DType::F16);
        let bf16 = narrow.input_dtype("bf16", Shape::from([1]), DType::BF16);
        let f16_output = narrow.exp2(f16).unwrap();
        let bf16_output = narrow.exp2(bf16).unwrap();
        let f16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, f16_output).unwrap())
                .unwrap()
                .source;
        let bf16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, bf16_output).unwrap())
                .unwrap()
                .source;
        assert!(f16_source.contains("rg_f32_to_f16(exp2("));
        assert!(bf16_source.contains("rg_f32_to_bf16(exp2("));

        // Exp2 VJPs retain an Exp2 node and typed ln(2) multiplication. The
        // JIT renderer must accept that generated forward subexpression too.
        let mut differentiated = Graph::new();
        let input = differentiated.input_dtype("input", Shape::from([1]), DType::F64);
        let output = differentiated.exp2(input).unwrap();
        let loss = differentiated.sum_all(output).unwrap();
        let gradient = differentiated.grad(loss, input).unwrap();
        let gradient_uop = crate::lower_graph_elementwise(&differentiated, gradient).unwrap();
        assert!(
            CpuJit::render(&gradient_uop)
                .unwrap()
                .source
                .contains("exp2(")
        );
    }

    #[test]
    fn log2_emits_the_cpu_oracle_double_path_and_preserves_vector_fallback() {
        // Public Log2 has the same storage lattice as Exp2: exact inputs have
        // an F32 result contract, floating storage is retained, and the
        // result boundary alone narrows F16/BF16 lanes.
        for dtype in [
            DType::Bool,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([5]), dtype);
            let output = graph.log2(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();

            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains("log2("), "{dtype:?}");
            assert_eq!(scalar.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            // The deliberately narrow B1 ABI does not admit Log2. A vector
            // request must keep using the deterministic per-lane renderer.
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(vector.source.contains("log2("), "{dtype:?}");
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert_eq!(
                vector.cache_key,
                CpuJit::render_vectorized(&uop).unwrap().cache_key
            );
        }

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", Shape::from([1]), DType::F16);
        let bf16 = narrow.input_dtype("bf16", Shape::from([1]), DType::BF16);
        let f16_output = narrow.log2(f16).unwrap();
        let bf16_output = narrow.log2(bf16).unwrap();
        let f16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, f16_output).unwrap())
                .unwrap()
                .source;
        let bf16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, bf16_output).unwrap())
                .unwrap()
                .source;
        assert!(f16_source.contains("rg_f32_to_f16(log2("));
        assert!(bf16_source.contains("rg_f32_to_bf16(log2("));

        // Log2 VJPs carry a source-width ln(2) denominator. Rendering the
        // gradient validates that the Log2 source remains admitted alongside
        // the typed multiply/divide nodes created by autograd.
        let mut differentiated = Graph::new();
        let input = differentiated.input_dtype("input", Shape::from([1]), DType::F64);
        let output = differentiated.log2(input).unwrap();
        let loss = differentiated.sum_all(output).unwrap();
        let gradient = differentiated.grad(loss, input).unwrap();
        let gradient_uop = crate::lower_graph_elementwise(&differentiated, gradient).unwrap();
        assert_eq!(
            CpuJit::render(&gradient_uop)
                .unwrap()
                .abi
                .buffers
                .last()
                .unwrap()
                .dtype,
            DType::F64
        );
    }

    #[test]
    fn sin_emits_the_cpu_oracle_path_and_keeps_b1_fail_closed() {
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([5]), dtype);
            let output = graph.sin(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains("sin("), "{dtype:?}");
            assert_eq!(scalar.cache_key, CpuJit::render(&uop).unwrap().cache_key);
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(vector.source.contains("sin("), "{dtype:?}");
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
        }

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", Shape::from([1]), DType::F16);
        let bf16 = narrow.input_dtype("bf16", Shape::from([1]), DType::BF16);
        let f16_out = narrow.sin(f16).unwrap();
        let bf16_out = narrow.sin(bf16).unwrap();
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, f16_out).unwrap())
                .unwrap()
                .source
                .contains("rg_f32_to_f16(sin(")
        );
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, bf16_out).unwrap())
                .unwrap()
                .source
                .contains("rg_f32_to_bf16(sin(")
        );

        // The source VJP is `sin(pi/2 - x) * upstream`, not a raw Cos node.
        let mut differentiated = Graph::new();
        let input = differentiated.input_dtype("input", Shape::from([1]), DType::F64);
        let output = differentiated.sin(input).unwrap();
        let loss = differentiated.sum_all(output).unwrap();
        let gradient = differentiated.grad(loss, input).unwrap();
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&differentiated, gradient).unwrap())
                .unwrap()
                .source
                .contains("sin(")
        );
    }

    #[test]
    fn trunc_emits_float_c11_and_exact_storage_identity_with_vector_fallback() {
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([5]), dtype);
            let output = graph.trunc(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let scalar = CpuJit::render(&uop).unwrap();
            if dtype.is_float() {
                // C11 `trunc(double)` is the CPU/generic widened operation,
                // including signed zero, finite fractions, NaN, and infinity.
                assert!(scalar.source.contains("trunc("), "{dtype:?}");
            } else {
                // Bool/integer Trunc is storage-exact identity: do not route
                // exact I64/U64 lanes through a lossy floating expression.
                assert!(!scalar.source.contains("trunc("), "{dtype:?}");
            }
            assert_eq!(scalar.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            // B1 intentionally remains limited to Neg/Abs. Trunc therefore
            // takes the deterministic scalar-expression-per-lane fallback.
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            if dtype.is_float() {
                assert!(vector.source.contains("trunc("), "{dtype:?}");
            }
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert_eq!(
                vector.cache_key,
                CpuJit::render_vectorized(&uop).unwrap().cache_key
            );
        }

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", Shape::from([1]), DType::F16);
        let bf16 = narrow.input_dtype("bf16", Shape::from([1]), DType::BF16);
        let f16_output = narrow.trunc(f16).unwrap();
        let bf16_output = narrow.trunc(bf16).unwrap();
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, f16_output).unwrap())
                .unwrap()
                .source
                .contains("rg_f32_to_f16(trunc(")
        );
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, bf16_output).unwrap())
                .unwrap()
                .source
                .contains("rg_f32_to_bf16(trunc(")
        );

        // Floor, Ceil, Round, and float division modes source-literally use
        // Trunc. Their generated graphs must now be admitted without changing
        // any raw non-float division contract.
        let mut composed = Graph::new();
        let lhs = composed.input_dtype("lhs", Shape::from([1]), DType::F64);
        let rhs = composed.input_dtype("rhs", Shape::from([1]), DType::F64);
        for output in [
            composed.floor(lhs).unwrap(),
            composed.ceil(lhs).unwrap(),
            composed.round(lhs).unwrap(),
            composed.trunc_div(lhs, rhs).unwrap(),
            composed.floor_div(lhs, rhs).unwrap(),
        ] {
            assert!(
                CpuJit::render(&crate::lower_graph_elementwise(&composed, output).unwrap())
                    .unwrap()
                    .source
                    .contains("trunc(")
            );
        }

        // Reverse mode for Trunc is the existing explicit zero graph.
        let output = composed.trunc(lhs).unwrap();
        let loss = composed.sum_all(output).unwrap();
        let gradient = composed.grad(loss, lhs).unwrap();
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&composed, gradient).unwrap()).is_ok()
        );
    }

    #[test]
    fn isinf_emits_typed_predicate_and_admits_source_boolean_compositions() {
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([5]), dtype);
            let output = graph.isinf(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let scalar = CpuJit::render(&uop).unwrap();
            assert_eq!(scalar.abi.buffers.last().unwrap().dtype, DType::Bool);
            if dtype.is_float() {
                // C11 isinf recognizes both infinities while excluding finite
                // values, either signed zero, and NaN.
                assert!(scalar.source.contains("(uint8_t)isinf("), "{dtype:?}");
            } else {
                // Exact non-float lanes never take a floating conversion.
                assert!(!scalar.source.contains("isinf("), "{dtype:?}");
            }
            assert_eq!(scalar.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            // B1 remains deliberately limited to Neg/Abs; IsInf follows the
            // deterministic scalar-expression-per-lane vector fallback.
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            if dtype.is_float() {
                assert!(vector.source.contains("(uint8_t)isinf("), "{dtype:?}");
            }
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert_eq!(
                vector.cache_key,
                CpuJit::render_vectorized(&uop).unwrap().cache_key
            );
        }

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", Shape::from([]), DType::F16);
        let bf16 = narrow.input_dtype("bf16", Shape::from([0]), DType::BF16);
        let f16_output = narrow.isinf(f16).unwrap();
        let bf16_output = narrow.isinf(bf16).unwrap();
        let f16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, f16_output).unwrap())
                .unwrap()
                .source;
        let bf16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, bf16_output).unwrap())
                .unwrap()
                .source;
        assert!(f16_source.contains("rg_f16_to_f32"));
        assert!(f16_source.contains("(uint8_t)isinf("));
        assert!(bf16_source.contains("rg_bf16_to_f32"));
        assert!(bf16_source.contains("(uint8_t)isinf("));

        // Default both-sign IsInf and source-literal IsFinite must retain the
        // raw predicate, typed Bool logical-or/not, and no gradient route.
        let mut composed = Graph::new();
        let input = composed.input_dtype("input", Shape::from([1]), DType::F64);
        let both = composed.isinf_with_signs(input, true, true).unwrap();
        let positive = composed.isinf_with_signs(input, true, false).unwrap();
        let finite = composed.isfinite(input).unwrap();
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&composed, both).unwrap())
                .unwrap()
                .source
                .contains("(uint8_t)isinf(")
        );
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&composed, positive).unwrap())
                .unwrap()
                .source
                .contains("==")
        );
        let finite_source =
            CpuJit::render(&crate::lower_graph_elementwise(&composed, finite).unwrap())
                .unwrap()
                .source;
        assert!(finite_source.contains("(uint8_t)isinf("));
        assert!(finite_source.contains("||"));
        assert!(finite_source.contains("!="));
        assert!(matches!(
            composed.grad(both, input),
            Err(crate::Error::NonDifferentiableTarget(node)) if node == both
        ));
    }

    #[test]
    fn sign_emits_tinygrad_ordered_branches_and_keeps_b1_fail_closed() {
        for (dtype, branch) in [
            (DType::Bool, "!=0"),
            (DType::I8, "<0?-1:(("),
            (DType::I16, "<0?-1:(("),
            (DType::I32, "<0?-1:(("),
            (DType::I64, "<0?-1:(("),
            (DType::U8, "==0?0:1"),
            (DType::U16, "==0?0:1"),
            (DType::U32, "==0?0:1"),
            (DType::U64, "==0?0:1"),
            (DType::F16, "==0.0?0.0:"),
            (DType::BF16, "==0.0?0.0:"),
            (DType::F32, "==0.0?0.0:"),
            (DType::F64, "==0.0?0.0:"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([5]), dtype);
            let output = graph.sign(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();

            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains(branch), "{dtype:?}");
            assert_eq!(scalar.cache_key, CpuJit::render(&uop).unwrap().cache_key);
            if dtype.is_float() {
                assert!(scalar.source.contains("<0.0?-1.0:1.0"), "{dtype:?}");
            }

            // The B1 vector ABI deliberately only admits Neg/Abs unary
            // instructions. Sign uses the deterministic scalar expression
            // per lane instead of silently extending that narrower contract.
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(vector.source.contains(branch), "{dtype:?}");
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert_eq!(
                vector.cache_key,
                CpuJit::render_vectorized(&uop).unwrap().cache_key
            );
        }

        let mut narrow = Graph::new();
        let f16 = narrow.input_dtype("f16", Shape::from([1]), DType::F16);
        let bf16 = narrow.input_dtype("bf16", Shape::from([1]), DType::BF16);
        let f16_output = narrow.sign(f16).unwrap();
        let bf16_output = narrow.sign(bf16).unwrap();
        let f16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, f16_output).unwrap())
                .unwrap()
                .source;
        let bf16_source =
            CpuJit::render(&crate::lower_graph_elementwise(&narrow, bf16_output).unwrap())
                .unwrap()
                .source;
        assert!(f16_source.contains("rg_f32_to_f16("));
        assert!(bf16_source.contains("rg_f32_to_bf16("));

        // Public Abs is source-literally `x * sign(x)`, and Sign's VJP is an
        // explicit zero. Both generated graphs must now remain JIT-admitted.
        let mut composed = Graph::new();
        let input = composed.input_dtype("input", Shape::from([1]), DType::F64);
        let absolute = composed.abs(input).unwrap();
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&composed, absolute).unwrap())
                .unwrap()
                .source
                .contains("==0.0?0.0:")
        );
        let output = composed.sign(input).unwrap();
        let loss = composed.sum_all(output).unwrap();
        let gradient = composed.grad(loss, input).unwrap();
        assert!(
            CpuJit::render(&crate::lower_graph_elementwise(&composed, gradient).unwrap()).is_ok()
        );
    }

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
        malformed[2].bytes_mut().fill(0x7a);
        let output_before = malformed[2].bytes().to_vec();
        assert!(matches!(
            k.call(&mut malformed, &[]),
            Err(JitError::InvalidBuffer(_))
        ));
        assert_eq!(malformed[2].bytes(), output_before);
    }

    #[test]
    fn durable_native_key_distinguishes_vector_policy_and_is_deterministic() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([5]), DType::F32);
        let output = graph.square(input).unwrap();
        let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
        let scalar = CpuJit::render(&uop).unwrap();
        let vector = CpuJit::render_vectorized(&uop).unwrap();

        assert_ne!(scalar.cache_key, vector.cache_key);
        assert_eq!(
            vector.cache_key,
            CpuJit::render_vectorized(&uop).unwrap().cache_key
        );
        assert_eq!(
            native_cache_key("b1", "source"),
            native_cache_key("b1", "source")
        );
        assert_ne!(
            native_cache_key("b1", "source"),
            native_cache_key("scalar", "source")
        );
        assert_ne!(
            native_cache_key("b1", "source"),
            native_cache_key("b1", "source+tail")
        );
    }

    #[test]
    fn corrupt_durable_library_is_evicted_and_rebuilt_once() {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::from([1]));
        let output = graph.neg(input).unwrap();
        let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
        let mut rendered = CpuJit::render(&uop).unwrap();
        // Keep this fixture isolated from normal JIT cache keys while using
        // the real temporary directory and publication path.
        rendered.cache_key = format!("{}-corruption-retry", rendered.cache_key);
        let path = compile_cached(&rendered).unwrap();
        std::fs::write(&path, b"not a shared library").unwrap();

        let kernel = JitKernel::load(&rendered).unwrap();
        assert_ne!(std::fs::read(&path).unwrap(), b"not a shared library");
        assert_eq!(compile_cached(&rendered).unwrap(), path);

        drop(kernel);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_log2_matches_cpu_oracle_and_renders_narrow_storage() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let output = graph.log2(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("log2("));
            assert_eq!(rendered.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            let values = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [0.5, 1.0, 8.0].into_iter().map(Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, values.len(), true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            assert_eq!(native.storage(), expected.storage(), "{dtype:?}");
        }

        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.log2(input).unwrap();
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("log2("), "{dtype:?}");
            assert!(rendered.source.contains(decode), "{dtype:?}");
            assert!(rendered.source.contains(encode), "{dtype:?}");
        }
    }

    #[test]
    fn native_exp2_matches_cpu_oracle_and_renders_narrow_storage() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let output = graph.exp2(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("exp2("));
            assert_eq!(rendered.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            let values = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [-1.0, 0.0, 3.0].into_iter().map(Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, values.len(), true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            assert_eq!(native.storage(), expected.storage(), "{dtype:?}");
        }

        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.exp2(input).unwrap();
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("exp2("), "{dtype:?}");
            assert!(rendered.source.contains(decode), "{dtype:?}");
            assert!(rendered.source.contains(encode), "{dtype:?}");
        }
    }

    #[test]
    fn native_sin_matches_cpu_oracle_and_renders_narrow_storage() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let output = graph.sin(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("sin("));
            assert_eq!(rendered.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            let values = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [-1.0, 0.0, 0.5].into_iter().map(Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, values.len(), true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            let tolerance = if dtype == DType::F32 { 1e-6 } else { 2e-15 };
            for index in 0..native.len() {
                assert!(
                    (native.scalar_at(index).as_f64() - expected.scalar_at(index).as_f64()).abs()
                        <= tolerance,
                    "{dtype:?} index={index}"
                );
            }
        }

        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.sin(input).unwrap();
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("sin("), "{dtype:?}");
            assert!(rendered.source.contains(decode), "{dtype:?}");
            assert!(rendered.source.contains(encode), "{dtype:?}");
        }
    }

    #[test]
    fn native_tan_matches_cpu_oracle_and_renders_narrow_storage() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let output = graph.tan(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("sin("));
            assert_eq!(rendered.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            let values = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [-1.0, 0.0, 0.5].into_iter().map(Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, values.len(), true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            let tolerance = if dtype == DType::F32 { 1e-6 } else { 2e-15 };
            for index in 0..native.len() {
                assert!(
                    (native.scalar_at(index).as_f64() - expected.scalar_at(index).as_f64()).abs()
                        <= tolerance,
                    "{dtype:?} index={index}"
                );
            }
        }

        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.tan(input).unwrap();
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("sin("), "{dtype:?}");
            assert!(rendered.source.contains(decode), "{dtype:?}");
            assert!(rendered.source.contains(encode), "{dtype:?}");
        }
    }

    #[test]
    fn native_cos_matches_cpu_oracle_and_renders_narrow_storage() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let output = graph.cos(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("sin("));
            assert_eq!(rendered.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            let values = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [-1.0, 0.0, 0.5].into_iter().map(Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, values.len(), true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            let tolerance = if dtype == DType::F32 { 1e-6 } else { 2e-15 };
            for index in 0..native.len() {
                assert!(
                    (native.scalar_at(index).as_f64() - expected.scalar_at(index).as_f64()).abs()
                        <= tolerance,
                    "{dtype:?} index={index}"
                );
            }
        }

        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.cos(input).unwrap();
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("sin("), "{dtype:?}");
            assert!(rendered.source.contains(decode), "{dtype:?}");
            assert!(rendered.source.contains(encode), "{dtype:?}");
        }
    }

    #[test]
    fn native_log_matches_cpu_oracle_and_renders_narrow_storage() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let output = graph.log(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("log2("));
            assert_eq!(rendered.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            let values = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [0.5, 1.0, 2.0].into_iter().map(Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, values.len(), true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            let tolerance = if dtype == DType::F32 { 1e-6 } else { 2e-15 };
            for index in 0..native.len() {
                assert!(
                    (native.scalar_at(index).as_f64() - expected.scalar_at(index).as_f64()).abs()
                        <= tolerance,
                    "{dtype:?} index={index}"
                );
            }
        }

        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.log(input).unwrap();
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("log2("), "{dtype:?}");
            assert!(rendered.source.contains(decode), "{dtype:?}");
            assert!(rendered.source.contains(encode), "{dtype:?}");
        }
    }

    #[test]
    fn native_trunc_matches_cpu_oracle_and_renders_narrow_storage() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let output = graph.trunc(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("trunc("));
            assert_eq!(rendered.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            let values = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [-1.75, -0.0, 2.5].into_iter().map(Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, values.len(), true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            assert_eq!(native.storage(), expected.storage(), "{dtype:?}");
        }

        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.trunc(input).unwrap();
            let rendered =
                CpuJit::render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("trunc("), "{dtype:?}");
            assert!(rendered.source.contains(decode), "{dtype:?}");
            assert!(rendered.source.contains(encode), "{dtype:?}");
        }
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
        let quotient = div_graph.binary(crate::BinaryOp::Div, n, d).unwrap();
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
    fn raw_division_family_matches_oracle_and_keeps_distinct_signed_semantics() {
        let operations = [
            crate::BinaryOp::Div,
            crate::BinaryOp::FloorDiv,
            crate::BinaryOp::TruncDiv,
            crate::BinaryOp::Mod,
            crate::BinaryOp::FMod,
        ];
        for (dtype, lhs, rhs) in [
            (
                DType::I64,
                TensorData::from_scalars(
                    [5],
                    DType::I64,
                    [
                        Scalar::I(i64::MIN),
                        Scalar::I(-7),
                        Scalar::I(-7),
                        Scalar::I(7),
                        Scalar::I(7),
                    ],
                )
                .unwrap(),
                TensorData::from_scalars(
                    [5],
                    DType::I64,
                    [
                        Scalar::I(-1),
                        Scalar::I(3),
                        Scalar::I(-3),
                        Scalar::I(3),
                        Scalar::I(-3),
                    ],
                )
                .unwrap(),
            ),
            (
                DType::F64,
                TensorData::from_scalars(
                    [4],
                    DType::F64,
                    [
                        Scalar::F(-7.5),
                        Scalar::F(-7.5),
                        Scalar::F(7.5),
                        Scalar::F(7.5),
                    ],
                )
                .unwrap(),
                TensorData::from_scalars(
                    [4],
                    DType::F64,
                    [
                        Scalar::F(3.0),
                        Scalar::F(-3.0),
                        Scalar::F(3.0),
                        Scalar::F(-3.0),
                    ],
                )
                .unwrap(),
            ),
        ] {
            for op in operations {
                let mut graph = Graph::new();
                let lhs_id = graph.input_dtype("lhs", lhs.shape().clone(), dtype);
                let rhs_id = graph.input_dtype("rhs", rhs.shape().clone(), dtype);
                let output = graph.binary(op, lhs_id, rhs_id).unwrap();
                let bindings = HashMap::from([
                    ("lhs".to_string(), lhs.clone()),
                    ("rhs".to_string(), rhs.clone()),
                ]);
                let expected = CpuBackend.execute(&graph, output, &bindings).unwrap();
                let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
                let rendered = CpuJit::render(&uop).unwrap();
                match (dtype, op) {
                    (DType::I64, crate::BinaryOp::FloorDiv) => {
                        assert!(rendered.source.matches("rg_sfdiv(").count() >= 2)
                    }
                    (DType::I64, crate::BinaryOp::Mod) => {
                        assert!(rendered.source.matches("rg_smod(").count() >= 2)
                    }
                    (DType::I64, crate::BinaryOp::FMod) => {
                        assert!(rendered.source.matches("rg_srem(").count() >= 2)
                    }
                    (DType::F64, crate::BinaryOp::FloorDiv) => {
                        assert!(rendered.source.contains("floor(((double)"))
                    }
                    (DType::F64, crate::BinaryOp::TruncDiv) => {
                        assert!(rendered.source.contains("trunc(((double)"))
                    }
                    (DType::F64, crate::BinaryOp::Mod) => {
                        assert!(rendered.source.contains("-floor(((double)"))
                    }
                    (DType::F64, crate::BinaryOp::FMod) => {
                        assert!(rendered.source.contains("fmod((double)"))
                    }
                    _ => {}
                }

                for kernel in [
                    CpuJit::compile(&uop).unwrap(),
                    CpuJit::compile_vectorized(&uop).unwrap(),
                ] {
                    let mut buffers = [
                        JitBuffer::from_tensor(&lhs, false),
                        JitBuffer::from_tensor(&rhs, false),
                        JitBuffer::zeroed(dtype, expected.len(), true),
                    ];
                    kernel.call(&mut buffers, &[]).unwrap();
                    let actual = buffers[2]
                        .clone()
                        .into_tensor(expected.shape().clone())
                        .unwrap();
                    assert_eq!(actual.storage(), expected.storage(), "{dtype:?} {op:?}");
                }
            }
        }

        // Every exact integer quotient/remainder form reports the same first
        // zero divisor without evaluating undefined C integer arithmetic.
        for op in operations {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", Shape::from([1]), DType::I64);
            let rhs = graph.input_dtype("rhs", Shape::from([1]), DType::I64);
            let output = graph.binary(op, lhs, rhs).unwrap();
            let kernel =
                CpuJit::compile(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert_eq!(
                kernel.call(
                    &mut [
                        JitBuffer::from_tensor(
                            &TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap(),
                            false,
                        ),
                        JitBuffer::from_tensor(
                            &TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
                            false,
                        ),
                        JitBuffer::zeroed(DType::I64, 1, true),
                    ],
                    &[],
                ),
                Err(JitError::DivisionByZero { index: 0 }),
                "{op:?}"
            );
        }
    }

    #[test]
    fn scalar_exact_negation_is_wrapping_and_cache_separated() {
        for (dtype, value) in [
            (DType::Bool, Scalar::Bool(true)),
            (DType::I8, Scalar::I(i8::MIN.into())),
            (DType::I64, Scalar::I(i64::MIN)),
            (DType::U64, Scalar::U(1)),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([1]), dtype);
            let output = graph.neg(input).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let scalar = CpuJit::render(&uop).unwrap();
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert_ne!(scalar.cache_key, vector.cache_key, "{dtype:?}");
            if matches!(dtype, DType::I8 | DType::I64) {
                assert!(scalar.source.contains("rg_i"), "{dtype:?}");
            }

            let values = TensorData::from_scalars(Shape::from([1]), dtype, [value]).unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), values.clone())]),
                )
                .unwrap();
            let kernel = CpuJit::compile(&uop).unwrap();
            let vector_kernel = CpuJit::compile_vectorized(&uop).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, 1, true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let mut vector_buffers = [
                JitBuffer::from_tensor(&values, false),
                JitBuffer::zeroed(dtype, 1, true),
            ];
            vector_kernel.call(&mut vector_buffers, &[]).unwrap();
            let native = buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            let vector_native = vector_buffers[1]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            assert_eq!(native.storage(), expected.storage(), "{dtype:?}");
            assert_eq!(vector_native.storage(), expected.storage(), "{dtype:?}");
        }
    }

    #[test]
    fn native_domain_failure_restores_the_mutable_output_buffer() {
        let mut graph = Graph::new();
        let numerator = graph.input_dtype("numerator", Shape::from([2]), DType::I64);
        let denominator = graph.input_dtype("denominator", Shape::from([2]), DType::I64);
        let quotient = graph
            .binary(crate::BinaryOp::Div, numerator, denominator)
            .unwrap();
        let kernel =
            CpuJit::compile(&crate::lower_graph_elementwise(&graph, quotient).unwrap()).unwrap();
        let mut numerator = JitBuffer::zeroed(DType::I64, 2, false);
        for (bytes, value) in numerator.bytes_mut().chunks_exact_mut(8).zip([8i64, 9]) {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        let mut denominator = JitBuffer::zeroed(DType::I64, 2, false);
        for (bytes, value) in denominator.bytes_mut().chunks_exact_mut(8).zip([2i64, 0]) {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        let mut output = JitBuffer::zeroed(DType::I64, 2, true);
        let sentinel = [0x5au8; 16];
        output.bytes_mut().copy_from_slice(&sentinel);
        let numerator_before = numerator.bytes().to_vec();
        let denominator_before = denominator.bytes().to_vec();

        let mut buffers = [numerator, denominator, output];
        assert_eq!(
            kernel.call(&mut buffers, &[]),
            Err(JitError::DivisionByZero { index: 1 })
        );
        assert_eq!(buffers[0].bytes(), numerator_before);
        assert_eq!(buffers[1].bytes(), denominator_before);
        assert_eq!(buffers[2].bytes(), sentinel);
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
    fn scalar_and_vector_jit_bf16_casts_preserve_nan_payloads_exactly() {
        let input_bits = [
            0x0000_0000u32,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x3f80_8000,
            0x3f81_8000,
            0xbf80_8000,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0x7f80_7fff,
            0x7f81_0000,
            0x7fc0_0000,
            0x7fff_ffff,
            0xff80_0001,
            0xffff_ffff,
        ];
        let expected = [
            0x0000u16, 0x8000, 0x0000, 0x0080, 0x3f80, 0x3f82, 0xbf80, 0x7f80, 0xff80, 0x7f81,
            0x7f81, 0x7f81, 0x7fc0, 0x7fff, 0xff81, 0xffff,
        ];
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([16]), DType::F32);
        let output = graph.cast(input, DType::BF16).unwrap();
        let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
        for vectorized in [false, true] {
            let rendered = if vectorized {
                CpuJit::render_vectorized(&uop).unwrap()
            } else {
                CpuJit::render(&uop).unwrap()
            };
            assert!(rendered.source.contains("(b&0x7f800000)==0x7f800000"));
            assert!(rendered.source.contains("(hi|1)"));
            let kernel = if vectorized {
                CpuJit::compile_vectorized(&uop).unwrap()
            } else {
                CpuJit::compile(&uop).unwrap()
            };
            let mut input_buffer = JitBuffer::zeroed(DType::F32, 16, false);
            for (raw, bits) in input_buffer.bytes_mut().chunks_exact_mut(4).zip(input_bits) {
                raw.copy_from_slice(&bits.to_ne_bytes());
            }
            let output_buffer = JitBuffer::zeroed(DType::BF16, 16, true);
            let mut buffers = [input_buffer, output_buffer];
            kernel.call(&mut buffers, &[]).unwrap();
            let actual = buffers[1]
                .bytes()
                .chunks_exact(2)
                .map(|raw| u16::from_ne_bytes(raw.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "vectorized={vectorized}");
        }
    }

    #[test]
    fn typed_narrow_casts_use_a_storage_roundtrip_before_fused_alu() {
        for (dtype, marker) in [
            (DType::F16, "rg_f16_to_f32(rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32(rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([4]), DType::F32);
            let cast = graph.cast(input, dtype).unwrap();
            let output = graph.mul(cast, cast).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains(marker), "{dtype:?}");
            assert!(scalar.source.contains(RENDERER_VERSION));
            assert_eq!(scalar.cache_key, CpuJit::render(&uop).unwrap().cache_key);

            // B1 explicitly rejects tagged narrow Casts and uses the scalar
            // per-lane source instead of a raw-u16 deferred conversion.
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert!(vector.source.contains(marker), "{dtype:?}");
        }
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
    fn float8_reduction_renderer_decodes_lanes_and_encodes_results() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let format = dtype.float8_format().unwrap();
            let input = TensorData::from_storage(
                [2, 3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    format,
                    [-0.0, 1.0, 2.0, -1.0, 0.5, 4.0]
                        .into_iter()
                        .map(|value| format.encode(value))
                        .collect(),
                )),
            )
            .unwrap();
            let inputs = HashMap::from([("x".into(), input.clone())]);

            for kind in [
                crate::ReduceKind::Sum,
                crate::ReduceKind::Mean,
                crate::ReduceKind::Product,
                crate::ReduceKind::Max,
                crate::ReduceKind::Min,
            ] {
                let mut graph = Graph::new();
                let x = graph.input_dtype("x", Shape::from([2, 3]), dtype);
                let output = graph.reduce(x, kind, Some(vec![1]), false).unwrap();
                let uop = crate::lower_graph_reduction(&graph, output).unwrap();
                let rendered = CpuJit::render(&uop).unwrap();
                assert!(
                    rendered.source.matches("rg_f8_decode(").count() > 1,
                    "{dtype:?} {kind:?}"
                );
                assert!(
                    rendered.source.matches("rg_f8_encode(").count() > 1,
                    "{dtype:?} {kind:?}"
                );
                let vectorized = CpuJit::render_vectorized(&uop).unwrap();
                assert!(
                    vectorized.source.matches("rg_f8_decode(").count() > 1
                        && vectorized.source.matches("rg_f8_encode(").count() > 1,
                    "{dtype:?} {kind:?}"
                );
                assert_eq!(
                    crate::execute_elementwise(&graph, output, &inputs)
                        .unwrap()
                        .storage(),
                    CpuBackend
                        .execute(&graph, output, &inputs)
                        .unwrap()
                        .storage(),
                    "captured {dtype:?} {kind:?}",
                );

                // One native execution per format proves that the generated
                // decoder/encoder declarations compose and that the raw-byte
                // ABI agrees with the CPU oracle. The other kinds share this
                // exact boundary and retain their existing typed formulas.
                if kind == crate::ReduceKind::Sum {
                    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
                    let jit = CpuJit::compile(&uop).unwrap();
                    let mut buffers = [
                        JitBuffer::from_tensor(&input, false),
                        JitBuffer::zeroed(dtype, expected.len(), true),
                    ];
                    jit.call(&mut buffers, &[]).unwrap();
                    let native = buffers
                        .into_iter()
                        .nth(1)
                        .unwrap()
                        .into_tensor(expected.shape().clone())
                        .unwrap();
                    assert_eq!(native.storage(), expected.storage(), "{dtype:?}");
                }
            }

            let empty = TensorData::from_storage(
                [2, 0],
                Storage::Float8(crate::Float8Storage::from_raw(format, Vec::new())),
            )
            .unwrap();
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", Shape::from([2, 0]), dtype);
            let output = graph
                .reduce(x, crate::ReduceKind::Mean, Some(vec![1]), false)
                .unwrap();
            let inputs = HashMap::from([("x".into(), empty.clone())]);
            let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
            let jit =
                CpuJit::compile(&crate::lower_graph_reduction(&graph, output).unwrap()).unwrap();
            let mut buffers = [
                JitBuffer::from_tensor(&empty, false),
                JitBuffer::zeroed(dtype, expected.len(), true),
            ];
            jit.call(&mut buffers, &[]).unwrap();
            let native = buffers
                .into_iter()
                .nth(1)
                .unwrap()
                .into_tensor(expected.shape().clone())
                .unwrap();
            assert_eq!(native.storage(), expected.storage(), "empty {dtype:?}");
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
    fn reduction_renderer_product_bool_and_extrema_preserve_typed_contracts() {
        let mut numeric = Graph::new();
        let input = numeric.input_dtype("input", Shape::from([2, 3]), DType::F32);
        let product = numeric
            .reduce(input, crate::ReduceKind::Product, Some(vec![1]), false)
            .unwrap();
        let product_uop = crate::lower_graph_reduction(&numeric, product).unwrap();
        let product_scalar = CpuJit::render(&product_uop).unwrap();
        assert!(product_scalar.source.contains(RENDERER_VERSION));
        assert!(product_scalar.source.contains("rg_acc = 1;"));
        assert!(product_scalar.source.contains("rg_acc *"));
        assert_eq!(
            product_scalar.cache_key,
            CpuJit::render(&product_uop).unwrap().cache_key
        );
        assert!(CpuJit::render_vectorized(&product_uop).is_ok());

        for (dtype, conversion) in [
            (DType::F16, "rg_f32_to_f16"),
            (DType::BF16, "rg_f32_to_bf16"),
        ] {
            let mut narrow = Graph::new();
            let input = narrow.input_dtype("input", Shape::from([1, 2]), dtype);
            let product = narrow
                .reduce(input, crate::ReduceKind::Product, Some(vec![1]), false)
                .unwrap();
            let source = CpuJit::render(&crate::lower_graph_reduction(&narrow, product).unwrap())
                .unwrap()
                .source;
            assert!(source.contains(conversion), "{dtype:?}");
        }

        let mut boolean = Graph::new();
        let input = boolean.input_dtype("input", Shape::from([2, 2]), DType::Bool);
        let product = boolean
            .reduce(input, crate::ReduceKind::Product, Some(vec![1]), true)
            .unwrap();
        let bool_uop = crate::lower_graph_reduction(&boolean, product).unwrap();
        let bool_scalar = CpuJit::render(&bool_uop).unwrap();
        assert!(bool_scalar.source.contains("uint8_t rg_acc = 1;"));
        assert!(bool_scalar.source.contains("rg_acc &&"));
        assert!(CpuJit::render_vectorized(&bool_uop).is_ok());

        let mut extrema = Graph::new();
        let input = extrema.input_dtype("input", Shape::from([1, 3]), DType::F32);
        let maximum = extrema
            .reduce(input, crate::ReduceKind::Max, Some(vec![1]), false)
            .unwrap();
        let extrema_scalar =
            CpuJit::render(&crate::lower_graph_reduction(&extrema, maximum).unwrap())
                .unwrap()
                .source;
        assert!(extrema_scalar.contains("int rg_seen = 0;"));
        assert!(extrema_scalar.contains("if (!rg_seen)"));
        assert!(extrema_scalar.contains("!isnan(rg_acc)"));

        for (dtype, kind, comparison) in [
            (DType::Bool, crate::ReduceKind::Max, "> rg_acc"),
            (DType::I64, crate::ReduceKind::Min, "< rg_acc"),
            (DType::U64, crate::ReduceKind::Max, "> rg_acc"),
        ] {
            let mut integral = Graph::new();
            let input = integral.input_dtype("input", Shape::from([1, 3]), dtype);
            let output = integral.reduce(input, kind, Some(vec![1]), false).unwrap();
            let uop = crate::lower_graph_reduction(&integral, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("int rg_seen = 0;"), "{dtype:?}");
            assert!(rendered.source.contains(comparison), "{dtype:?}");
            assert!(!rendered.source.contains("isnan((uint64_t)"), "{dtype:?}");
            assert!(CpuJit::render_vectorized(&uop).is_ok(), "{dtype:?}");
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
            if len == 5 {
                let mut malformed = program.clone();
                malformed.tail_elements = 2;
                let ids = vector_source
                    .abi
                    .buffers
                    .iter()
                    .enumerate()
                    .map(|(index, buffer)| (buffer.id, index))
                    .collect::<BTreeMap<_, _>>();
                assert!(matches!(
                    render_vector_program(&malformed, &vector_source.abi, &ids, len),
                    Err(JitError::Unsupported(_))
                ));
            }
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
    fn portable_vector_tail_domain_failure_restores_output() {
        let mut graph = Graph::new();
        let numerator = graph.input_dtype("numerator", Shape::from([5]), DType::I32);
        let denominator = graph.input_dtype("denominator", Shape::from([5]), DType::I32);
        let quotient = graph
            .binary(crate::BinaryOp::Div, numerator, denominator)
            .unwrap();
        let uop = crate::lower_graph_elementwise(&graph, quotient).unwrap();
        let rendered = CpuJit::render_vectorized(&uop).unwrap();
        assert!(rendered.source.contains("VectorProgram key"));
        let kernel = CpuJit::compile_vectorized(&uop).unwrap();
        let mut numerator = JitBuffer::zeroed(DType::I32, 5, false);
        let mut denominator = JitBuffer::zeroed(DType::I32, 5, false);
        for ((numerator, denominator), divisor) in numerator
            .bytes_mut()
            .chunks_exact_mut(4)
            .zip(denominator.bytes_mut().chunks_exact_mut(4))
            .zip([1i32, 1, 1, 1, 0])
        {
            numerator.copy_from_slice(&8i32.to_ne_bytes());
            denominator.copy_from_slice(&divisor.to_ne_bytes());
        }
        let mut output = JitBuffer::zeroed(DType::I32, 5, true);
        output.bytes_mut().fill(0x3c);
        let before = output.bytes().to_vec();
        let mut buffers = [numerator, denominator, output];

        assert_eq!(
            kernel.call(&mut buffers, &[]),
            Err(JitError::DivisionByZero { index: 4 })
        );
        assert_eq!(buffers[2].bytes(), before);
    }

    #[test]
    fn portable_b2_exact_vectors_execute() {
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
    fn narrow_select_and_cast_vector_requests_use_legacy_per_lane_rendering() {
        for (dtype, decode, encode) in [
            (DType::F16, "rg_f16_to_f32", "rg_f32_to_f16"),
            (DType::BF16, "rg_bf16_to_f32", "rg_f32_to_bf16"),
        ] {
            let mut graph = Graph::new();
            let condition = graph.input_dtype("condition", Shape::from([5]), DType::Bool);
            let on_true = graph.input_dtype("on_true", Shape::from([5]), dtype);
            let on_false = graph.input_dtype("on_false", Shape::from([5]), dtype);
            // Keep a live Select branch and both widening/narrowing Casts so
            // raw F16/BF16 signed-zero, NaN/infinity, and fractional lanes
            // must traverse the source-correct conversion helpers.
            let selected = graph.select(condition, on_true, on_false).unwrap();
            let widened = graph.cast(selected, DType::F32).unwrap();
            let output = graph.cast(widened, dtype).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();

            let scalar = CpuJit::render(&uop).unwrap();
            assert!(scalar.source.contains(decode), "{dtype:?}");
            assert!(scalar.source.contains(encode), "{dtype:?}");
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
            assert!(vector.source.contains(decode), "{dtype:?}");
            assert!(vector.source.contains(encode), "{dtype:?}");
            assert_eq!(
                vector.cache_key,
                CpuJit::render_vectorized(&uop).unwrap().cache_key
            );

            let linear = CpuJit::linearize(&uop).unwrap();
            let spaces = crate::MemorySpacePlan::from_linear(&linear).unwrap();
            let program = crate::VectorProgram::from_linear(&linear, &spaces).unwrap();
            assert!(matches!(
                program.b1_eligibility(),
                Err(crate::VectorIrError::Unsupported(reason))
                    if reason == "portable narrow vector ABI needs tagged float lanes"
            ));
        }

        for dtype in [DType::F32, DType::I32] {
            let mut graph = Graph::new();
            let condition = graph.input_dtype("condition", Shape::from([5]), DType::Bool);
            let on_true = graph.input_dtype("on_true", Shape::from([5]), dtype);
            let on_false = graph.input_dtype("on_false", Shape::from([5]), dtype);
            let output = graph.select(condition, on_true, on_false).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            assert!(
                CpuJit::render_vectorized(&uop)
                    .unwrap()
                    .source
                    .contains("B2 VectorProgram"),
                "{dtype:?} Select remains B2-eligible"
            );
        }
    }

    #[test]
    fn portable_b2_reports_first_division_and_shift_failure() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([5]), DType::I32);
        let y = graph.input_dtype("y", Shape::from([5]), DType::I32);
        let out = graph.binary(crate::BinaryOp::Div, x, y).unwrap();
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
    fn scalar_signed_right_shift_is_defined_and_matches_cpu_oracle() {
        for (dtype, values, counts) in [
            (
                DType::I8,
                vec![Scalar::I(i8::MIN.into()), Scalar::I(-3), Scalar::I(1)],
                vec![Scalar::I(0), Scalar::I(1), Scalar::I(7)],
            ),
            (
                DType::I64,
                vec![Scalar::I(i64::MIN), Scalar::I(-3), Scalar::I(1)],
                vec![Scalar::I(0), Scalar::I(1), Scalar::I(63)],
            ),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([3]), dtype);
            let shift = graph.input_dtype("shift", Shape::from([3]), dtype);
            let output = graph.shr(input, shift).unwrap();
            let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
            let rendered = CpuJit::render(&uop).unwrap();
            assert!(rendered.source.contains("rg_sshr((uint64_t)"), "{dtype:?}");
            assert_eq!(rendered.source, CpuJit::render(&uop).unwrap().source);

            let input_data = TensorData::from_scalars(Shape::from([3]), dtype, values).unwrap();
            let shift_data = TensorData::from_scalars(Shape::from([3]), dtype, counts).unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        ("input".into(), input_data.clone()),
                        ("shift".into(), shift_data.clone()),
                    ]),
                )
                .unwrap();
            let scalar = CpuJit::compile(&uop).unwrap();
            let vector = CpuJit::compile_vectorized(&uop).unwrap();
            let mut scalar_buffers = [
                JitBuffer::from_tensor(&input_data, false),
                JitBuffer::from_tensor(&shift_data, false),
                JitBuffer::zeroed(dtype, 3, true),
            ];
            scalar.call(&mut scalar_buffers, &[]).unwrap();
            let mut vector_buffers = [
                JitBuffer::from_tensor(&input_data, false),
                JitBuffer::from_tensor(&shift_data, false),
                JitBuffer::zeroed(dtype, 3, true),
            ];
            vector.call(&mut vector_buffers, &[]).unwrap();
            let scalar_output = scalar_buffers[2]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            let vector_output = vector_buffers[2]
                .clone()
                .into_tensor(expected.shape().clone())
                .unwrap();
            assert_eq!(
                scalar_output.storage(),
                expected.storage(),
                "scalar {dtype:?}"
            );
            assert_eq!(
                vector_output.storage(),
                expected.storage(),
                "vector {dtype:?}"
            );

            let invalid = TensorData::from_scalars(
                Shape::from([3]),
                dtype,
                [Scalar::I(1), Scalar::I(dtype.bits().into()), Scalar::I(1)],
            )
            .unwrap();
            assert_eq!(
                scalar.call(
                    &mut [
                        JitBuffer::from_tensor(&input_data, false),
                        JitBuffer::from_tensor(&invalid, false),
                        JitBuffer::zeroed(dtype, 3, true),
                    ],
                    &[],
                ),
                Err(JitError::InvalidShift { index: 1 }),
                "{dtype:?}"
            );
        }
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

    #[test]
    fn raw_matmul_uses_dtype_width_accumulators_and_renderer_v16() {
        let mut f32_graph = Graph::new();
        let f32_lhs = f32_graph.input_dtype("lhs", Shape::from([1, 3]), DType::F32);
        let f32_rhs = f32_graph.input_dtype("rhs", Shape::from([3, 1]), DType::F32);
        let f32_output = f32_graph.matmul(f32_lhs, f32_rhs).unwrap();
        let f32_kernel = crate::lower_graph_matmul(&f32_graph, f32_output).unwrap();
        let f32_rendered = CpuJit::render(&f32_kernel).unwrap();
        assert!(f32_rendered.source.contains(RENDERER_VERSION));
        assert!(f32_rendered.source.contains("float rg_acc=0.0f;"));
        assert!(f32_rendered.source.contains("float rg_product=(float)"));
        assert!(
            f32_rendered
                .source
                .contains("rg_acc=(float)(rg_acc+rg_product);")
        );
        assert!(!f32_rendered.source.contains("double rg_acc=0.0;"));
        assert_eq!(
            f32_rendered.cache_key,
            CpuJit::render(&f32_kernel).unwrap().cache_key
        );

        // This adversarial contraction distinguishes per-step F32 storage
        // rounding from a double accumulator narrowed only at the end.
        let oracle = CpuBackend
            .execute(
                &f32_graph,
                f32_output,
                &HashMap::from([
                    (
                        "lhs".into(),
                        TensorData::from_storage([1, 3], Storage::F32(vec![1.0e10, 1.0, -1.0e10]))
                            .unwrap(),
                    ),
                    (
                        "rhs".into(),
                        TensorData::from_storage([3, 1], Storage::F32(vec![1.0; 3])).unwrap(),
                    ),
                ]),
            )
            .unwrap();
        let Scalar::F(value) = oracle.scalar_at(0) else {
            panic!("F32 matmul must retain F32 scalar storage")
        };
        assert_eq!((value as f32).to_bits(), 0.0f32.to_bits());

        let mut f64_graph = Graph::new();
        let f64_lhs = f64_graph.input_dtype("lhs", Shape::from([3]), DType::F64);
        let f64_rhs = f64_graph.input_dtype("rhs", Shape::from([3]), DType::F64);
        let f64_output = f64_graph.matmul(f64_lhs, f64_rhs).unwrap();
        let f64_rendered =
            CpuJit::render(&crate::lower_graph_matmul(&f64_graph, f64_output).unwrap()).unwrap();
        assert!(f64_rendered.source.contains("double rg_acc=0.0;"));
        assert!(!f64_rendered.source.contains("float rg_product=(float)"));
    }
}
