//! Deterministic phase-one PTX rendering and Driver launch glue.
//!
//! The renderer intentionally accepts only the fused elementwise UOp subset
//! that has a clear PTX contract. The CPU UOp interpreter remains the semantic
//! oracle; only exact static F32/F64 sum/mean reductions are admitted. Narrow
//! floats remain rejected outside the validated operation-scoped storage ABI;
//! guarded integer division/shifts and device-status reporting remain
//! fail-closed.

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
#[path = "ptx_matmul.rs"]
mod matmul;
#[cfg(test)]
#[path = "ptx_matmul_tests.rs"]
mod matmul_tests;

pub const PTX_RENDERER_VERSION: &str = "rustgrad-ptx-elementwise-v26";
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
    pub launch: PtxLaunchGeometry,
    pub semantic_program: Option<KernelSemanticProgram>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PtxLaunchGeometry {
    Linear,
    Exact(LaunchConfig),
}
#[derive(Clone, Debug)]
pub enum KernelSemanticProgram {
    UOp(Arc<UOp>),
    Matmul(Arc<crate::MatmulKernelPlan>),
    TiledMatmul(Arc<crate::TiledMatmulPayload>),
    TensorCoreMatmul(Arc<crate::TensorCoreMatmulPayload>),
}
impl RenderedPtx {
    fn validate(&self) -> Result<(), PtxError> {
        if let PtxLaunchGeometry::Exact(config) = self.launch
            && (config.grid.contains(&0) || config.block.contains(&0) || self.extent == 0)
        {
            return Err(PtxError::InvalidBinding(
                "exact PTX launch geometry is empty".into(),
            ));
        }
        if let Some(KernelSemanticProgram::TiledMatmul(payload)) = &self.semantic_program {
            payload
                .validate()
                .map_err(|error| PtxError::InvalidBinding(error.to_string()))?;
            let expected = payload
                .tile
                .launch_geometry(&payload.matmul)
                .map_err(|error| PtxError::InvalidBinding(error.to_string()))?;
            if self.launch != PtxLaunchGeometry::Exact(expected)
                || self.extent
                    != payload
                        .matmul
                        .output_shape
                        .numel()
                        .map_err(|_| PtxError::Overflow)?
                || self.buffers.len() != 3
                || self.buffers[0].id != payload.matmul.lhs.index() as u64
                || self.buffers[1].id != payload.matmul.rhs.index() as u64
                || self.buffers[2].id != payload.matmul.output.index() as u64
                || self.buffers[0].dtype != DType::F32
                || self.buffers[1].dtype != DType::F32
                || self.buffers[2].dtype != DType::F32
                || self.buffers[..2].iter().any(|buffer| buffer.mutable)
                || !self.buffers[2].mutable
                || self.buffers[0].source_shape != payload.matmul.lhs_shape
                || self.buffers[1].source_shape != payload.matmul.rhs_shape
                || self.buffers[2].source_shape != payload.matmul.output_shape
                || self.buffers[0].elements
                    != payload
                        .matmul
                        .lhs_shape
                        .numel()
                        .map_err(|_| PtxError::Overflow)?
                || self.buffers[1].elements
                    != payload
                        .matmul
                        .rhs_shape
                        .numel()
                        .map_err(|_| PtxError::Overflow)?
                || self.buffers[2].elements != self.extent
            {
                return Err(PtxError::InvalidBinding(
                    "tiled PTX launch disagrees with its payload".into(),
                ));
            }
        }
        if let Some(KernelSemanticProgram::TensorCoreMatmul(payload)) = &self.semantic_program {
            payload
                .validate()
                .map_err(|error| PtxError::InvalidBinding(error.to_string()))?;
            let expected = payload
                .tensor_core
                .launch_geometry(&payload.matmul)
                .map_err(|error| PtxError::InvalidBinding(error.to_string()))?;
            let plan = &payload.matmul;
            if self.launch != PtxLaunchGeometry::Exact(expected)
                || self.extent != plan.output_shape.numel().map_err(|_| PtxError::Overflow)?
                || self.buffers.len() != 3
                || self.buffers[0].id != plan.lhs.index() as u64
                || self.buffers[1].id != plan.rhs.index() as u64
                || self.buffers[2].id != plan.output.index() as u64
                || self.buffers[0].dtype != payload.tensor_core.input_dtype
                || self.buffers[1].dtype != payload.tensor_core.input_dtype
                || self.buffers[2].dtype != payload.tensor_core.output_dtype
                || self.buffers[..2].iter().any(|buffer| buffer.mutable)
                || !self.buffers[2].mutable
                || self.buffers[0].source_shape != plan.lhs_shape
                || self.buffers[1].source_shape != plan.rhs_shape
                || self.buffers[2].source_shape != plan.output_shape
                || self.buffers[0].elements
                    != plan.lhs_shape.numel().map_err(|_| PtxError::Overflow)?
                || self.buffers[1].elements
                    != plan.rhs_shape.numel().map_err(|_| PtxError::Overflow)?
                || self.buffers[2].elements != self.extent
            {
                return Err(PtxError::InvalidBinding(
                    "tensor-core PTX launch disagrees with its payload".into(),
                ));
            }
        }
        Ok(())
    }

    fn effective_block_size(&self, requested: u32) -> Result<u32, PtxError> {
        if requested == 0 {
            return Err(PtxError::InvalidBinding("zero block size".into()));
        }
        match self.launch {
            PtxLaunchGeometry::Linear => Ok(requested),
            PtxLaunchGeometry::Exact(config) => config
                .block
                .into_iter()
                .try_fold(1u32, |threads, dimension| threads.checked_mul(dimension))
                .ok_or(PtxError::Overflow),
        }
    }

    fn launch_config(&self, block_size: u32) -> Result<LaunchConfig, PtxError> {
        match self.launch {
            PtxLaunchGeometry::Linear => {
                let grid = self
                    .extent
                    .checked_add(block_size as usize - 1)
                    .ok_or(PtxError::Overflow)?
                    / block_size as usize;
                Ok(LaunchConfig {
                    grid: [u32::try_from(grid).map_err(|_| PtxError::Overflow)?, 1, 1],
                    block: [block_size, 1, 1],
                    shared_bytes: 0,
                })
            }
            PtxLaunchGeometry::Exact(config) => Ok(config),
        }
    }
    fn validate_pointer_alignment(&self, index: usize, pointer: u64) -> Result<(), PtxError> {
        let alignment = if matches!(
            self.semantic_program,
            Some(KernelSemanticProgram::TensorCoreMatmul(_))
        ) {
            16
        } else {
            1
        };
        if pointer % alignment as u64 != 0 {
            return Err(PtxError::InvalidBinding(format!(
                "buffer {index} pointer is not {alignment}-byte aligned"
            )));
        }
        Ok(())
    }
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
    pub program: KernelSemanticProgram,
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
    /// Renders the explicit correctness-first serial policy for a validated
    /// Matmul plan with the fixed lhs/rhs/output ABI.
    pub fn render_matmul_plan(
        &self,
        plan: &crate::MatmulKernelPlan,
    ) -> Result<RenderedPtx, PtxError> {
        matmul::render_serial(self, plan)
    }
    /// Renders a validated selected tiled payload with exact workgroup launch
    /// geometry and dynamic shared-memory requirements.
    pub fn render_tiled_matmul_plan(
        &self,
        payload: &crate::TiledMatmulPayload,
    ) -> Result<RenderedPtx, PtxError> {
        matmul::render_tiled(self, payload)
    }
    /// Renders a capability-validated single-warp m16n8k16 MMA payload.
    pub fn render_tensor_core_matmul_plan(
        &self,
        payload: &crate::TensorCoreMatmulPayload,
    ) -> Result<RenderedPtx, PtxError> {
        matmul::render_tensor_core(self, payload)
    }
}

fn render(renderer: &PtxRenderer, root: &UOp) -> Result<RenderedPtx, PtxError> {
    if matches!(root.kind(), UOpKind::Random) {
        let UArg::Random(plan) = root.arg() else {
            return Err(PtxError::Unsupported("random payload is absent".into()));
        };
        return render_random(renderer, root, plan);
    }
    if matches!(root.kind(), UOpKind::Matmul) {
        return match root.arg() {
            UArg::Matmul(plan) => matmul::render_serial(renderer, plan),
            UArg::TiledMatmul(payload) => matmul::render_tiled(renderer, payload),
            UArg::TensorCoreMatmul(payload) => matmul::render_tensor_core(renderer, payload),
            _ => Err(PtxError::Unsupported("matmul payload is absent".into())),
        };
    }
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
    // Narrow storage is deliberately not a generic elementwise capability.
    // The only exceptions are completely validated public Sign, Abs, Neg,
    // Reciprocal, Mul, Add, Sub, Div, Eq, Ne, ordered-Lt, direct-mask Select,
    // the strict public ReLU, LeakyReLU, Maximum, and Minimum roots; each has
    // a source-proven typed storage boundary.
    let storage_mode = scoped_storage_plan(store, renderer.sm)?;
    let mut abi = BTreeMap::new();
    let reduction = reduction_spec(store)?;
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
            if reduction.is_some() {
                reject_reduction_storage_dtype(dtype)?;
            } else if storage_mode.is_some() {
                reject_sign_storage_dtype(dtype)?;
            } else {
                reject_dtype(dtype)?;
            }
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
    if let Some(reduction) = reduction {
        return render_reduction(renderer, store, &buffers, *out_id, *extent, reduction);
    }
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
        // `%r90` is reserved for the BF16 raw-bit decode.  Keep this pool
        // explicit even for ordinary kernels so cache keys fully describe the
        // versioned Sign narrow-storage ABI.
        "  .reg .b32 %r<96>;".into(),
        "  .reg .b64 %rd<32>;".into(),
        "  .reg .f32 %f<32>;".into(),
        // `%fd31` is the scoped Reciprocal widening scratch. Keep the
        // declaration in renderer identity so no older artifact is reused.
        "  .reg .f64 %fd<32>;".into(),
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
        "%r3",
        false,
        storage_mode,
    )?;
    let out = buffers.iter().find(|b| b.id == *out_id).unwrap();
    let oi = ids[out_id] + 1;
    let value = narrow_storage_result(&mut lines, value, out.dtype, storage_mode);
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
        launch: PtxLaunchGeometry::Linear,
        semantic_program: Some(KernelSemanticProgram::UOp(Arc::new(root.clone()))),
    })
}

/// Renders an immutable captured Threefry source.  It has exactly one output
/// pointer and never observes the process-local stream registry.
fn render_random(
    renderer: &PtxRenderer,
    root: &UOp,
    plan: &crate::random::plan::RandomKernelPlan,
) -> Result<RenderedPtx, PtxError> {
    plan.validate()
        .map_err(|error| PtxError::Unsupported(error.to_string()))?;
    let kind = plan.kind;
    let supported = match kind {
        crate::RandomKind::Uniform { .. } | crate::RandomKind::Normal { .. } => {
            matches!(
                plan.dtype,
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            )
        }
        crate::RandomKind::RandInt { .. } => matches!(
            plan.dtype,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
        ),
    };
    if !supported {
        return Err(PtxError::Unsupported(format!(
            "PTX Threefry {:?} dtype {:?}",
            kind, plan.dtype
        )));
    }
    if plan.dtype == DType::F16 && renderer.sm < 53 {
        return Err(PtxError::Unsupported(
            "F16 Threefry conversion requires sm_53 or newer".into(),
        ));
    }
    let extent = plan.shape.numel().map_err(|_| PtxError::Overflow)?;
    if extent > u32::MAX as usize {
        return Err(PtxError::Unsupported(
            "PTX Threefry linear extent exceeds u32 thread indexing".into(),
        ));
    }
    let buffer = PtxBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.shape.clone(),
        elements: extent,
        mutable: true,
    };
    let entry = format!("rg_random_e{extent}");
    let mut lines = vec![
        format!("// {PTX_RENDERER_VERSION} captured-threefry ABI {PTX_ABI_VERSION}"),
        ".version 7.0".into(),
        format!(".target sm_{}", renderer.sm),
        ".address_size 64".into(),
        "".into(),
        format!(".visible .entry {entry}("),
        "  .param .u64 p0,".into(),
        "  .param .u64 extent".into(),
        ")".into(),
        "{".into(),
        "  .reg .pred %p<8>;".into(),
        "  .reg .b32 %r<96>;".into(),
        "  .reg .b64 %rd<64>;".into(),
        "  .reg .f32 %f<16>;".into(),
        "  .reg .f64 %fd<16>;".into(),
        "  ld.param.u64 %rd10, [p0];".into(),
        "  ld.param.u64 %rd0, [extent];".into(),
        "  mov.u32 %r0, %ctaid.x;".into(),
        "  mov.u32 %r1, %ntid.x;".into(),
        "  mov.u32 %r2, %tid.x;".into(),
        "  mad.lo.u32 %r3, %r0, %r1, %r2;".into(),
        "  cvt.u64.u32 %rd11, %r3;".into(),
        "  setp.ge.u64 %p0, %rd11, %rd0;".into(),
        "  @%p0 bra DONE;".into(),
    ];
    if let crate::RandomKind::Normal { mean, std } = kind {
        lines.push("  mul.wide.u32 %rd12, %r3, 2;".into());
        emit_random_word(&mut lines, "%rd12", "%r50", plan);
        lines.push("  add.u64 %rd12, %rd12, 1;".into());
        emit_random_word(&mut lines, "%rd12", "%r51", plan);
        lines.extend([
            "  shr.u32 %r50, %r50, 9;".into(),
            "  or.b32 %r50, %r50, 0x3f800000;".into(),
            "  mov.b32 %f0, %r50;".into(),
            "  add.rn.f32 %f0, %f0, -1.0;".into(),
            "  shr.u32 %r51, %r51, 9;".into(),
            "  or.b32 %r51, %r51, 0x3f800000;".into(),
            "  mov.b32 %f1, %r51;".into(),
            "  add.rn.f32 %f1, %f1, -1.0;".into(),
            "  mul.rn.f32 %f2, %f0, 6.283185307179586;".into(),
            "  cos.approx.f32 %f2, %f2;".into(),
            "  sub.rn.f32 %f3, 1.0, %f1;".into(),
            "  lg2.approx.f32 %f3, %f3;".into(),
            "  mul.rn.f32 %f3, %f3, -1.3862943611198906;".into(),
            "  sqrt.rn.f32 %f3, %f3;".into(),
            "  mul.rn.f32 %f0, %f2, %f3;".into(),
            "  cvt.rn.f64.f32 %fd0, %f0;".into(),
            format!("  mov.b64 %fd1, 0x{:016x};", std.to_bits()),
            format!("  mov.b64 %fd2, 0x{:016x};", mean.to_bits()),
            "  mul.rn.f64 %fd0, %fd0, %fd1;".into(),
            "  add.rn.f64 %fd0, %fd0, %fd2;".into(),
        ]);
    } else if matches!(kind, crate::RandomKind::RandInt { .. }) {
        lines.push("  cvt.u64.u32 %rd12, %r3;".into());
        emit_random_word(&mut lines, "%rd12", "%r50", plan);
        lines.extend([
            "  shr.u32 %r50, %r50, 9;".into(),
            "  or.b32 %r50, %r50, 0x3f800000;".into(),
            "  mov.b32 %f0, %r50;".into(),
            "  add.rn.f32 %f0, %f0, -1.0;".into(),
            "  cvt.rn.f64.f32 %fd0, %f0;".into(),
        ]);
    } else {
        match plan.dtype {
            DType::F16 | DType::BF16 => {
                lines.push("  shr.u32 %r4, %r3, 1;".into());
                lines.push("  cvt.u64.u32 %rd12, %r4;".into());
                emit_random_word(&mut lines, "%rd12", "%r50", plan);
                lines.push("  and.b32 %r5, %r3, 1;".into());
                lines.push("  setp.ne.u32 %p1, %r5, 0;".into());
                lines.push("  shr.u32 %r6, %r50, 16;".into());
                lines.push("  selp.b32 %r6, %r6, %r50, %p1;".into());
                lines.push("  and.b32 %r6, %r6, 0xffff;".into());
                if plan.dtype == DType::F16 {
                    lines.extend([
                        "  shr.u32 %r6, %r6, 6;".into(),
                        "  or.b32 %r6, %r6, 0x3c00;".into(),
                        "  cvt.rn.f32.f16 %f0, %r6;".into(),
                    ]);
                } else {
                    lines.extend([
                        "  shr.u32 %r6, %r6, 9;".into(),
                        "  or.b32 %r6, %r6, 0x3f80;".into(),
                        "  shl.b32 %r7, %r6, 16;".into(),
                        "  mov.b32 %f0, %r7;".into(),
                    ]);
                }
                lines.push("  add.rn.f32 %f0, %f0, -1.0;".into());
                lines.push("  cvt.rn.f64.f32 %fd0, %f0;".into());
            }
            DType::F32 => {
                lines.push("  cvt.u64.u32 %rd12, %r3;".into());
                emit_random_word(&mut lines, "%rd12", "%r50", plan);
                lines.extend([
                    "  shr.u32 %r50, %r50, 9;".into(),
                    "  or.b32 %r50, %r50, 0x3f800000;".into(),
                    "  mov.b32 %f0, %r50;".into(),
                    "  add.rn.f32 %f0, %f0, -1.0;".into(),
                    "  cvt.rn.f64.f32 %fd0, %f0;".into(),
                ]);
            }
            DType::F64 => {
                lines.push("  mul.wide.u32 %rd12, %r3, 2;".into());
                emit_random_word(&mut lines, "%rd12", "%r50", plan);
                lines.push("  add.u64 %rd12, %rd12, 1;".into());
                emit_random_word(&mut lines, "%rd12", "%r51", plan);
                lines.extend([
                    "  cvt.u64.u32 %rd13, %r51;".into(),
                    "  shl.b64 %rd13, %rd13, 32;".into(),
                    "  cvt.u64.u32 %rd14, %r50;".into(),
                    "  or.b64 %rd13, %rd13, %rd14;".into(),
                    "  shr.u64 %rd13, %rd13, 12;".into(),
                    "  or.b64 %rd13, %rd13, 0x3ff0000000000000;".into(),
                    "  mov.b64 %fd0, %rd13;".into(),
                    "  add.rn.f64 %fd0, %fd0, -1.0;".into(),
                ]);
            }
            _ => unreachable!(),
        }
    }
    if let crate::RandomKind::Uniform { low, high } = kind {
        lines.extend([
            format!("  mov.b64 %fd1, 0x{:016x};", (high - low).to_bits()),
            format!("  mov.b64 %fd2, 0x{:016x};", low.to_bits()),
            "  mul.rn.f64 %fd0, %fd0, %fd1;".into(),
            "  add.rn.f64 %fd0, %fd0, %fd2;".into(),
        ]);
    } else if let crate::RandomKind::RandInt { low, high } = kind {
        lines.extend([
            format!(
                "  mov.b64 %fd1, 0x{:016x};",
                ((high - low) as f64).to_bits()
            ),
            format!("  mov.b64 %fd2, 0x{:016x};", (low as f64).to_bits()),
            "  mul.rn.f64 %fd0, %fd0, %fd1;".into(),
            "  add.rn.f64 %fd0, %fd0, %fd2;".into(),
        ]);
    }
    lines.extend([
        format!("  mul.wide.u32 %rd15, %r3, {};", plan.dtype.itemsize()),
        "  add.u64 %rd15, %rd10, %rd15;".into(),
    ]);
    match plan.dtype {
        DType::F16 => lines.extend([
            "  cvt.rn.f32.f64 %f1, %fd0;".into(),
            "  cvt.rn.f16.f32 %r60, %f1;".into(),
            "  st.global.b16 [%rd15], %r60;".into(),
        ]),
        DType::BF16 => lines.extend([
            "  cvt.rn.f32.f64 %f1, %fd0;".into(),
            "  mov.b32 %r60, %f1;".into(),
            "  shr.u32 %r61, %r60, 16;".into(),
            "  and.b32 %r61, %r61, 1;".into(),
            "  add.u32 %r61, %r61, 0x7fff;".into(),
            "  add.u32 %r60, %r60, %r61;".into(),
            "  shr.u32 %r60, %r60, 16;".into(),
            "  st.global.b16 [%rd15], %r60;".into(),
        ]),
        DType::F32 => lines.extend([
            "  cvt.rn.f32.f64 %f1, %fd0;".into(),
            "  st.global.f32 [%rd15], %f1;".into(),
        ]),
        DType::F64 => lines.push("  st.global.f64 [%rd15], %fd0;".into()),
        DType::I8 => lines.extend([
            "  cvt.rzi.s32.f64 %r60, %fd0;".into(),
            "  st.global.s8 [%rd15], %r60;".into(),
        ]),
        DType::I16 => lines.extend([
            "  cvt.rzi.s32.f64 %r60, %fd0;".into(),
            "  st.global.s16 [%rd15], %r60;".into(),
        ]),
        DType::I32 => lines.extend([
            "  cvt.rzi.s32.f64 %r60, %fd0;".into(),
            "  st.global.s32 [%rd15], %r60;".into(),
        ]),
        DType::I64 => lines.extend([
            "  cvt.rzi.s64.f64 %rd60, %fd0;".into(),
            "  st.global.s64 [%rd15], %rd60;".into(),
        ]),
        DType::U8 => lines.extend([
            "  cvt.rzi.u32.f64 %r60, %fd0;".into(),
            "  st.global.u8 [%rd15], %r60;".into(),
        ]),
        DType::U16 => lines.extend([
            "  cvt.rzi.u32.f64 %r60, %fd0;".into(),
            "  st.global.u16 [%rd15], %r60;".into(),
        ]),
        DType::U32 => lines.extend([
            "  cvt.rzi.u32.f64 %r60, %fd0;".into(),
            "  st.global.u32 [%rd15], %r60;".into(),
        ]),
        DType::U64 => lines.extend([
            "  cvt.rzi.u64.f64 %rd60, %fd0;".into(),
            "  st.global.u64 [%rd15], %rd60;".into(),
        ]),
        _ => unreachable!(),
    }
    lines.extend(["DONE:".into(), "  ret;".into(), "}".into()]);
    let source = lines.join("\n") + "\n";
    Ok(RenderedPtx {
        source: source.clone(),
        source_map: BTreeMap::new(),
        buffers: vec![buffer],
        extent,
        cache_key: stable_key(&(PTX_RENDERER_VERSION, renderer.sm, &source, plan)),
        entry,
        launch: PtxLaunchGeometry::Linear,
        semantic_program: Some(KernelSemanticProgram::UOp(Arc::new(root.clone()))),
    })
}

fn emit_random_word(
    lines: &mut Vec<String>,
    word: &str,
    out: &str,
    plan: &crate::random::plan::RandomKernelPlan,
) {
    // `words` splits at 2^32-1 words.  This inline sequence reproduces its
    // chunk counter carry, low-lane-then-high-lane packing, and both Threefry passes.
    lines.extend([
        format!("  div.u64 %rd20, {word}, 0xffffffff;"),
        "  mul.lo.u64 %rd21, %rd20, 0xffffffff;".into(),
        format!("  sub.u64 %rd22, {word}, %rd21;"),
        format!("  mov.u64 %rd23, {};", plan.word_count),
        "  sub.u64 %rd23, %rd23, %rd21;".into(),
        "  min.u64 %rd23, %rd23, 0xffffffff;".into(),
        "  add.u64 %rd24, %rd23, 1;".into(),
        "  shr.u64 %rd24, %rd24, 1;".into(),
        "  setp.lt.u64 %p2, %rd22, %rd24;".into(),
        "  sub.u64 %rd25, %rd22, %rd24;".into(),
        "  selp.u64 %rd25, %rd22, %rd25, %p2;".into(),
        "  cvt.u32.u64 %r10, %rd21;".into(),
        "  shr.u64 %rd26, %rd21, 32;".into(),
        "  cvt.u32.u64 %r11, %rd26;".into(),
        format!("  add.cc.u32 %r10, %r10, {};", plan.stream.counter[0]),
        format!("  addc.u32 %r11, %r11, {};", plan.stream.counter[1]),
        format!("  mov.u32 %r20, {};", plan.stream.key[0]),
        format!("  mov.u32 %r21, {};", plan.stream.key[1]),
        format!(
            "  mov.u32 %r22, {};",
            plan.stream.key[0] ^ plan.stream.key[1] ^ 0x1bd1_1bda
        ),
        "  add.u32 %r30, %r10, %r20;".into(),
        "  add.u32 %r31, %r11, %r21;".into(),
    ]);
    emit_threefry(lines, "%r30", "%r31", "%r20", "%r21", "%r22");
    lines.extend([
        "  cvt.u32.u64 %r32, %rd25;".into(),
        "  cvt.u32.u64 %r33, %rd24;".into(),
        "  add.u32 %r33, %r32, %r33;".into(),
        "  add.u32 %r34, %r30, 0;".into(),
        "  add.u32 %r35, %r31, 0;".into(),
        "  xor.b32 %r36, %r34, %r35;".into(),
        "  xor.b32 %r36, %r36, 0x1bd11bda;".into(),
    ]);
    emit_threefry(lines, "%r32", "%r33", "%r34", "%r35", "%r36");
    lines.extend([format!("  selp.b32 {out}, %r32, %r33, %p2;")]);
}

fn emit_threefry(lines: &mut Vec<String>, a: &str, b: &str, k0: &str, k1: &str, k2: &str) {
    const ROT: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];
    for round in 0..20 {
        lines.push(format!("  add.u32 {a}, {a}, {b};"));
        lines.push(format!("  shl.b32 %r40, {b}, {};", ROT[round % 8]));
        lines.push(format!("  shr.u32 %r41, {b}, {};", 32 - ROT[round % 8]));
        lines.push(format!("  or.b32 {b}, %r40, %r41;"));
        lines.push(format!("  xor.b32 {b}, {b}, {a};"));
        if round % 4 == 3 {
            let z = round / 4 + 1;
            let keys = [k0, k1, k2];
            lines.push(format!("  add.u32 {a}, {a}, {};", keys[z % 3]));
            lines.push(format!("  add.u32 {b}, {b}, {};", keys[(z + 1) % 3]));
            lines.push(format!("  add.u32 {b}, {b}, {z};"));
        }
    }
}
/// Validates the sole operation-scoped extension to the generic elementwise
/// storage ABI.  This is intentionally separate from `reject_dtype`: callers
/// must first prove one of the strict public roots above.
fn reject_sign_storage_dtype(dtype: DType) -> Result<(), PtxError> {
    match dtype {
        DType::Bool
        | DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScopedStorageMode {
    Sign,
    Abs,
    Neg,
    NegBool,
    Reciprocal,
    ReciprocalCast,
    Sqrt,
    SqrtCast,
    Rsqrt,
    Mul,
    Add,
    Sub,
    SubBool,
    Div,
    Eq,
    Ne,
    LogicalNot,
    IsInf,
    OrderedLt,
    InclusiveLt,
    Select,
    Relu,
    LeakyRelu,
    Extrema,
    Clamp,
}

/// Validates the only roots allowed to use the narrow-storage ABI. The Abs
/// arm is the exact public `x * x.sign()` DAG; Reciprocal accepts either its
/// direct floating ALU root or its exact public nonfloat `Cast(F32)` root;
/// Mul/Add/Sub delegate to a two-input source-LUB proof. Div proves the literal
/// source `lhs * reciprocal(rhs)` graph, including the reciprocal storage boundary.
/// None is a generic unary,
/// conversion, or binary admission.
fn scoped_storage_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    if matches!(value.kind(), UOpKind::GraphBinary(crate::BinaryOp::Add))
        && value.ty().map(|ty| ty.scalar) == Some(DType::Bool)
        && matches!(value.sources().get(1).map(|node| node.kind()), Some(UOpKind::GraphLogical(crate::LogicalOp::Not)))
    {
        return scoped_bool_sub_plan(store);
    }
    if matches!(value.kind(), UOpKind::GraphCompare(crate::CompareOp::Eq)) {
        return scoped_eq_plan(store, sm);
    }
    if matches!(value.kind(), UOpKind::GraphCompare(crate::CompareOp::Ne)) {
        if let Some(mode) = scoped_logical_not_plan(store, sm)? { return Ok(Some(mode)); }
        if let Some(mode) = scoped_inclusive_lt_plan(store, sm)? { return Ok(Some(mode)); }
        return scoped_ne_plan(store, sm);
    }
    if matches!(value.kind(), UOpKind::GraphCompare(crate::CompareOp::Lt)) {
        return scoped_ordered_lt_plan(store, sm);
    }
    if matches!(value.kind(), UOpKind::GraphUnary(crate::UnaryOp::IsInf)) {
        return scoped_isinf_plan(store, sm);
    }
    if matches!(value.kind(), UOpKind::Ternary(crate::uop::Ternary::Where)) {
        if let Some(mode) = scoped_relu_plan(store, sm)? {
            return Ok(Some(mode));
        }
        if let Some(mode) = scoped_leaky_relu_plan(store, sm)? {
            return Ok(Some(mode));
        }
        if let Some(mode) = scoped_clamp_plan(store, sm)? {
            return Ok(Some(mode));
        }
        return scoped_select_plan(store, sm);
    }
    if matches!(value.kind(), UOpKind::GraphBinary(crate::BinaryOp::Add)) {
        return scoped_binary_plan(store, sm, crate::BinaryOp::Add, ScopedStorageMode::Add);
    }
    if matches!(value.kind(), UOpKind::GraphBinary(crate::BinaryOp::Sub)) {
        if value.ty().map(|ty| ty.scalar) == Some(DType::Bool) {
            return Err(PtxError::Unsupported("raw Bool Sub is not public subtraction".into()));
        }
        return scoped_binary_plan(store, sm, crate::BinaryOp::Sub, ScopedStorageMode::Sub);
    }
    if matches!(value.kind(), UOpKind::GraphBinary(crate::BinaryOp::Mul))
        && matches!(
            value.sources().get(1).map(|node| node.kind()),
            Some(UOpKind::GraphUnary(crate::UnaryOp::Reciprocal))
        )
    {
        return scoped_div_plan(store, sm);
    }
    if matches!(value.kind(), UOpKind::GraphBinary(crate::BinaryOp::Mul))
        && !matches!(
            value.sources().get(1).map(|node| node.kind()),
            Some(UOpKind::GraphUnary(crate::UnaryOp::Sign))
        )
    {
        return scoped_binary_plan(store, sm, crate::BinaryOp::Mul, ScopedStorageMode::Mul);
    }
    if let UOpKind::GraphBinary(op @ (crate::BinaryOp::Maximum | crate::BinaryOp::Minimum)) = value.kind() {
        return scoped_binary_plan(store, sm, *op, ScopedStorageMode::Extrema);
    }
    if matches!(value.kind(), UOpKind::GraphUnary(crate::UnaryOp::Reciprocal)) {
        if let Some(mode) = scoped_rsqrt_plan(store, sm)? {
            return Ok(Some(mode));
        }
    }
    let (load, mode) = match value.kind() {
        UOpKind::GraphUnary(crate::UnaryOp::Sign) => {
            let [load] = value.sources() else {
                return Err(PtxError::Unsupported("Sign must have one input".into()));
            };
            (load, ScopedStorageMode::Sign)
        }
        UOpKind::GraphBinary(crate::BinaryOp::Mul) => {
            let [input, sign] = value.sources() else {
                return Err(PtxError::Unsupported("Abs Mul must have two inputs".into()));
            };
            let UOpKind::GraphUnary(crate::UnaryOp::Sign) = sign.kind() else {
                return Ok(None);
            };
            let [sign_input] = sign.sources() else {
                return Err(PtxError::Unsupported("Abs Sign must have one input".into()));
            };
            if input != sign_input {
                return Ok(None);
            }
            (input, ScopedStorageMode::Abs)
        }
        UOpKind::GraphUnary(crate::UnaryOp::Neg) => {
            let [load] = value.sources() else {
                return Err(PtxError::Unsupported("Neg must have one input".into()));
            };
            (load, ScopedStorageMode::Neg)
        }
        UOpKind::GraphLogical(crate::LogicalOp::Not) => {
            let [load] = value.sources() else {
                return Err(PtxError::Unsupported("logical Neg must have one input".into()));
            };
            (load, ScopedStorageMode::NegBool)
        }
        UOpKind::GraphUnary(crate::UnaryOp::Reciprocal) => {
            let [reciprocal_input] = value.sources() else {
                return Err(PtxError::Unsupported("Reciprocal must have one input".into()));
            };
            if matches!(reciprocal_input.kind(), UOpKind::Load) {
                (reciprocal_input, ScopedStorageMode::Reciprocal)
            } else {
                let UOpKind::Cast = reciprocal_input.kind() else {
                    return Ok(None);
                };
                let [load] = reciprocal_input.sources() else {
                    return Err(PtxError::Unsupported("Reciprocal Cast must have one input".into()));
                };
                if !matches!(load.kind(), UOpKind::Load)
                    || reciprocal_input.ty().map(|ty| ty.scalar) != Some(DType::F32)
                {
                    return Ok(None);
                }
                (load, ScopedStorageMode::ReciprocalCast)
            }
        }
        UOpKind::GraphUnary(crate::UnaryOp::Sqrt) => {
            let [sqrt_input] = value.sources() else {
                return Err(PtxError::Unsupported("Sqrt must have one input".into()));
            };
            if matches!(sqrt_input.kind(), UOpKind::Load) {
                (sqrt_input, ScopedStorageMode::Sqrt)
            } else {
                let UOpKind::Cast = sqrt_input.kind() else { return Ok(None) };
                let [load] = sqrt_input.sources() else {
                    return Err(PtxError::Unsupported("Sqrt Cast must have one input".into()));
                };
                if !matches!(load.kind(), UOpKind::Load)
                    || sqrt_input.ty().map(|ty| ty.scalar) != Some(DType::F32)
                {
                    return Ok(None);
                }
                (load, ScopedStorageMode::SqrtCast)
            }
        }
        _ => return Ok(None),
    };
    if !matches!(load.kind(), UOpKind::Load) || load.sources().len() != 1 {
        return Err(PtxError::Unsupported(
            "scoped narrow-storage ABI requires a direct load".into(),
        ));
    }
    let Some(output_index) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let Some(input_index) = load.sources().first() else {
        return Err(PtxError::Unsupported("scoped load without index".into()));
    };
    if !matches!(output_index.kind(), UOpKind::Index)
        || !matches!(input_index.kind(), UOpKind::Index)
    {
        return Err(PtxError::Unsupported(
            "scoped narrow-storage ABI requires typed indices".into(),
        ));
    }
    let dtype = value
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped scoped operation".into()))?
        .scalar;
    if mode == ScopedStorageMode::Abs
        && value.sources()[1].ty().map(|ty| ty.scalar) != Some(dtype)
    {
        return Err(PtxError::Unsupported(
            "Abs narrow-storage ABI requires a preserved Sign dtype".into(),
        ));
    }
    if mode == ScopedStorageMode::NegBool && dtype != DType::Bool {
        return Err(PtxError::Unsupported(
            "logical Neg narrow-storage ABI requires Bool".into(),
        ));
    }
    if mode == ScopedStorageMode::Neg && dtype == DType::Bool {
        return Err(PtxError::Unsupported(
            "numeric Neg narrow-storage ABI excludes Bool".into(),
        ));
    }
    let input_dtype = load
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped scoped input".into()))?
        .scalar;
    let reciprocal_direct = mode == ScopedStorageMode::Reciprocal;
    let reciprocal_cast = mode == ScopedStorageMode::ReciprocalCast;
    let sqrt_direct = mode == ScopedStorageMode::Sqrt;
    let sqrt_cast = mode == ScopedStorageMode::SqrtCast;
    if ((reciprocal_direct || sqrt_direct)
        && (!input_dtype.is_float() || input_dtype != dtype))
        || ((reciprocal_cast || sqrt_cast) && (input_dtype.is_float() || dtype != DType::F32))
        || (!reciprocal_direct
            && !reciprocal_cast
            && !sqrt_direct
            && !sqrt_cast
            && (input_dtype != dtype
                || output_index.ty().map(|ty| ty.scalar) != Some(dtype)
                || input_index.ty().map(|ty| ty.scalar) != Some(dtype)))
        || output_index.ty().map(|ty| ty.scalar) != Some(dtype)
        || input_index.ty().map(|ty| ty.scalar) != Some(input_dtype)
    {
        return Err(PtxError::Unsupported(
            "scoped storage ABI has incompatible input/output dtypes".into(),
        ));
    }
    let output_shape = match output_index.arg() {
        UArg::BufferIndex { output_shape, .. } => output_shape,
        _ => {
            return Err(PtxError::Unsupported(
                "scoped narrow-storage ABI requires a concrete output buffer".into(),
            ));
        }
    };
    if (sqrt_direct || sqrt_cast) && matches!(input_index.arg(), UArg::ViewBufferIndex { .. }) {
        return Err(PtxError::Unsupported(
            "scoped Sqrt does not admit affine-view inputs".into(),
        ));
    }
    let input_shape = match input_index.arg() {
        UArg::BufferIndex { output_shape, .. } | UArg::ViewBufferIndex { output_shape, .. } => {
            output_shape
        }
        _ => {
            return Err(PtxError::Unsupported(
                "scoped narrow-storage ABI requires a concrete input buffer".into(),
            ));
        }
    };
    if input_shape != output_shape
        || output_shape.numel().map_err(|_| PtxError::Overflow)?
            != match output_index.arg() {
                UArg::BufferIndex { elements, .. } => *elements,
                _ => unreachable!(),
            }
    {
        return Err(PtxError::Unsupported(
            "scoped narrow-storage ABI requires matching concrete extents".into(),
        ));
    }
    reject_sign_storage_dtype(dtype)?;
    if dtype == DType::F16 && sm < 53 && mode != ScopedStorageMode::Neg {
        return Err(PtxError::Unsupported(
            "F16 scoped storage conversion requires sm_53 or newer".into(),
        ));
    }
    Ok(Some(mode))
}

/// Public rsqrt is not raw RSQRT: tinygrad literally composes a typed SQRT
/// result with Reciprocal. Prove the entire unary chain so the renderer can
/// retain the otherwise-fused SQRT storage boundary without admitting generic
/// unary compounds.
fn scoped_rsqrt_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let [output, reciprocal] = store.sources() else { return Err(PtxError::Unsupported("Rsqrt Store needs index and value".into())) };
    let UOpKind::GraphUnary(crate::UnaryOp::Reciprocal) = reciprocal.kind() else { return Ok(None) };
    let [sqrt] = reciprocal.sources() else { return Err(PtxError::Unsupported("Rsqrt Reciprocal needs Sqrt".into())) };
    let UOpKind::GraphUnary(crate::UnaryOp::Sqrt) = sqrt.kind() else { return Ok(None) };
    let [sqrt_input] = sqrt.sources() else { return Err(PtxError::Unsupported("Rsqrt Sqrt needs input".into())) };
    let UArg::BufferIndex { elements, output_shape, .. } = output.arg() else {
        return Err(PtxError::Unsupported("Rsqrt needs a concrete output".into()));
    };
    let dtype = reciprocal.ty().ok_or_else(|| PtxError::Unsupported("untyped Rsqrt output".into()))?.scalar;
    if !dtype.is_float()
        || sqrt.ty().map(|ty| ty.scalar) != Some(dtype)
        || output.ty().map(|ty| ty.scalar) != Some(dtype)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
        || elements.checked_mul(dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("Rsqrt result descriptor is invalid".into()));
    }
    let (load, cast) = match sqrt_input.kind() {
        UOpKind::Load => (sqrt_input, None),
        UOpKind::Cast => {
            let [load] = sqrt_input.sources() else { return Err(PtxError::Unsupported("Rsqrt Cast arity".into())) };
            if !matches!(load.kind(), UOpKind::Load) { return Err(PtxError::Unsupported("Rsqrt Cast needs direct load".into())) }
            (load, Some(sqrt_input))
        }
        _ => return Err(PtxError::Unsupported("Rsqrt needs direct load or public F32 Cast".into())),
    };
    let [index] = load.sources() else { return Err(PtxError::Unsupported("Rsqrt load needs index".into())) };
    let UArg::BufferIndex { elements: input_elements, input_shape, output_shape: input_output, .. } = index.arg() else {
        return Err(PtxError::Unsupported("Rsqrt does not admit affine views".into()));
    };
    let input_dtype = load.ty().ok_or_else(|| PtxError::Unsupported("untyped Rsqrt input".into()))?.scalar;
    if index.ty().map(|ty| ty.scalar) != Some(input_dtype)
        || input_output != output_shape
        || input_shape != output_shape
        || input_shape.numel().map_err(|_| PtxError::Overflow)? != *input_elements
        || input_elements.checked_mul(input_dtype.itemsize()).is_none()
        || sqrt_input.ty().map(|ty| ty.scalar) != Some(dtype)
        || ((input_dtype == dtype) != cast.is_none())
        || (cast.is_some() && (input_dtype.is_float() || dtype != DType::F32))
        || (cast.is_none() && (!input_dtype.is_float() || input_dtype != dtype))
    {
        return Err(PtxError::Unsupported("Rsqrt input/cast chain is not source-exact".into()));
    }
    if (input_dtype == DType::F16 || dtype == DType::F16) && sm < 53 {
        return Err(PtxError::Unsupported("F16 public Rsqrt requires sm_53 or newer".into()));
    }
    reject_sign_storage_dtype(input_dtype)?;
    reject_sign_storage_dtype(dtype)?;
    Ok(Some(ScopedStorageMode::Rsqrt))
}

/// A scoped public binary exception is intentionally a whole-root proof, not
/// a generic binary admission. Each operand is a direct public input or its
/// one source-LUB Cast, and both index descriptors are the exact output
/// broadcast domain.
fn scoped_binary_plan(
    store: &UOp,
    sm: u32,
    op: crate::BinaryOp,
    mode: ScopedStorageMode,
) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    let UOpKind::GraphBinary(actual_op) = value.kind() else {
        return Ok(None);
    };
    if *actual_op != op {
        return Ok(None);
    }
    let [left, right] = value.sources() else {
        return Err(PtxError::Unsupported("Mul must have two inputs".into()));
    };
    let Some(output_index) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let UArg::BufferIndex {
        elements: output_elements,
        output_shape,
        ..
    } = output_index.arg()
    else {
        return Err(PtxError::Unsupported(
            "scoped Mul requires a concrete output buffer".into(),
        ));
    };
    let output_dtype = value
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped Mul output".into()))?
        .scalar;
    if output_index.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(output_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported(
            "scoped Mul output descriptor is invalid".into(),
        ));
    }

    fn operand<'a>(node: &'a UOp) -> Result<(&'a UOp, Option<&'a UOp>), PtxError> {
        match node.kind() {
            UOpKind::Load => Ok((node, None)),
            UOpKind::Cast => {
                let [load] = node.sources() else {
                    return Err(PtxError::Unsupported("Mul Cast must have one input".into()));
                };
                if !matches!(load.kind(), UOpKind::Load) {
                    return Err(PtxError::Unsupported(
                        "scoped Mul Cast must consume a direct load".into(),
                    ));
                }
                Ok((load, Some(node)))
            }
            _ => Err(PtxError::Unsupported(
                "scoped Mul needs only direct loads and source casts".into(),
            )),
        }
    }
    fn index<'a>(load: &'a UOp) -> Result<&'a UOp, PtxError> {
        let [index] = load.sources() else {
            return Err(PtxError::Unsupported("Mul load must have one index".into()));
        };
        if !matches!(index.kind(), UOpKind::Index) {
            return Err(PtxError::Unsupported("Mul load needs a typed index".into()));
        }
        Ok(index)
    }
    let (left_load, left_cast) = operand(left)?;
    let (right_load, right_cast) = operand(right)?;
    let left_dtype = left_load
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped Mul lhs".into()))?
        .scalar;
    let right_dtype = right_load
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped Mul rhs".into()))?
        .scalar;
    let source_dtype = if matches!(
        (left_dtype, right_dtype),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        left_dtype.promote(right_dtype)
    };
    if source_dtype != output_dtype || left.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || right.ty().map(|ty| ty.scalar) != Some(output_dtype)
    {
        return Err(PtxError::Unsupported(
            "scoped Mul output does not match source promotion".into(),
        ));
    }
    for (load, cast, source_dtype) in [
        (left_load, left_cast, left_dtype),
        (right_load, right_cast, right_dtype),
    ] {
        if (source_dtype == output_dtype) != cast.is_none()
            || cast.is_some_and(|node| node.ty().map(|ty| ty.scalar) != Some(output_dtype))
        {
            return Err(PtxError::Unsupported(
                "scoped Mul operands must use exactly the source LUB casts".into(),
            ));
        }
        let input_index = index(load)?;
        let UArg::BufferIndex {
            elements,
            input_shape,
            output_shape: operand_output,
            ..
        } = input_index.arg()
        else {
            return Err(PtxError::Unsupported(
                "scoped Mul does not admit affine-view operands".into(),
            ));
        };
        if input_index.ty().map(|ty| ty.scalar) != Some(source_dtype)
            || operand_output != output_shape
            || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
            || elements.checked_mul(source_dtype.itemsize()).is_none()
        {
            return Err(PtxError::Unsupported(
                "scoped Mul operand descriptor is invalid".into(),
            ));
        }
    }
    let left_shape = match index(left_load)?.arg() {
        UArg::BufferIndex { input_shape, .. } => input_shape,
        _ => unreachable!(),
    };
    let right_shape = match index(right_load)?.arg() {
        UArg::BufferIndex { input_shape, .. } => input_shape,
        _ => unreachable!(),
    };
    if left_shape
        .broadcast_with(right_shape)
        .map_err(|_| PtxError::Unsupported("scoped Mul broadcast is invalid".into()))?
        != output_shape.clone()
    {
        return Err(PtxError::Unsupported(
            "scoped Mul operands do not produce the output broadcast shape".into(),
        ));
    }
    if (output_dtype == DType::F16
        || (mode == ScopedStorageMode::Extrema
            && (left_dtype == DType::F16 || right_dtype == DType::F16)))
        && sm < 53
    {
        return Err(PtxError::Unsupported(
            "F16 scoped binary conversion requires sm_53 or newer".into(),
        ));
    }
    reject_sign_storage_dtype(output_dtype)?;
    Ok(Some(mode))
}

/// Proves a public source-LUB comparison value against its concrete output
/// domain. Store roots and predicate-Select roots share this exact proof, so
/// neither path can admit an affine input or arbitrary predicate compound.
fn scoped_compare_value_proof(
    value: &UOp,
    domain_shape: &Shape,
    sm: u32,
    expected: crate::CompareOp,
) -> Result<Option<Shape>, PtxError> {
    let UOpKind::GraphCompare(actual) = value.kind() else {
        return Ok(None);
    };
    if *actual != expected {
        return Ok(None);
    }
    let [left, right] = value.sources() else {
        return Err(PtxError::Unsupported("public Eq needs two inputs".into()));
    };
    if value.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || domain_shape.numel().map_err(|_| PtxError::Overflow)?.checked_mul(DType::Bool.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public Eq output descriptor is invalid".into()));
    }
    fn operand<'a>(node: &'a UOp) -> Result<(&'a UOp, Option<&'a UOp>), PtxError> {
        match node.kind() {
            UOpKind::Load => Ok((node, None)),
            UOpKind::Cast => {
                let [load] = node.sources() else {
                    return Err(PtxError::Unsupported("public Eq Cast needs one input".into()));
                };
                if !matches!(load.kind(), UOpKind::Load) {
                    return Err(PtxError::Unsupported("public Eq Cast must consume a direct load".into()));
                }
                Ok((load, Some(node)))
            }
            _ => Err(PtxError::Unsupported("public Eq needs only direct loads and source casts".into())),
        }
    }
    fn index<'a>(load: &'a UOp, output: &Shape) -> Result<&'a UOp, PtxError> {
        let [index] = load.sources() else {
            return Err(PtxError::Unsupported("public Eq load needs one index".into()));
        };
        let UArg::BufferIndex { elements, input_shape, output_shape, .. } = index.arg() else {
            return Err(PtxError::Unsupported("public Eq does not admit affine-view inputs".into()));
        };
        let dtype = load.ty().ok_or_else(|| PtxError::Unsupported("untyped Eq input".into()))?.scalar;
        if index.ty().map(|ty| ty.scalar) != Some(dtype)
            || output_shape != output
            || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
            || elements.checked_mul(dtype.itemsize()).is_none()
        {
            return Err(PtxError::Unsupported("public Eq input descriptor is invalid".into()));
        }
        Ok(index)
    }
    let (left_load, left_cast) = operand(left)?;
    let (right_load, right_cast) = operand(right)?;
    let left_dtype = left_load.ty().ok_or_else(|| PtxError::Unsupported("untyped Eq lhs".into()))?.scalar;
    let right_dtype = right_load.ty().ok_or_else(|| PtxError::Unsupported("untyped Eq rhs".into()))?.scalar;
    let comparison_dtype = if matches!((left_dtype, right_dtype), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
        DType::F32
    } else {
        left_dtype.promote(right_dtype)
    };
    if left.ty().map(|ty| ty.scalar) != Some(comparison_dtype)
        || right.ty().map(|ty| ty.scalar) != Some(comparison_dtype)
    {
        return Err(PtxError::Unsupported("public Eq operands do not use source promotion".into()));
    }
    for (load, cast, source_dtype) in [(left_load, left_cast, left_dtype), (right_load, right_cast, right_dtype)] {
        if (source_dtype == comparison_dtype) != cast.is_none()
            || cast.is_some_and(|node| node.ty().map(|ty| ty.scalar) != Some(comparison_dtype))
        {
            return Err(PtxError::Unsupported("public Eq must use exactly the source LUB casts".into()));
        }
    }
    let left_index = index(left_load, domain_shape)?;
    let right_index = index(right_load, domain_shape)?;
    let left_shape = match left_index.arg() { UArg::BufferIndex { input_shape, .. } => input_shape, _ => unreachable!() };
    let right_shape = match right_index.arg() { UArg::BufferIndex { input_shape, .. } => input_shape, _ => unreachable!() };
    let comparison_shape = left_shape.broadcast_with(right_shape)
        .map_err(|_| PtxError::Unsupported("public Eq broadcast is invalid".into()))?;
    if comparison_shape.numel().map_err(|_| PtxError::Overflow)?.checked_mul(comparison_dtype.itemsize()).is_none() {
        return Err(PtxError::Unsupported("public Eq broadcast/output extent is invalid".into()));
    }
    if matches!(left_dtype, DType::F16) || matches!(right_dtype, DType::F16) || comparison_dtype == DType::F16 {
        if sm < 53 {
            return Err(PtxError::Unsupported("F16 public Eq conversion requires sm_53 or newer".into()));
        }
    }
    reject_sign_storage_dtype(left_dtype)?;
    reject_sign_storage_dtype(right_dtype)?;
    reject_sign_storage_dtype(comparison_dtype)?;
    Ok(Some(comparison_shape))
}

/// Store wrapper for the reusable source-LUB comparison value proof.
fn scoped_compare_value_plan(
    store: &UOp,
    sm: u32,
    expected: crate::CompareOp,
    mode: ScopedStorageMode,
) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    let Some(output_index) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let UArg::BufferIndex { elements: output_elements, output_shape, .. } = output_index.arg() else {
        return Err(PtxError::Unsupported("public Eq requires a concrete output buffer".into()));
    };
    if output_index.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(DType::Bool.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public Eq output descriptor is invalid".into()));
    }
    if scoped_compare_value_proof(value, output_shape, sm, expected)?.as_ref() != Some(output_shape) {
        return Ok(None);
    }
    Ok(Some(mode))
}

fn scoped_eq_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    scoped_compare_value_plan(store, sm, crate::CompareOp::Eq, ScopedStorageMode::Eq)
}

fn scoped_ne_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    scoped_compare_value_plan(store, sm, crate::CompareOp::Ne, ScopedStorageMode::Ne)
}

/// Public logical-not is precisely tinygrad's `Cast(Bool, input) != True`.
/// Keep the cast explicit in the proof: otherwise a raw Bool `Ne` could look
/// like a public logical-not root after UOp lowering.
fn scoped_logical_not_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(output) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let Some(value) = store.sources().get(1) else { return Ok(None) };
    let UOpKind::GraphCompare(crate::CompareOp::Ne) = value.kind() else { return Ok(None) };
    let [cast, truth] = value.sources() else { return Ok(None) };
    let UOpKind::Cast = cast.kind() else { return Ok(None) };
    let [load] = cast.sources() else { return Ok(None) };
    if !matches!(load.kind(), UOpKind::Load)
        || cast.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || value.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || truth.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || !matches!(truth.kind(), UOpKind::Const)
        || !matches!(truth.arg(), UArg::Scalar { dtype: DType::Bool, bits: 1 })
        || !truth.sources().is_empty()
    {
        return Ok(None);
    }
    let UArg::BufferIndex { elements: output_elements, output_shape, .. } = output.arg() else {
        return Err(PtxError::Unsupported("public logical-not requires a concrete output buffer".into()));
    };
    if output.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(DType::Bool.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public logical-not output descriptor is invalid".into()));
    }
    let [input] = load.sources() else {
        return Err(PtxError::Unsupported("public logical-not load needs one index".into()));
    };
    let input_dtype = load.ty().ok_or_else(|| PtxError::Unsupported("untyped public logical-not input".into()))?.scalar;
    let UArg::BufferIndex { elements, input_shape, output_shape: input_output, .. } = input.arg() else {
        return Err(PtxError::Unsupported("public logical-not does not admit affine-view inputs".into()));
    };
    if input.ty().map(|ty| ty.scalar) != Some(input_dtype)
        || input_shape != output_shape
        || input_output != output_shape
        || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
        || elements.checked_mul(input_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public logical-not input descriptor is invalid".into()));
    }
    if input_dtype == DType::F16 && sm < 53 {
        return Err(PtxError::Unsupported("F16 public logical-not conversion requires sm_53 or newer".into()));
    }
    reject_sign_storage_dtype(input_dtype)?;
    Ok(Some(ScopedStorageMode::LogicalNot))
}

/// Public IsInf is one direct raw predicate root. Its F16/BF16/F32/F64
/// classification is performed from storage bits by the renderer, so NaN
/// payloads cannot be confused with either infinity and integers stay false.
fn scoped_isinf_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(output) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let Some(value) = store.sources().get(1) else { return Ok(None) };
    let UOpKind::GraphUnary(crate::UnaryOp::IsInf) = value.kind() else { return Ok(None) };
    let [load] = value.sources() else { return Ok(None) };
    if !matches!(load.kind(), UOpKind::Load)
        || value.ty().map(|ty| ty.scalar) != Some(DType::Bool)
    {
        return Ok(None);
    }
    let UArg::BufferIndex { elements: output_elements, output_shape, .. } = output.arg() else {
        return Err(PtxError::Unsupported("public IsInf requires a concrete output buffer".into()));
    };
    if output.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(DType::Bool.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public IsInf output descriptor is invalid".into()));
    }
    let [input] = load.sources() else {
        return Err(PtxError::Unsupported("public IsInf load needs one index".into()));
    };
    let input_dtype = load.ty().ok_or_else(|| PtxError::Unsupported("untyped public IsInf input".into()))?.scalar;
    let UArg::BufferIndex { elements, input_shape, output_shape: input_output, .. } = input.arg() else {
        return Err(PtxError::Unsupported("public IsInf does not admit affine-view inputs".into()));
    };
    if input.ty().map(|ty| ty.scalar) != Some(input_dtype)
        || input_shape != output_shape
        || input_output != output_shape
        || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
        || elements.checked_mul(input_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public IsInf input descriptor is invalid".into()));
    }
    if input_dtype == DType::F16 && sm < 53 {
        return Err(PtxError::Unsupported("F16 public IsInf conversion requires sm_53 or newer".into()));
    }
    reject_sign_storage_dtype(input_dtype)?;
    Ok(Some(ScopedStorageMode::IsInf))
}

fn scoped_inclusive_lt_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(output) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let Some(outer) = store.sources().get(1) else { return Ok(None) };
    let UOpKind::GraphCompare(crate::CompareOp::Ne) = outer.kind() else { return Ok(None) };
    let [cast, truth] = outer.sources() else { return Ok(None) };
    let UOpKind::Cast = cast.kind() else { return Ok(None) };
    let [inner] = cast.sources() else { return Ok(None) };
    if outer.ty().map(|t| t.scalar) != Some(DType::Bool)
        || cast.ty().map(|t| t.scalar) != Some(DType::Bool)
        || inner.ty().map(|t| t.scalar) != Some(DType::Bool)
        || !matches!(inner.kind(), UOpKind::GraphCompare(crate::CompareOp::Lt))
        || !matches!(truth.kind(), UOpKind::Const)
        || !matches!(truth.arg(), UArg::Scalar { dtype: DType::Bool, bits: 1 })
        || truth.ty().map(|t| t.scalar) != Some(DType::Bool) { return Ok(None) }
    let proof = UOp::new(UOpKind::Store, None, vec![output.clone(), inner.clone()], UArg::None);
    if scoped_compare_value_plan(&proof, sm, crate::CompareOp::Lt, ScopedStorageMode::OrderedLt)?.is_none() { return Ok(None) }
    Ok(Some(ScopedStorageMode::InclusiveLt))
}

/// This admits both public Less and public Greater: tinygrad Greater is the
/// literal reversed-input CMPLT graph, so it is intentionally structurally
/// equivalent to Less with the same reversed branches.
fn scoped_ordered_lt_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    scoped_compare_value_plan(store, sm, crate::CompareOp::Lt, ScopedStorageMode::OrderedLt)
}

/// Returns the source comparison shape only for an already-admitted public
/// predicate value.  `Ne` is either direct unordered-ne or the exact public
/// inclusive shell `Cast(Bool, Lt) != Const(Bool(true))`.
fn scoped_select_predicate_shape(
    value: &UOp,
    domain_shape: &Shape,
    sm: u32,
) -> Result<Option<Shape>, PtxError> {
    match value.kind() {
        UOpKind::GraphCompare(crate::CompareOp::Eq) => {
            scoped_compare_value_proof(value, domain_shape, sm, crate::CompareOp::Eq)
        }
        UOpKind::GraphCompare(crate::CompareOp::Lt) => {
            scoped_compare_value_proof(value, domain_shape, sm, crate::CompareOp::Lt)
        }
        UOpKind::GraphCompare(crate::CompareOp::Ne) => {
            if let Some(shape) = scoped_compare_value_proof(value, domain_shape, sm, crate::CompareOp::Ne)? {
                return Ok(Some(shape));
            }
            let [cast, truth] = value.sources() else { return Ok(None) };
            let UOpKind::Cast = cast.kind() else { return Ok(None) };
            let [inner] = cast.sources() else { return Ok(None) };
            if value.ty().map(|ty| ty.scalar) != Some(DType::Bool)
                || cast.ty().map(|ty| ty.scalar) != Some(DType::Bool)
                || inner.ty().map(|ty| ty.scalar) != Some(DType::Bool)
                || !matches!(truth.kind(), UOpKind::Const)
                || !matches!(truth.arg(), UArg::Scalar { dtype: DType::Bool, bits: 1 })
                || truth.ty().map(|ty| ty.scalar) != Some(DType::Bool)
            {
                return Ok(None);
            }
            scoped_compare_value_proof(inner, domain_shape, sm, crate::CompareOp::Lt)
        }
        _ => Ok(None),
    }
}

/// ReLU is the one scalar-constant comparison/Select root admitted by this
/// phase. The checked-in public helper literally lowers `zero < input` then
/// `where(input, zero)`. Keeping the complete shape here prevents a generic
/// scalar predicate or constant payload from inheriting the narrow ABI.
fn scoped_relu_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    if store.sources().len() != 2 {
        return Err(PtxError::Unsupported(
            "public ReLU Store needs exactly an index and value".into(),
        ));
    }
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    let UOpKind::Ternary(crate::uop::Ternary::Where) = value.kind() else {
        return Ok(None);
    };
    let [condition, on_true, on_false] = value.sources() else {
        return Err(PtxError::Unsupported("public ReLU needs three Select inputs".into()));
    };
    let Some(output) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let UArg::BufferIndex {
        elements: output_elements,
        output_shape,
        ..
    } = output.arg()
    else {
        return Err(PtxError::Unsupported(
            "public ReLU requires a concrete output buffer".into(),
        ));
    };
    let output_dtype = output
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped public ReLU output".into()))?
        .scalar;
    if output.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || value.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(output_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported(
            "public ReLU output descriptor is invalid".into(),
        ));
    }

    let UOpKind::GraphCompare(crate::CompareOp::Lt) = condition.kind() else {
        return Ok(None);
    };
    let [zero, input] = condition.sources() else {
        return Err(PtxError::Unsupported(
            "public ReLU ordered predicate needs two inputs".into(),
        ));
    };
    if input != on_true || zero != on_false {
        return Ok(None);
    }
    if condition.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || input.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || zero.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || on_true.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || on_false.ty().map(|ty| ty.scalar) != Some(output_dtype)
    {
        return Err(PtxError::Unsupported(
            "public ReLU predicate/payload dtypes are invalid".into(),
        ));
    }
    if !matches!(zero.kind(), UOpKind::Const)
        || !matches!(zero.arg(), UArg::Scalar { dtype, bits: 0 } if *dtype == output_dtype)
        || !zero.sources().is_empty()
    {
        return Ok(None);
    }
    let UOpKind::Load = input.kind() else {
        return Ok(None);
    };
    let [input_index] = input.sources() else {
        return Err(PtxError::Unsupported(
            "public ReLU input load needs one index".into(),
        ));
    };
    let UArg::BufferIndex {
        elements: input_elements,
        input_shape,
        output_shape: input_output_shape,
        ..
    } = input_index.arg()
    else {
        return Err(PtxError::Unsupported(
            "public ReLU does not admit affine-view input".into(),
        ));
    };
    if input_index.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || input_output_shape != output_shape
        || input_shape != output_shape
        || input_shape.numel().map_err(|_| PtxError::Overflow)? != *input_elements
        || input_elements.checked_mul(output_dtype.itemsize()).is_none()
        || output_shape
            .numel()
            .map_err(|_| PtxError::Overflow)?
            .checked_mul(DType::Bool.itemsize())
            .is_none()
    {
        return Err(PtxError::Unsupported(
            "public ReLU input/predicate descriptor is invalid".into(),
        ));
    }
    if output_dtype == DType::F16 && sm < 53 {
        return Err(PtxError::Unsupported(
            "F16 public ReLU conversion requires sm_53 or newer".into(),
        ));
    }
    reject_sign_storage_dtype(output_dtype)?;
    Ok(Some(ScopedStorageMode::Relu))
}

/// LeakyReLU owns a single compound exception: tinygrad's literal
/// `(input < zero).where(slope * input, input)`.  The proof deliberately
/// ties the predicate load, Mul rhs, and false branch to one graph input, and
/// admits no other scalar-constant Compare/Select or Mul/Select composition.
fn scoped_leaky_relu_plan(
    store: &UOp,
    sm: u32,
) -> Result<Option<ScopedStorageMode>, PtxError> {
    if store.sources().len() != 2 {
        return Err(PtxError::Unsupported(
            "public LeakyReLU Store needs exactly an index and value".into(),
        ));
    }
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    let UOpKind::Ternary(crate::uop::Ternary::Where) = value.kind() else {
        return Ok(None);
    };
    let [condition, scaled, input_value] = value.sources() else {
        return Err(PtxError::Unsupported(
            "public LeakyReLU needs three Select inputs".into(),
        ));
    };
    let Some(output) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let UArg::BufferIndex {
        elements: output_elements,
        output_shape,
        ..
    } = output.arg()
    else {
        return Err(PtxError::Unsupported(
            "public LeakyReLU requires a concrete output buffer".into(),
        ));
    };
    let output_dtype = output
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped public LeakyReLU output".into()))?
        .scalar;
    if value.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(output_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported(
            "public LeakyReLU output descriptor is invalid".into(),
        ));
    }

    let UOpKind::GraphCompare(crate::CompareOp::Lt) = condition.kind() else {
        return Ok(None);
    };
    let [predicate_input, zero] = condition.sources() else {
        return Err(PtxError::Unsupported(
            "public LeakyReLU ordered predicate needs two inputs".into(),
        ));
    };
    let UOpKind::GraphBinary(crate::BinaryOp::Mul) = scaled.kind() else {
        return Ok(None);
    };
    let [slope_value, scaled_input] = scaled.sources() else {
        return Err(PtxError::Unsupported(
            "public LeakyReLU scale branch needs slope * input".into(),
        ));
    };
    if scaled_input != input_value {
        return Ok(None);
    }
    if condition.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || scaled.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || input_value.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || !matches!(zero.kind(), UOpKind::Const)
        || !matches!(zero.arg(), UArg::Scalar { bits: 0, .. })
        || !zero.sources().is_empty()
    {
        return Ok(None);
    }

    fn direct_load<'a>(node: &'a UOp, role: &str) -> Result<&'a UOp, PtxError> {
        if !matches!(node.kind(), UOpKind::Load) {
            return Err(PtxError::Unsupported(format!(
                "public LeakyReLU {role} must be a direct load"
            )));
        }
        Ok(node)
    }
    fn source_value<'a>(node: &'a UOp, role: &str) -> Result<(&'a UOp, Option<&'a UOp>), PtxError> {
        match node.kind() {
            UOpKind::Load => Ok((node, None)),
            UOpKind::Cast => {
                let [load] = node.sources() else {
                    return Err(PtxError::Unsupported(format!(
                        "public LeakyReLU {role} Cast needs one input"
                    )));
                };
                Ok((direct_load(load, role)?, Some(node)))
            }
            _ => Err(PtxError::Unsupported(format!(
                "public LeakyReLU {role} needs a direct load or source cast"
            ))),
        }
    }
    fn descriptor<'a>(load: &'a UOp, output: &Shape, role: &str) -> Result<(&'a Shape, DType), PtxError> {
        let [index] = load.sources() else {
            return Err(PtxError::Unsupported(format!(
                "public LeakyReLU {role} load needs one index"
            )));
        };
        let UArg::BufferIndex {
            elements,
            input_shape,
            output_shape,
            ..
        } = index.arg()
        else {
            return Err(PtxError::Unsupported(format!(
                "public LeakyReLU {role} does not admit affine views"
            )));
        };
        let dtype = load
            .ty()
            .ok_or_else(|| PtxError::Unsupported(format!("untyped LeakyReLU {role}")))?
            .scalar;
        if index.ty().map(|ty| ty.scalar) != Some(dtype)
            || output_shape != output
            || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
            || elements.checked_mul(dtype.itemsize()).is_none()
        {
            return Err(PtxError::Unsupported(format!(
                "public LeakyReLU {role} descriptor is invalid"
            )));
        }
        Ok((input_shape, dtype))
    }

    let predicate_input = direct_load(predicate_input, "predicate input")?;
    let (input_shape, input_dtype) = descriptor(predicate_input, output_shape, "predicate input")?;
    if zero.ty().map(|ty| ty.scalar) != Some(input_dtype) {
        return Ok(None);
    }
    let (input_load, input_cast) = source_value(input_value, "input value")?;
    if input_load != predicate_input {
        return Ok(None);
    }
    let (slope_load, slope_cast) = source_value(slope_value, "slope value")?;
    let (slope_shape, slope_dtype) = descriptor(slope_load, output_shape, "slope value")?;
    let promotion = if matches!(
        (input_dtype, slope_dtype),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        input_dtype.promote(slope_dtype)
    };
    if promotion != output_dtype
        || input_value.ty().map(|ty| ty.scalar) != Some(promotion)
        || slope_value.ty().map(|ty| ty.scalar) != Some(promotion)
    {
        return Err(PtxError::Unsupported(
            "public LeakyReLU values do not use source promotion".into(),
        ));
    }
    for (_load, cast, source_dtype, role) in [
        (input_load, input_cast, input_dtype, "input value"),
        (slope_load, slope_cast, slope_dtype, "slope value"),
    ] {
        if (source_dtype == promotion) != cast.is_none()
            || cast.is_some_and(|node| node.ty().map(|ty| ty.scalar) != Some(promotion))
        {
            return Err(PtxError::Unsupported(format!(
                "public LeakyReLU {role} must use exactly one source-LUB cast"
            )));
        }
    }
    let value_shape = input_shape
        .broadcast_with(slope_shape)
        .map_err(|_| PtxError::Unsupported("public LeakyReLU Mul broadcast is invalid".into()))?;
    if input_shape
        .broadcast_with(&value_shape)
        .map_err(|_| PtxError::Unsupported("public LeakyReLU predicate broadcast is invalid".into()))?
        != output_shape.clone()
        || value_shape != output_shape.clone()
        || input_shape
            .numel()
            .map_err(|_| PtxError::Overflow)?
            .checked_mul(DType::Bool.itemsize())
            .is_none()
    {
        return Err(PtxError::Unsupported(
            "public LeakyReLU does not prove the three-way broadcast".into(),
        ));
    }
    if [input_dtype, slope_dtype, promotion].contains(&DType::F16) && sm < 53 {
        return Err(PtxError::Unsupported(
            "F16 public LeakyReLU conversion requires sm_53 or newer".into(),
        ));
    }
    reject_sign_storage_dtype(input_dtype)?;
    reject_sign_storage_dtype(slope_dtype)?;
    reject_sign_storage_dtype(promotion)?;
    Ok(Some(ScopedStorageMode::LeakyRelu))
}

/// The three public Clamp forms are strict ordered Compare/Select roots. A
/// lower stage is `value < bound ? bound : value`; an upper stage is the
/// literal reversed-Lt `bound < value ? bound : value`. The two-bound form
/// may only feed that exact lower result (or its required next-stage cast)
/// into the upper stage.
fn scoped_clamp_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let [output, root] = store.sources() else {
        return Err(PtxError::Unsupported("Clamp Store needs index and value".into()));
    };
    let UArg::BufferIndex { elements, output_shape, .. } = output.arg() else {
        return Err(PtxError::Unsupported("Clamp needs concrete output".into()));
    };
    let output_dtype = output.ty().ok_or_else(|| PtxError::Unsupported("untyped Clamp output".into()))?.scalar;
    if root.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
        || elements.checked_mul(output_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("Clamp output descriptor is invalid".into()));
    }
    fn source_lub(a: DType, b: DType) -> DType {
        if matches!((a, b), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
            DType::F32
        } else {
            a.promote(b)
        }
    }
    // A Clamp leaf is an input load or exactly one source-LUB cast of that
    // load. It deliberately excludes constants and affine views: this is an
    // operation-scoped public root, not generic nested Select admission.
    fn leaf<'a>(node: &'a UOp, target: DType, domain: &Shape) -> Result<(DType, &'a Shape), PtxError> {
        let (load, cast) = match node.kind() {
            UOpKind::Load => (node, None),
            UOpKind::Cast => {
                let [load] = node.sources() else {
                    return Err(PtxError::Unsupported("Clamp Cast arity".into()));
                };
                if !matches!(load.kind(), UOpKind::Load) {
                    return Err(PtxError::Unsupported("Clamp Cast needs direct load".into()));
                }
                (load, Some(node))
            }
            _ => return Err(PtxError::Unsupported("Clamp needs direct loads and source casts".into())),
        };
        let [index] = load.sources() else {
            return Err(PtxError::Unsupported("Clamp load arity".into()));
        };
        let UArg::BufferIndex { elements, input_shape, output_shape, .. } = index.arg() else {
            return Err(PtxError::Unsupported("Clamp does not admit affine views".into()));
        };
        let source = load.ty().ok_or_else(|| PtxError::Unsupported("untyped Clamp load".into()))?.scalar;
        if node.ty().map(|ty| ty.scalar) != Some(target)
            || (source == target) != cast.is_none()
            || output_shape != domain
            || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
            || elements.checked_mul(source.itemsize()).is_none()
        {
            return Err(PtxError::Unsupported("Clamp source-LUB leaf is invalid".into()));
        }
        Ok((source, input_shape))
    }
    fn parts<'a>(node: &'a UOp) -> Result<(&'a UOp, &'a UOp, &'a UOp, &'a UOp, &'a UOp, DType), PtxError> {
        let UOpKind::Ternary(crate::uop::Ternary::Where) = node.kind() else {
            return Err(PtxError::Unsupported("Clamp stage needs Select".into()));
        };
        let [condition, bound, value] = node.sources() else {
            return Err(PtxError::Unsupported("Clamp Select arity".into()));
        };
        let UOpKind::GraphCompare(crate::CompareOp::Lt) = condition.kind() else {
            return Err(PtxError::Unsupported("Clamp needs ordered Lt".into()));
        };
        let [left, right] = condition.sources() else {
            return Err(PtxError::Unsupported("Clamp comparison arity".into()));
        };
        let dtype = node.ty().ok_or_else(|| PtxError::Unsupported("untyped Clamp stage".into()))?.scalar;
        if condition.ty().map(|ty| ty.scalar) != Some(DType::Bool) {
            return Err(PtxError::Unsupported("Clamp predicate must be Bool".into()));
        }
        Ok((left, right, bound, value, condition, dtype))
    }
    fn extent(shape: &Shape, dtype: DType) -> Result<(), PtxError> {
        shape.numel().map_err(|_| PtxError::Overflow)?.checked_mul(dtype.itemsize()).ok_or(PtxError::Overflow)?;
        Ok(())
    }

    let (left, right, bound, value, _condition, root_dtype) = parts(root)?;
    let lower_root = left == value && right == bound;
    let upper_root = left == bound && right == value;
    if !lower_root && !upper_root {
        return Err(PtxError::Unsupported("Clamp predicate/branch order is not source-literal".into()));
    }

    let mut f16 = output_dtype == DType::F16 || root_dtype == DType::F16;
    let (bound_source, bound_shape) = leaf(bound, root_dtype, output_shape)?;
    f16 |= bound_source == DType::F16;

    if lower_root {
        let (value_source, value_shape) = leaf(value, root_dtype, output_shape)?;
        f16 |= value_source == DType::F16;
        let stage_shape = value_shape.broadcast_with(bound_shape)
            .map_err(|_| PtxError::Unsupported("Clamp lower broadcast is invalid".into()))?;
        if source_lub(value_source, bound_source) != root_dtype || stage_shape != output_shape.clone() {
            return Err(PtxError::Unsupported("Clamp lower descriptors/promotion are invalid".into()));
        }
        extent(&stage_shape, root_dtype)?;
        reject_sign_storage_dtype(value_source)?;
        reject_sign_storage_dtype(bound_source)?;
    } else {
        // The upper-only root has a leaf value. The two-bound root is the
        // only permitted nested form: its value is the exact lower Select,
        // optionally followed by the required next-stage source-LUB cast.
        let lower = match value.kind() {
            UOpKind::Ternary(crate::uop::Ternary::Where) => Some((value, false)),
            UOpKind::Cast => {
                let [inner] = value.sources() else {
                    return Err(PtxError::Unsupported("Clamp intermediate Cast arity".into()));
                };
                matches!(inner.kind(), UOpKind::Ternary(crate::uop::Ternary::Where)).then_some((inner, true))
            }
            _ => None,
        };
        if let Some((lower, casted)) = lower {
            let (lower_left, lower_right, lower_bound, lower_value, _lower_condition, lower_dtype) = parts(lower)?;
            if lower_left != lower_value || lower_right != lower_bound {
                return Err(PtxError::Unsupported("Clamp lower stage is not source-literal".into()));
            }
            if value.ty().map(|ty| ty.scalar) != Some(root_dtype)
                || (lower_dtype == root_dtype) != !casted
            {
                return Err(PtxError::Unsupported("Clamp intermediate storage boundary is invalid".into()));
            }
            let (input_source, input_shape) = leaf(lower_value, lower_dtype, output_shape)?;
            let (min_source, min_shape) = leaf(lower_bound, lower_dtype, output_shape)?;
            let lower_shape = input_shape.broadcast_with(min_shape)
                .map_err(|_| PtxError::Unsupported("Clamp lower broadcast is invalid".into()))?;
            let final_shape = lower_shape.broadcast_with(bound_shape)
                .map_err(|_| PtxError::Unsupported("Clamp upper broadcast is invalid".into()))?;
            if source_lub(input_source, min_source) != lower_dtype
                || source_lub(lower_dtype, bound_source) != root_dtype
                || final_shape != output_shape.clone()
            {
                return Err(PtxError::Unsupported("Clamp stage descriptors/promotion are invalid".into()));
            }
            extent(&lower_shape, lower_dtype)?;
            extent(&final_shape, root_dtype)?;
            f16 |= input_source == DType::F16 || min_source == DType::F16 || lower_dtype == DType::F16;
            reject_sign_storage_dtype(input_source)?;
            reject_sign_storage_dtype(min_source)?;
        } else {
            let (value_source, value_shape) = leaf(value, root_dtype, output_shape)?;
            let stage_shape = value_shape.broadcast_with(bound_shape)
                .map_err(|_| PtxError::Unsupported("Clamp upper broadcast is invalid".into()))?;
            if source_lub(value_source, bound_source) != root_dtype || stage_shape != output_shape.clone() {
                return Err(PtxError::Unsupported("Clamp upper descriptors/promotion are invalid".into()));
            }
            extent(&stage_shape, root_dtype)?;
            f16 |= value_source == DType::F16;
            reject_sign_storage_dtype(value_source)?;
        }
    }
    if f16 && sm < 53 {
        return Err(PtxError::Unsupported("F16 Clamp requires sm_53 or newer".into()));
    }
    reject_sign_storage_dtype(bound_source)?;
    reject_sign_storage_dtype(root_dtype)?;
    Ok(Some(ScopedStorageMode::Clamp))
}

/// Public `where` is the only ternary root admitted through the narrow PTX
/// ABI.  Its condition is a direct Bool input or an already-proven public
/// comparison value; each payload is a direct input or its source-LUB Cast.
fn scoped_select_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    let UOpKind::Ternary(crate::uop::Ternary::Where) = value.kind() else {
        return Ok(None);
    };
    let [condition, on_true, on_false] = value.sources() else {
        return Err(PtxError::Unsupported("public Select needs three inputs".into()));
    };
    let Some(output) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let UArg::BufferIndex { elements: output_elements, output_shape, .. } = output.arg() else {
        return Err(PtxError::Unsupported("public Select requires a concrete output buffer".into()));
    };
    let output_dtype = output.ty().ok_or_else(|| PtxError::Unsupported("untyped Select output".into()))?.scalar;
    if value.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(output_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public Select output descriptor is invalid".into()));
    }
    fn direct_load<'a>(node: &'a UOp, role: &str) -> Result<&'a UOp, PtxError> {
        if !matches!(node.kind(), UOpKind::Load) {
            return Err(PtxError::Unsupported(format!("public Select {role} must be a direct load")));
        }
        Ok(node)
    }
    fn payload<'a>(node: &'a UOp) -> Result<(&'a UOp, Option<&'a UOp>), PtxError> {
        match node.kind() {
            UOpKind::Load => Ok((node, None)),
            UOpKind::Cast => {
                let [load] = node.sources() else {
                    return Err(PtxError::Unsupported("public Select Cast needs one input".into()));
                };
                if !matches!(load.kind(), UOpKind::Load) {
                    return Err(PtxError::Unsupported("public Select Cast must consume a direct load".into()));
                }
                Ok((load, Some(node)))
            }
            _ => Err(PtxError::Unsupported("public Select payloads need only direct loads and source casts".into())),
        }
    }
    fn index<'a>(load: &'a UOp, output: &Shape, role: &str) -> Result<&'a UOp, PtxError> {
        let [index] = load.sources() else {
            return Err(PtxError::Unsupported(format!("public Select {role} load needs one index")));
        };
        let UArg::BufferIndex { elements, input_shape, output_shape, .. } = index.arg() else {
            return Err(PtxError::Unsupported(format!("public Select {role} does not admit affine views")));
        };
        let dtype = load.ty().ok_or_else(|| PtxError::Unsupported(format!("untyped Select {role}")))?.scalar;
        if index.ty().map(|ty| ty.scalar) != Some(dtype)
            || output_shape != output
            || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
            || elements.checked_mul(dtype.itemsize()).is_none()
        {
            return Err(PtxError::Unsupported(format!("public Select {role} descriptor is invalid")));
        }
        Ok(index)
    }

    let (condition_shape, condition_dtype) = if matches!(condition.kind(), UOpKind::Load) {
        let condition = direct_load(condition, "condition")?;
        let dtype = condition.ty().ok_or_else(|| PtxError::Unsupported("untyped Select condition".into()))?.scalar;
        if dtype != DType::Bool {
            return Err(PtxError::Unsupported("public Select condition must be Bool".into()));
        }
        let index = index(condition, output_shape, "condition")?;
        let shape = match index.arg() { UArg::BufferIndex { input_shape, .. } => input_shape.clone(), _ => unreachable!() };
        (shape, dtype)
    } else {
        let Some(shape) = scoped_select_predicate_shape(condition, output_shape, sm)? else {
            return Err(PtxError::Unsupported("public Select condition is not an admitted predicate root".into()));
        };
        (shape, DType::Bool)
    };
    let (true_load, true_cast) = payload(on_true)?;
    let (false_load, false_cast) = payload(on_false)?;
    let true_dtype = true_load.ty().ok_or_else(|| PtxError::Unsupported("untyped Select true payload".into()))?.scalar;
    let false_dtype = false_load.ty().ok_or_else(|| PtxError::Unsupported("untyped Select false payload".into()))?.scalar;
    let payload_dtype = if matches!((true_dtype, false_dtype), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
        DType::F32
    } else {
        true_dtype.promote(false_dtype)
    };
    if output_dtype != payload_dtype
        || on_true.ty().map(|ty| ty.scalar) != Some(payload_dtype)
        || on_false.ty().map(|ty| ty.scalar) != Some(payload_dtype)
    {
        return Err(PtxError::Unsupported("public Select payloads do not use source promotion".into()));
    }
    for (load, cast, source_dtype) in [(true_load, true_cast, true_dtype), (false_load, false_cast, false_dtype)] {
        if (source_dtype == payload_dtype) != cast.is_none()
            || cast.is_some_and(|node| node.ty().map(|ty| ty.scalar) != Some(payload_dtype))
        {
            return Err(PtxError::Unsupported("public Select must use exactly the source LUB casts".into()));
        }
    }
    let true_index = index(true_load, output_shape, "true payload")?;
    let false_index = index(false_load, output_shape, "false payload")?;
    fn input_shape(index: &UOp) -> &Shape {
        match index.arg() {
            UArg::BufferIndex { input_shape, .. } => input_shape,
            _ => unreachable!(),
        }
    }
    let value_shape = input_shape(true_index)
        .broadcast_with(input_shape(false_index))
        .map_err(|_| PtxError::Unsupported("public Select payload broadcast is invalid".into()))?;
    if condition_shape
        .broadcast_with(&value_shape)
        .map_err(|_| PtxError::Unsupported("public Select condition broadcast is invalid".into()))?
        != output_shape.clone()
    {
        return Err(PtxError::Unsupported("public Select does not prove a three-way broadcast".into()));
    }
    if [condition_dtype, true_dtype, false_dtype, payload_dtype].contains(&DType::F16) && sm < 53 {
        return Err(PtxError::Unsupported("F16 public Select conversion requires sm_53 or newer".into()));
    }
    reject_sign_storage_dtype(condition_dtype)?;
    reject_sign_storage_dtype(true_dtype)?;
    reject_sign_storage_dtype(false_dtype)?;
    reject_sign_storage_dtype(payload_dtype)?;
    Ok(Some(ScopedStorageMode::Select))
}

/// True division is not raw `DIV`: its public graph first performs the source
/// LUB, lifts an integral/Bool dividend and divisor to F32, rounds the direct
/// Reciprocal result at that dtype, then performs one ordered Mul.  This
/// validator recognizes only that whole chain, so an arbitrary reciprocal-Mul
/// compound cannot inherit the scoped storage ABI.
fn scoped_div_plan(store: &UOp, sm: u32) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    let UOpKind::GraphBinary(crate::BinaryOp::Mul) = value.kind() else {
        return Ok(None);
    };
    let [dividend, reciprocal] = value.sources() else {
        return Err(PtxError::Unsupported("public Div Mul needs two inputs".into()));
    };
    let UOpKind::GraphUnary(crate::UnaryOp::Reciprocal) = reciprocal.kind() else {
        return Ok(None);
    };
    let [divisor] = reciprocal.sources() else {
        return Err(PtxError::Unsupported("public Div Reciprocal needs one input".into()));
    };
    let Some(output_index) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let UArg::BufferIndex { elements: output_elements, output_shape, .. } = output_index.arg() else {
        return Err(PtxError::Unsupported("public Div requires a concrete output buffer".into()));
    };
    let output_dtype = value.ty().ok_or_else(|| PtxError::Unsupported("untyped Div output".into()))?.scalar;
    if !output_dtype.is_float()
        || output_index.ty().map(|ty| ty.scalar) != Some(output_dtype)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(output_dtype.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("public Div output descriptor is invalid".into()));
    }

    fn source_promote(left: DType, right: DType) -> DType {
        if matches!((left, right), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
            DType::F32
        } else {
            left.promote(right)
        }
    }
    fn path<'a>(mut node: &'a UOp, original: DType, targets: &[DType]) -> Result<&'a UOp, PtxError> {
        for target in targets.iter().rev() {
            let UOpKind::Cast = node.kind() else {
                return Err(PtxError::Unsupported("public Div is missing a required source cast".into()));
            };
            if node.ty().map(|ty| ty.scalar) != Some(*target) {
                return Err(PtxError::Unsupported("public Div cast target is not source-exact".into()));
            }
            let [input] = node.sources() else {
                return Err(PtxError::Unsupported("public Div Cast needs one input".into()));
            };
            node = input;
        }
        if !matches!(node.kind(), UOpKind::Load)
            || node.ty().map(|ty| ty.scalar) != Some(original)
        {
            return Err(PtxError::Unsupported("public Div needs direct typed input loads".into()));
        }
        Ok(node)
    }
    fn index<'a>(load: &'a UOp, output: &Shape) -> Result<&'a UOp, PtxError> {
        let [index] = load.sources() else {
            return Err(PtxError::Unsupported("public Div load needs one index".into()));
        };
        let UArg::BufferIndex { elements, input_shape, output_shape, .. } = index.arg() else {
            return Err(PtxError::Unsupported("public Div does not admit affine-view inputs".into()));
        };
        let dtype = load.ty().ok_or_else(|| PtxError::Unsupported("untyped Div input".into()))?.scalar;
        if index.ty().map(|ty| ty.scalar) != Some(dtype)
            || output_shape != output
            || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
            || elements.checked_mul(dtype.itemsize()).is_none()
        {
            return Err(PtxError::Unsupported("public Div input descriptor is invalid".into()));
        }
        Ok(index)
    }
    // `dividend` can itself be a Cast chain, so recover its original dtype by
    // walking to the direct load before calculating the public LUB.
    fn load_dtype(mut node: &UOp) -> Result<DType, PtxError> {
        while matches!(node.kind(), UOpKind::Cast) {
            let [input] = node.sources() else {
                return Err(PtxError::Unsupported("public Div Cast needs one input".into()));
            };
            node = input;
        }
        if !matches!(node.kind(), UOpKind::Load) {
            return Err(PtxError::Unsupported("public Div needs direct input loads".into()));
        }
        node.ty().map(|ty| ty.scalar).ok_or_else(|| PtxError::Unsupported("untyped Div input".into()))
    }
    let lhs_dtype = load_dtype(dividend)?;
    let rhs_dtype = load_dtype(divisor)?;
    let division_dtype = source_promote(lhs_dtype, rhs_dtype);
    let dividend_dtype = if division_dtype.is_float() { division_dtype } else { DType::F32 };
    let reciprocal_dtype = if division_dtype.is_float() { division_dtype } else { DType::F32 };
    let expected_output = source_promote(dividend_dtype, reciprocal_dtype);
    if output_dtype != expected_output
        || dividend.ty().map(|ty| ty.scalar) != Some(dividend_dtype)
        || reciprocal.ty().map(|ty| ty.scalar) != Some(reciprocal_dtype)
        || divisor.ty().map(|ty| ty.scalar) != Some(reciprocal_dtype)
    {
        return Err(PtxError::Unsupported("public Div dtype flow is not source-exact".into()));
    }
    let mut lhs_targets = Vec::new();
    if lhs_dtype != division_dtype { lhs_targets.push(division_dtype); }
    if division_dtype != dividend_dtype { lhs_targets.push(dividend_dtype); }
    let mut rhs_targets = Vec::new();
    if rhs_dtype != division_dtype { rhs_targets.push(division_dtype); }
    if division_dtype != reciprocal_dtype { rhs_targets.push(reciprocal_dtype); }
    let lhs_load = path(dividend, lhs_dtype, &lhs_targets)?;
    let rhs_load = path(divisor, rhs_dtype, &rhs_targets)?;
    let lhs_index = index(lhs_load, output_shape)?;
    let rhs_index = index(rhs_load, output_shape)?;
    let lhs_shape = match lhs_index.arg() { UArg::BufferIndex { input_shape, .. } => input_shape, _ => unreachable!() };
    let rhs_shape = match rhs_index.arg() { UArg::BufferIndex { input_shape, .. } => input_shape, _ => unreachable!() };
    if lhs_shape.broadcast_with(rhs_shape).map_err(|_| PtxError::Unsupported("public Div broadcast is invalid".into()))? != output_shape.clone() {
        return Err(PtxError::Unsupported("public Div inputs do not produce the output broadcast shape".into()));
    }
    for dtype in [division_dtype, dividend_dtype, reciprocal_dtype, output_dtype] {
        output_shape.numel().map_err(|_| PtxError::Overflow)?.checked_mul(dtype.itemsize()).ok_or(PtxError::Overflow)?;
    }
    if output_dtype == DType::F16 && sm < 53 {
        return Err(PtxError::Unsupported("F16 public Div conversion requires sm_53 or newer".into()));
    }
    reject_sign_storage_dtype(output_dtype)?;
    Ok(Some(ScopedStorageMode::Div))
}

/// Bool public subtraction is structurally `Add(lhs, LogicalNot(rhs))`. Keep
/// the ordered source form explicit: raw Bool Sub/XOR and a swapped Not are
/// not interchangeable roots.
fn scoped_bool_sub_plan(store: &UOp) -> Result<Option<ScopedStorageMode>, PtxError> {
    let Some(value) = store.sources().get(1) else {
        return Err(PtxError::Unsupported("Store without value".into()));
    };
    let UOpKind::GraphBinary(crate::BinaryOp::Add) = value.kind() else {
        return Ok(None);
    };
    let [lhs, not_rhs] = value.sources() else {
        return Err(PtxError::Unsupported("Bool Sub needs two inputs".into()));
    };
    let UOpKind::Load = lhs.kind() else {
        return Err(PtxError::Unsupported("Bool Sub lhs must be a direct load".into()));
    };
    let UOpKind::GraphLogical(crate::LogicalOp::Not) = not_rhs.kind() else {
        return Err(PtxError::Unsupported("Bool Sub rhs must be LogicalNot".into()));
    };
    let [rhs] = not_rhs.sources() else {
        return Err(PtxError::Unsupported("Bool Sub Not must have one input".into()));
    };
    if !matches!(rhs.kind(), UOpKind::Load)
        || lhs.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || rhs.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || not_rhs.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || value.ty().map(|ty| ty.scalar) != Some(DType::Bool)
    {
        return Err(PtxError::Unsupported(
            "Bool Sub must preserve Bool Load-Not-Add dtypes".into(),
        ));
    }
    let Some(output_index) = store.sources().first() else {
        return Err(PtxError::Unsupported("Store without index".into()));
    };
    let UArg::BufferIndex {
        elements: output_elements,
        output_shape,
        ..
    } = output_index.arg()
    else {
        return Err(PtxError::Unsupported("Bool Sub requires a concrete output".into()));
    };
    if output_index.ty().map(|ty| ty.scalar) != Some(DType::Bool)
        || output_shape.numel().map_err(|_| PtxError::Overflow)? != *output_elements
        || output_elements.checked_mul(DType::Bool.itemsize()).is_none()
    {
        return Err(PtxError::Unsupported("Bool Sub output descriptor is invalid".into()));
    }
    fn input_shape<'a>(load: &'a UOp, output: &Shape) -> Result<&'a Shape, PtxError> {
        let [index] = load.sources() else {
            return Err(PtxError::Unsupported("Bool Sub load needs one index".into()));
        };
        let UArg::BufferIndex {
            elements,
            input_shape,
            output_shape,
            ..
        } = index.arg()
        else {
            return Err(PtxError::Unsupported(
                "Bool Sub does not admit affine-view operands".into(),
            ));
        };
        if index.ty().map(|ty| ty.scalar) != Some(DType::Bool)
            || output_shape != output
            || input_shape.numel().map_err(|_| PtxError::Overflow)? != *elements
            || elements.checked_mul(DType::Bool.itemsize()).is_none()
        {
            return Err(PtxError::Unsupported("Bool Sub input descriptor is invalid".into()));
        }
        Ok(input_shape)
    }
    let lhs_shape = input_shape(lhs, output_shape)?;
    let rhs_shape = input_shape(rhs, output_shape)?;
    if lhs_shape
        .broadcast_with(rhs_shape)
        .map_err(|_| PtxError::Unsupported("Bool Sub broadcast is invalid".into()))?
        != output_shape.clone()
    {
        return Err(PtxError::Unsupported("Bool Sub output shape is invalid".into()));
    }
    Ok(Some(ScopedStorageMode::SubBool))
}

/// Narrows the scoped F32/F64 working value exactly once at the final storage
/// boundary. Direct Sign writes canonical low-width encodings itself.
fn narrow_storage_result(
    lines: &mut Vec<String>,
    value: String,
    dtype: DType,
    mode: Option<ScopedStorageMode>,
) -> String {
    if mode != Some(ScopedStorageMode::Abs)
        && mode != Some(ScopedStorageMode::Reciprocal)
        && mode != Some(ScopedStorageMode::ReciprocalCast)
        && mode != Some(ScopedStorageMode::Sqrt)
        && mode != Some(ScopedStorageMode::SqrtCast)
        && mode != Some(ScopedStorageMode::Rsqrt)
        && mode != Some(ScopedStorageMode::Mul)
        && mode != Some(ScopedStorageMode::Add)
        && mode != Some(ScopedStorageMode::Sub)
        && mode != Some(ScopedStorageMode::Div)
    {
        return value;
    }
    let reciprocal = matches!(
        mode,
        Some(ScopedStorageMode::Reciprocal | ScopedStorageMode::ReciprocalCast)
    );
    let sqrt = matches!(mode, Some(ScopedStorageMode::Sqrt | ScopedStorageMode::SqrtCast | ScopedStorageMode::Rsqrt));
    let scoped_binary = matches!(
        mode,
        Some(ScopedStorageMode::Mul | ScopedStorageMode::Add | ScopedStorageMode::Sub | ScopedStorageMode::Div)
    );
    match dtype {
        DType::F16 => {
            if reciprocal || sqrt || scoped_binary {
                lines.push(format!("  cvt.rn.f32.f64 %f31, {value};"));
                lines.push("  cvt.rn.f16.f32 %r91, %f31;".into());
                return "%r91".into();
            }
            lines.push(format!("  cvt.rn.f16.f32 %r91, {value};"));
            "%r91".into()
        }
        DType::BF16 => {
            if reciprocal || sqrt || scoped_binary {
                lines.push(format!("  cvt.rn.f32.f64 %f31, {value};"));
                lines.push("  mov.b32 %r91, %f31;".into());
            } else {
                // Same raw ties-to-even conversion and NaN preservation used
                // by the typed reduction store path. `value` is F32 here.
                lines.push(format!("  mov.b32 %r91, {value};"));
            }
            lines.push("  and.b32 %r92, %r91, 0x7f800000;".into());
            lines.push("  setp.eq.u32 %p6, %r92, 0x7f800000;".into());
            lines.push("  and.b32 %r92, %r91, 0x007fffff;".into());
            lines.push("  setp.ne.u32 %p7, %r92, 0;".into());
            lines.push("  and.pred %p6, %p6, %p7;".into());
            lines.push("  shr.u32 %r93, %r91, 16;".into());
            lines.push("  and.b32 %r94, %r93, 0x7f;".into());
            lines.push("  setp.eq.u32 %p7, %r94, 0;".into());
            lines.push("  or.b32 %r94, %r93, 1;".into());
            lines.push("  selp.b32 %r93, %r94, %r93, %p7;".into());
            lines.push("  shr.u32 %r92, %r91, 16;".into());
            lines.push("  and.b32 %r92, %r92, 1;".into());
            lines.push("  add.u32 %r92, %r92, 0x7fff;".into());
            lines.push("  add.u32 %r91, %r91, %r92;".into());
            lines.push("  shr.u32 %r91, %r91, 16;".into());
            lines.push("  selp.b32 %r91, %r93, %r91, %p6;".into());
            "%r91".into()
        }
        DType::F32 if reciprocal || sqrt || scoped_binary => {
            lines.push(format!("  cvt.rn.f32.f64 %f31, {value};"));
            "%f31".into()
        }
        _ => value,
    }
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
fn reject_reduction_storage_dtype(dtype: DType) -> Result<(), PtxError> {
    match dtype {
        DType::Bool
        | DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F32
        | DType::F64
        | DType::F16
        | DType::BF16 => Ok(()),
    }
}
fn ptx_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "u8",
        DType::I8 => "s8",
        DType::U8 => "u8",
        DType::I16 => "s16",
        DType::U16 => "u16",
        DType::I32 => "s32",
        DType::U32 => "u32",
        DType::I64 => "s64",
        DType::U64 => "u64",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F16 | DType::BF16 => "b16",
    }
}

/// Materialize the logical value of a source-LUB Cast before scoped Add/Mul reads
/// it. Narrow float targets deliberately encode then decode, matching the
/// fused tagged-value interpreter rather than retaining an unrounded F32.
fn emit_typed_binary_cast(
    lines: &mut Vec<String>,
    dst: &str,
    source: String,
    source_dtype: DType,
    target: DType,
) -> Result<(), PtxError> {
    fn source_f32(
        lines: &mut Vec<String>,
        source: String,
        source_dtype: DType,
    ) -> Result<String, PtxError> {
        match source_dtype {
            DType::F16 | DType::BF16 | DType::F32 => Ok(source),
            DType::F64 => {
                lines.push(format!("  cvt.rn.f32.f64 %f31, {source};"));
                Ok("%f31".into())
            }
            DType::Bool | DType::I8 | DType::U8 | DType::I16 | DType::U16 | DType::I32
            | DType::U32 | DType::I64 | DType::U64 => {
                lines.push(format!(
                    "  cvt.rn.f32.{} %f31, {source};",
                    ptx_type(source_dtype)
                ));
                Ok("%f31".into())
            }
        }
    }
    match target {
        DType::F16 => {
            let value = source_f32(lines, source, source_dtype)?;
            lines.push(format!("  cvt.rn.f16.f32 %r91, {value};"));
            lines.push(format!("  cvt.rn.f32.f16 {dst}, %r91;"));
        }
        DType::BF16 => {
            let value = source_f32(lines, source, source_dtype)?;
            // This is the same payload-aware ties-to-even conversion used by
            // the final BF16 store, followed by a raw decode for Mul.
            lines.push(format!("  mov.b32 %r91, {value};"));
            lines.push("  and.b32 %r92, %r91, 0x7f800000;".into());
            lines.push("  setp.eq.u32 %p6, %r92, 0x7f800000;".into());
            lines.push("  and.b32 %r92, %r91, 0x007fffff;".into());
            lines.push("  setp.ne.u32 %p7, %r92, 0;".into());
            lines.push("  and.pred %p6, %p6, %p7;".into());
            lines.push("  shr.u32 %r93, %r91, 16;".into());
            lines.push("  and.b32 %r94, %r93, 0x7f;".into());
            lines.push("  setp.eq.u32 %p7, %r94, 0;".into());
            lines.push("  or.b32 %r94, %r93, 1;".into());
            lines.push("  selp.b32 %r93, %r94, %r93, %p7;".into());
            lines.push("  shr.u32 %r92, %r91, 16;".into());
            lines.push("  and.b32 %r92, %r92, 1;".into());
            lines.push("  add.u32 %r92, %r92, 0x7fff;".into());
            lines.push("  add.u32 %r91, %r91, %r92;".into());
            lines.push("  shr.u32 %r91, %r91, 16;".into());
            lines.push("  selp.b32 %r91, %r93, %r91, %p6;".into());
            lines.push("  shl.b32 %r91, %r91, 16;".into());
            lines.push(format!("  mov.b32 {dst}, %r91;"));
        }
        DType::F32 => match source_dtype {
            DType::F16 | DType::BF16 | DType::F32 => lines.push(format!("  mov.f32 {dst}, {source};")),
            DType::F64 => lines.push(format!("  cvt.rn.f32.f64 {dst}, {source};")),
            _ => lines.push(format!("  cvt.rn.f32.{} {dst}, {source};", ptx_type(source_dtype))),
        },
        DType::F64 => match source_dtype {
            DType::F64 => lines.push(format!("  mov.f64 {dst}, {source};")),
            DType::F16 | DType::BF16 | DType::F32 => {
                lines.push(format!("  cvt.rn.f64.f32 {dst}, {source};"));
            }
            _ => lines.push(format!("  cvt.rn.f64.{} {dst}, {source};", ptx_type(source_dtype))),
        },
        _ => lines.push(format!(
            "  cvt.{}.{} {dst}, {source};",
            ptx_type(target),
            ptx_type(source_dtype)
        )),
    }
    Ok(())
}

/// Materialize tinygrad's `cast(bool)` truthiness without routing through a
/// numeric PTX conversion. In particular, NaN is nonzero/truthy, both zero
/// signs are false, and I64/U64 retain their full-width comparison.
fn emit_logical_not_bool_cast(
    lines: &mut Vec<String>,
    dst: &str,
    source: String,
    source_dtype: DType,
) {
    let predicate_dtype = match source_dtype {
        DType::F16 | DType::BF16 | DType::F32 => "f32",
        DType::F64 => "f64",
        dtype => ptx_type(dtype),
    };
    let zero = if source_dtype.is_float() { "0.0" } else { "0" };
    // Ordered equality is false for NaN; inverting it therefore gives the
    // required source truthiness for NaN as well as every nonzero value.
    lines.push(format!("  setp.eq.{predicate_dtype} %p1, {source}, {zero};"));
    lines.push("  not.pred %p1, %p1;".into());
    lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
}

/// Classify infinities from the exact input storage encoding. This is stricter
/// than a floating comparison: all exponent-all-ones NaN payloads are rejected
/// before the Bool result is formed, and integral source lanes never convert.
fn emit_isinf_predicate(lines: &mut Vec<String>, dst: &str, source: String, dtype: DType) {
    match dtype {
        DType::Bool | DType::I8 | DType::U8 | DType::I16 | DType::U16 | DType::I32 | DType::U32 | DType::I64 | DType::U64 => {
            lines.push(format!("  mov.u32 {dst}, 0;"));
            return;
        }
        DType::F16 => {
            lines.push(format!("  and.b32 %r60, {source}, 0x7fff;"));
            lines.push("  setp.eq.u32 %p1, %r60, 0x7c00;".into());
        }
        DType::BF16 => {
            lines.push(format!("  and.b32 %r60, {source}, 0x7fff;"));
            lines.push("  setp.eq.u32 %p1, %r60, 0x7f80;".into());
        }
        DType::F32 => {
            lines.push(format!("  mov.b32 %r60, {source};"));
            lines.push("  and.b32 %r60, %r60, 0x7fffffff;".into());
            lines.push("  setp.eq.u32 %p1, %r60, 0x7f800000;".into());
        }
        DType::F64 => {
            lines.push(format!("  mov.b64 %rd60, {source};"));
            lines.push("  and.b64 %rd60, %rd60, 0x7fffffffffffffff;".into());
            lines.push("  setp.eq.u64 %p1, %rd60, 0x7ff0000000000000;".into());
        }
    }
    lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
}

/// Select payload casts observe the same logical storage boundary as the
/// fused interpreter, but narrow destinations remain encoded bits so `selp`
/// can forward an unchosen NaN payload or signed zero without decode/reencode.
fn emit_typed_select_cast(
    lines: &mut Vec<String>,
    dst: &str,
    source: String,
    source_dtype: DType,
    target: DType,
) -> Result<(), PtxError> {
    fn source_f32(lines: &mut Vec<String>, source: String, dtype: DType) -> Result<String, PtxError> {
        match dtype {
            DType::F16 => {
                lines.push(format!("  cvt.rn.f32.f16 %f31, {source};"));
                Ok("%f31".into())
            }
            DType::BF16 => {
                lines.push(format!("  shl.b32 %r90, {source}, 16;"));
                lines.push("  mov.b32 %f31, %r90;".into());
                Ok("%f31".into())
            }
            DType::F32 => Ok(source),
            DType::F64 => {
                lines.push(format!("  cvt.rn.f32.f64 %f31, {source};"));
                Ok("%f31".into())
            }
            _ => {
                lines.push(format!("  cvt.rn.f32.{} %f31, {source};", ptx_type(dtype)));
                Ok("%f31".into())
            }
        }
    }
    match target {
        DType::F16 => {
            let value = source_f32(lines, source, source_dtype)?;
            lines.push(format!("  cvt.rn.f16.f32 {dst}, {value};"));
        }
        DType::BF16 => {
            let value = source_f32(lines, source, source_dtype)?;
            // Match the tagged fused Cast conversion, including ties-to-even
            // and NaN payload quieting, but retain the final low b16 bits.
            lines.push(format!("  mov.b32 %r91, {value};"));
            lines.push("  and.b32 %r92, %r91, 0x7f800000;".into());
            lines.push("  setp.eq.u32 %p6, %r92, 0x7f800000;".into());
            lines.push("  and.b32 %r92, %r91, 0x007fffff;".into());
            lines.push("  setp.ne.u32 %p7, %r92, 0;".into());
            lines.push("  and.pred %p6, %p6, %p7;".into());
            lines.push("  shr.u32 %r93, %r91, 16;".into());
            lines.push("  and.b32 %r94, %r93, 0x7f;".into());
            lines.push("  setp.eq.u32 %p7, %r94, 0;".into());
            lines.push("  or.b32 %r94, %r93, 1;".into());
            lines.push("  selp.b32 %r93, %r94, %r93, %p7;".into());
            lines.push("  shr.u32 %r92, %r91, 16;".into());
            lines.push("  and.b32 %r92, %r92, 1;".into());
            lines.push("  add.u32 %r92, %r92, 0x7fff;".into());
            lines.push("  add.u32 %r91, %r91, %r92;".into());
            lines.push("  shr.u32 %r91, %r91, 16;".into());
            lines.push("  selp.b32 %r91, %r93, %r91, %p6;".into());
            lines.push(format!("  mov.b32 {dst}, %r91;"));
        }
        DType::F32 => {
            let value = source_f32(lines, source, source_dtype)?;
            lines.push(format!("  mov.f32 {dst}, {value};"));
        }
        DType::F64 => match source_dtype {
            DType::F64 => lines.push(format!("  mov.f64 {dst}, {source};")),
            DType::F16 | DType::BF16 | DType::F32 => {
                let value = source_f32(lines, source, source_dtype)?;
                lines.push(format!("  cvt.rn.f64.f32 {dst}, {value};"));
            }
            _ => lines.push(format!("  cvt.rn.f64.{} {dst}, {source};", ptx_type(source_dtype))),
        },
        _ => lines.push(format!("  cvt.{}.{} {dst}, {source};", ptx_type(target), ptx_type(source_dtype))),
    }
    Ok(())
}

/// Implements the logical SQRT storage boundary inside public Rsqrt without
/// allocating an intermediate buffer. The returned register is decoded at
/// the Sqrt result dtype, so Reciprocal consumes exactly a materialized Sqrt.
fn emit_rsqrt_sqrt_boundary(
    lines: &mut Vec<String>,
    dst: &str,
    source: String,
    source_dtype: DType,
    result_dtype: DType,
) -> Result<(), PtxError> {
    let wide = if source_dtype == DType::F64 {
        source
    } else {
        lines.push(format!("  cvt.rn.f64.f32 %fd31, {source};"));
        "%fd31".into()
    };
    lines.push(format!("  sqrt.rn.f64 %fd30, {wide};"));
    match result_dtype {
        DType::F16 => {
            lines.push("  cvt.rn.f32.f64 %f31, %fd30;".into());
            lines.push("  cvt.rn.f16.f32 %r91, %f31;".into());
            lines.push(format!("  cvt.rn.f32.f16 {dst}, %r91;"));
        }
        DType::BF16 => {
            lines.push("  cvt.rn.f32.f64 %f31, %fd30;".into());
            lines.push("  mov.b32 %r91, %f31;".into());
            lines.push("  and.b32 %r92, %r91, 0x7f800000;".into());
            lines.push("  setp.eq.u32 %p6, %r92, 0x7f800000;".into());
            lines.push("  and.b32 %r92, %r91, 0x007fffff;".into());
            lines.push("  setp.ne.u32 %p7, %r92, 0;".into());
            lines.push("  and.pred %p6, %p6, %p7;".into());
            lines.push("  shr.u32 %r93, %r91, 16;".into());
            lines.push("  and.b32 %r94, %r93, 0x7f;".into());
            lines.push("  setp.eq.u32 %p7, %r94, 0;".into());
            lines.push("  or.b32 %r94, %r93, 1;".into());
            lines.push("  selp.b32 %r93, %r94, %r93, %p7;".into());
            lines.push("  shr.u32 %r92, %r91, 16;".into());
            lines.push("  and.b32 %r92, %r92, 1;".into());
            lines.push("  add.u32 %r92, %r92, 0x7fff;".into());
            lines.push("  add.u32 %r91, %r91, %r92;".into());
            lines.push("  shr.u32 %r91, %r91, 16;".into());
            lines.push("  selp.b32 %r91, %r93, %r91, %p6;".into());
            lines.push("  shl.b32 %r91, %r91, 16;".into());
            lines.push(format!("  mov.b32 {dst}, %r91;"));
        }
        DType::F32 => lines.push(format!("  cvt.rn.f32.f64 {dst}, %fd30;")),
        DType::F64 => lines.push(format!("  mov.f64 {dst}, %fd30;")),
        _ => return Err(PtxError::Unsupported("public Rsqrt Sqrt is not floating".into())),
    }
    Ok(())
}

/// Implements the logical storage boundary of the Reciprocal inside the
/// public Div root without allocating an intermediate buffer.  The returned
/// register is decoded at the reciprocal result dtype, so the following Mul
/// observes exactly the value a materialized Reciprocal would expose.
fn emit_div_reciprocal_boundary(
    lines: &mut Vec<String>,
    dst: &str,
    source: String,
    source_dtype: DType,
    result_dtype: DType,
) -> Result<(), PtxError> {
    let wide = if source_dtype == DType::F64 {
        source
    } else {
        lines.push(format!("  cvt.rn.f64.f32 %fd31, {source};"));
        "%fd31".into()
    };
    lines.push(format!("  div.rn.f64 %fd30, 1.0, {wide};"));
    match result_dtype {
        DType::F16 => {
            lines.push("  cvt.rn.f32.f64 %f31, %fd30;".into());
            lines.push("  cvt.rn.f16.f32 %r91, %f31;".into());
            lines.push(format!("  cvt.rn.f32.f16 {dst}, %r91;"));
        }
        DType::BF16 => {
            lines.push("  cvt.rn.f32.f64 %f31, %fd30;".into());
            lines.push("  mov.b32 %r91, %f31;".into());
            lines.push("  and.b32 %r92, %r91, 0x7f800000;".into());
            lines.push("  setp.eq.u32 %p6, %r92, 0x7f800000;".into());
            lines.push("  and.b32 %r92, %r91, 0x007fffff;".into());
            lines.push("  setp.ne.u32 %p7, %r92, 0;".into());
            lines.push("  and.pred %p6, %p6, %p7;".into());
            lines.push("  shr.u32 %r93, %r91, 16;".into());
            lines.push("  and.b32 %r94, %r93, 0x7f;".into());
            lines.push("  setp.eq.u32 %p7, %r94, 0;".into());
            lines.push("  or.b32 %r94, %r93, 1;".into());
            lines.push("  selp.b32 %r93, %r94, %r93, %p7;".into());
            lines.push("  shr.u32 %r92, %r91, 16;".into());
            lines.push("  and.b32 %r92, %r92, 1;".into());
            lines.push("  add.u32 %r92, %r92, 0x7fff;".into());
            lines.push("  add.u32 %r91, %r91, %r92;".into());
            lines.push("  shr.u32 %r91, %r91, 16;".into());
            lines.push("  selp.b32 %r91, %r93, %r91, %p6;".into());
            lines.push("  shl.b32 %r91, %r91, 16;".into());
            lines.push(format!("  mov.b32 {dst}, %r91;"));
        }
        DType::F32 => lines.push(format!("  cvt.rn.f32.f64 {dst}, %fd30;")),
        DType::F64 => lines.push(format!("  mov.f64 {dst}, %fd30;")),
        _ => return Err(PtxError::Unsupported("public Div reciprocal is not floating".into())),
    }
    Ok(())
}

/// Select-mode narrow loads deliberately retain raw payload bits for the
/// payload `selp`.  A predicate condition must instead decode its logical
/// float value before ordered/unordered comparison.
fn emit_select_predicate_value(
    lines: &mut Vec<String>,
    value: String,
    dtype: DType,
    slot: u8,
) -> String {
    match dtype {
        DType::F16 => {
            let dst = format!("%f{slot}");
            lines.push(format!("  cvt.rn.f32.f16 {dst}, {value};"));
            dst
        }
        DType::BF16 => {
            let dst = format!("%f{slot}");
            lines.push(format!("  shl.b32 %r90, {value}, 16;"));
            lines.push(format!("  mov.b32 {dst}, %r90;"));
            dst
        }
        _ => value,
    }
}
fn emit(
    n: &UOp,
    ids: &BTreeMap<u64, usize>,
    lines: &mut Vec<String>,
    map: &mut BTreeMap<usize, usize>,
    linear: &str,
    allow_reduction_narrow: bool,
    storage_mode: Option<ScopedStorageMode>,
) -> Result<String, PtxError> {
    let id = map.len();
    map.insert(id, lines.len() + 1);
    let ty = n
        .ty()
        .ok_or_else(|| PtxError::Unsupported(format!("untyped {:?}", n.kind())))?
        .scalar;
    if allow_reduction_narrow {
        reject_reduction_storage_dtype(ty)?;
    } else if storage_mode.is_some() {
        reject_sign_storage_dtype(ty)?;
    } else {
        reject_dtype(ty)?;
    }
    let mut child = |i| {
        emit(
            &n.sources()[i],
            ids,
            lines,
            map,
            linear,
            allow_reduction_narrow,
            storage_mode,
        )
    };
    let dst = match ty {
        _ if storage_mode == Some(ScopedStorageMode::Rsqrt)
            && matches!(n.kind(), UOpKind::GraphUnary(crate::UnaryOp::Reciprocal)) =>
        {
            format!("%fd{id}")
        }
        _ if matches!(
            storage_mode,
            Some(
                ScopedStorageMode::Reciprocal
                    | ScopedStorageMode::ReciprocalCast
                    | ScopedStorageMode::Sqrt
                    | ScopedStorageMode::SqrtCast
            )
        )
            && matches!(n.kind(), UOpKind::GraphUnary(crate::UnaryOp::Reciprocal | crate::UnaryOp::Sqrt)) =>
        {
            format!("%fd{id}")
        }
        _ if matches!(
            storage_mode,
            Some(ScopedStorageMode::Mul | ScopedStorageMode::Add | ScopedStorageMode::Sub | ScopedStorageMode::Div | ScopedStorageMode::Eq | ScopedStorageMode::Ne | ScopedStorageMode::OrderedLt)
        )
            && matches!(
                n.kind(),
                UOpKind::GraphBinary(
                    crate::BinaryOp::Mul | crate::BinaryOp::Add | crate::BinaryOp::Sub
                )
            )
            && ty.is_float() =>
        {
            format!("%fd{id}")
        }
        DType::F16 | DType::BF16
            if storage_mode == Some(ScopedStorageMode::Neg)
                && matches!(n.kind(), UOpKind::Load | UOpKind::GraphUnary(crate::UnaryOp::Neg)) =>
        {
            format!("%r{id}")
        }
        DType::F16 | DType::BF16
            if storage_mode == Some(ScopedStorageMode::Sign)
                && matches!(n.kind(), UOpKind::GraphUnary(crate::UnaryOp::Sign)) =>
        {
            format!("%r{id}")
        }
        DType::F16 | DType::BF16
            if storage_mode == Some(ScopedStorageMode::IsInf)
                && matches!(n.kind(), UOpKind::Load) =>
        {
            format!("%r{id}")
        }
        DType::F16 | DType::BF16
            if matches!(
                storage_mode,
                Some(
                    ScopedStorageMode::Select
                        | ScopedStorageMode::Relu
                        | ScopedStorageMode::LeakyRelu
                        | ScopedStorageMode::Extrema
                        | ScopedStorageMode::Clamp
                )
            ) =>
        {
            format!("%r{id}")
        }
        DType::I64 | DType::U64 if storage_mode.is_some() => format!("%rd{id}"),
        DType::F16 | DType::BF16 | DType::F32 => format!("%f{id}"),
        DType::F64 => format!("%fd{id}"),
        DType::Bool => format!("%r{id}"),
        _ => format!("%r{id}"),
    };
    match n.kind() {
        UOpKind::Const => match n.arg() {
            UArg::Int(v) => lines.push(format!("  mov.{} {dst}, {v};", ptx_type(ty))),
            UArg::Scalar { dtype, bits } if *dtype == ty => {
                // Bool is logically one bit but PTX tensors store it in an
                // addressable byte.  A scalar Bool literal must use that
                // physical width as well; `mov.b1` is not a valid register
                // operation and would lose the canonical UOp Const boundary.
                let width = (dtype.itemsize() * 8) as u8;
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
            let off = broadcast_offset(input_shape.dims(), output_shape.dims(), linear)?;
            lines.extend(off);
            if let Some(view) = view {
                lines.extend(view_offset(view)?);
            }
            // All affine maps address elements.  Convert only after the signed
            // map has been proven in-range by its immutable descriptor.
            lines.push(format!("  mul.lo.u64 %rd28, %rd28, {};", ty.itemsize()));
            lines.push(format!("  add.u64 %rd29, %rd{b}0, %rd28;"));
            match ty {
                DType::F16 => {
                    lines.push(format!("  ld.global.b16 %r{id}, [%rd29];"));
                    if storage_mode != Some(ScopedStorageMode::Neg)
                        && storage_mode != Some(ScopedStorageMode::IsInf)
                        && !matches!(
                            storage_mode,
                            Some(
                                ScopedStorageMode::Select
                                    | ScopedStorageMode::Relu
                                    | ScopedStorageMode::LeakyRelu
                                    | ScopedStorageMode::Extrema
                                    | ScopedStorageMode::Clamp
                            )
                        )
                    {
                        lines.push(format!("  cvt.rn.f32.f16 {dst}, %r{id};"));
                    }
                }
                DType::BF16 => {
                    lines.push(format!("  ld.global.b16 %r{id}, [%rd29];"));
                    if storage_mode != Some(ScopedStorageMode::Neg)
                        && storage_mode != Some(ScopedStorageMode::IsInf)
                        && !matches!(
                            storage_mode,
                            Some(
                                ScopedStorageMode::Select
                                    | ScopedStorageMode::Relu
                                    | ScopedStorageMode::LeakyRelu
                                    | ScopedStorageMode::Extrema
                                    | ScopedStorageMode::Clamp
                            )
                        )
                    {
                        lines.push(format!("  shl.b32 %r90, %r{id}, 16;"));
                        lines.push(format!("  mov.b32 {dst}, %r90;"));
                    }
                }
                _ => lines.push(format!("  ld.global.{} {dst}, [%rd29];", ptx_type(ty))),
            }
        }
        UOpKind::Cast if matches!(
            storage_mode,
            Some(ScopedStorageMode::Mul | ScopedStorageMode::Add | ScopedStorageMode::Sub | ScopedStorageMode::Div | ScopedStorageMode::Eq | ScopedStorageMode::Ne | ScopedStorageMode::LogicalNot | ScopedStorageMode::OrderedLt | ScopedStorageMode::InclusiveLt | ScopedStorageMode::Select | ScopedStorageMode::LeakyRelu | ScopedStorageMode::Extrema | ScopedStorageMode::Clamp | ScopedStorageMode::Sqrt | ScopedStorageMode::SqrtCast | ScopedStorageMode::Rsqrt)
        ) => {
            let a = child(0)?;
            let source = n.sources()[0]
                .ty()
                .ok_or_else(|| PtxError::Unsupported("untyped Mul Cast input".into()))?
                .scalar;
            if storage_mode == Some(ScopedStorageMode::LogicalNot) {
                if ty != DType::Bool {
                    return Err(PtxError::Unsupported("public logical-not cast must target Bool".into()));
                }
                emit_logical_not_bool_cast(lines, &dst, a, source);
            } else if storage_mode == Some(ScopedStorageMode::InclusiveLt)
                && source == DType::Bool
                && ty == DType::Bool
            {
                // Public Le/Ge contains a no-op Bool Cast between the ordered
                // predicate and its canonical `!= true` inversion. Preserve
                // that explicitly without admitting arbitrary Bool casts.
                lines.push(format!("  mov.u32 {dst}, {a};"));
            } else if matches!(storage_mode, Some(ScopedStorageMode::Select | ScopedStorageMode::LeakyRelu | ScopedStorageMode::Extrema | ScopedStorageMode::Clamp)) {
                emit_typed_select_cast(lines, &dst, a, source, ty)?;
            } else {
                emit_typed_binary_cast(lines, &dst, a, source, ty)?;
            }
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
            if *op == crate::UnaryOp::IsInf && storage_mode == Some(ScopedStorageMode::IsInf) {
                let source_dtype = n.sources()[0]
                    .ty()
                    .ok_or_else(|| PtxError::Unsupported("untyped public IsInf source".into()))?
                    .scalar;
                if ty != DType::Bool {
                    return Err(PtxError::Unsupported("public IsInf must produce Bool".into()));
                }
                emit_isinf_predicate(lines, &dst, a, source_dtype);
                return Ok(dst);
            }
            if *op == crate::UnaryOp::Reciprocal
                && matches!(
                    storage_mode,
                    Some(ScopedStorageMode::Reciprocal | ScopedStorageMode::ReciprocalCast)
                )
            {
                // CPU/generic evaluates every floating unary through F64 and
                // only then crosses the tensor storage boundary. Use precise
                // F64 division, never PTX's approximate reciprocal path.
                let source_dtype = n.sources()[0]
                    .ty()
                    .ok_or_else(|| PtxError::Unsupported("untyped Reciprocal input".into()))?
                    .scalar;
                let wide = if source_dtype == DType::F64 {
                    a
                } else {
                    lines.push(format!("  cvt.rn.f64.f32 %fd31, {a};"));
                    "%fd31".into()
                };
                lines.push(format!("  div.rn.f64 {dst}, 1.0, {wide};"));
                return Ok(dst);
            }
            if *op == crate::UnaryOp::Sqrt
                && matches!(storage_mode, Some(ScopedStorageMode::Sqrt | ScopedStorageMode::SqrtCast))
            {
                // The generic and materialized CPU evaluators widen the
                // typed lane to F64 before SQRT. PTX's rounded F64 SQRT is
                // the corresponding non-approximate operation, after which
                // narrow/F32 stores cross their single output boundary.
                let source_dtype = n.sources()[0]
                    .ty()
                    .ok_or_else(|| PtxError::Unsupported("untyped Sqrt input".into()))?
                    .scalar;
                let wide = if source_dtype == DType::F64 {
                    a
                } else {
                    lines.push(format!("  cvt.rn.f64.f32 %fd31, {a};"));
                    "%fd31".into()
                };
                lines.push(format!("  sqrt.rn.f64 {dst}, {wide};"));
                return Ok(dst);
            }
            if *op == crate::UnaryOp::Sqrt && storage_mode == Some(ScopedStorageMode::Rsqrt) {
                let source_dtype = n.sources()[0].ty().ok_or_else(|| PtxError::Unsupported("untyped Rsqrt Sqrt input".into()))?.scalar;
                emit_rsqrt_sqrt_boundary(lines, &dst, a, source_dtype, ty)?;
                return Ok(dst);
            }
            if *op == crate::UnaryOp::Reciprocal && storage_mode == Some(ScopedStorageMode::Rsqrt) {
                let source_dtype = n.sources()[0].ty().ok_or_else(|| PtxError::Unsupported("untyped Rsqrt Reciprocal input".into()))?.scalar;
                let wide = if source_dtype == DType::F64 { a } else {
                    lines.push(format!("  cvt.rn.f64.f32 %fd31, {a};"));
                    "%fd31".into()
                };
                lines.push(format!("  div.rn.f64 {dst}, 1.0, {wide};"));
                return Ok(dst);
            }
            if *op == crate::UnaryOp::Reciprocal
                && storage_mode == Some(ScopedStorageMode::Div)
            {
                let source_dtype = n.sources()[0]
                    .ty()
                    .ok_or_else(|| PtxError::Unsupported("untyped public Div reciprocal input".into()))?
                    .scalar;
                emit_div_reciprocal_boundary(lines, &dst, a, source_dtype, ty)?;
                return Ok(dst);
            }
            if *op == crate::UnaryOp::Neg && storage_mode == Some(ScopedStorageMode::Neg) {
                match ty {
                    DType::I8 | DType::I16 | DType::I32 => {
                        lines.push(format!("  neg.s32 {dst}, {a};"));
                    }
                    DType::U8 | DType::U16 | DType::U32 => {
                        lines.push(format!("  sub.u32 {dst}, 0, {a};"));
                    }
                    DType::I64 => lines.push(format!("  neg.s64 {dst}, {a};")),
                    DType::U64 => lines.push(format!("  sub.u64 {dst}, 0, {a};")),
                    DType::F16 | DType::BF16 => {
                        lines.push(format!("  xor.b32 {dst}, {a}, 0x8000;"));
                    }
                    DType::F32 => {
                        lines.push(format!("  xor.b32 {dst}, {a}, 0x80000000;"));
                    }
                    DType::F64 => {
                        lines.push(format!("  xor.b64 {dst}, {a}, 0x8000000000000000;"));
                    }
                    DType::Bool => unreachable!(),
                }
                return Ok(dst);
            }
            if *op == crate::UnaryOp::Sign {
                // Sign is source-equivalent to ordered comparisons and
                // selects, rather than PTX's host-dependent `sign` intrinsic.
                // In particular, both zero signs produce +0 and unordered
                // floating lanes retain the source's positive-one branch.
                if matches!(ty, DType::F16 | DType::BF16)
                    && storage_mode == Some(ScopedStorageMode::Abs)
                {
                    lines.push(format!("  setp.eq.f32 %p1, {a}, 0.0;"));
                    lines.push(format!("  selp.f32 {dst}, 0.0, 1.0, %p1;"));
                    lines.push(format!("  setp.lt.f32 %p2, {a}, 0.0;"));
                    lines.push(format!("  selp.f32 {dst}, -1.0, {dst}, %p2;"));
                    return Ok(dst);
                }
                match ty {
                    DType::Bool => lines.push(format!("  mov.u32 {dst}, {a};")),
                    DType::U8 | DType::U16 | DType::U32 => {
                        lines.push(format!("  setp.ne.{} %p1, {a}, 0;", ptx_type(ty)));
                        lines.push(format!("  selp.b32 {dst}, 1, 0, %p1;"));
                    }
                    DType::U64 => {
                        lines.push("  setp.ne.u64 %p1, ".to_owned() + &a + ", 0;");
                        lines.push(format!("  selp.b64 {dst}, 1, 0, %p1;"));
                    }
                    DType::I8 | DType::I16 | DType::I32 => {
                        lines.push(format!("  setp.lt.{} %p1, {a}, 0;", ptx_type(ty)));
                        lines.push(format!("  selp.b32 {dst}, -1, 1, %p1;"));
                        lines.push(format!("  setp.ne.{} %p2, {a}, 0;", ptx_type(ty)));
                        lines.push(format!("  selp.b32 {dst}, {dst}, 0, %p2;"));
                    }
                    DType::I64 => {
                        lines.push(format!("  setp.lt.s64 %p1, {a}, 0;"));
                        lines.push(format!("  selp.b64 {dst}, -1, 1, %p1;"));
                        lines.push(format!("  setp.ne.s64 %p2, {a}, 0;"));
                        lines.push(format!("  selp.b64 {dst}, {dst}, 0, %p2;"));
                    }
                    DType::F16 => {
                        lines.push(format!("  setp.eq.f32 %p1, {a}, 0.0;"));
                        lines.push(format!("  selp.b32 {dst}, 0x0000, 0x3c00, %p1;"));
                        lines.push(format!("  setp.lt.f32 %p2, {a}, 0.0;"));
                        lines.push(format!("  selp.b32 {dst}, 0xbc00, {dst}, %p2;"));
                    }
                    DType::BF16 => {
                        lines.push(format!("  setp.eq.f32 %p1, {a}, 0.0;"));
                        lines.push(format!("  selp.b32 {dst}, 0x0000, 0x3f80, %p1;"));
                        lines.push(format!("  setp.lt.f32 %p2, {a}, 0.0;"));
                        lines.push(format!("  selp.b32 {dst}, 0xbf80, {dst}, %p2;"));
                    }
                    DType::F32 => {
                        lines.push(format!("  setp.eq.f32 %p1, {a}, 0.0;"));
                        lines.push(format!("  selp.f32 {dst}, 0.0, 1.0, %p1;"));
                        lines.push(format!("  setp.lt.f32 %p2, {a}, 0.0;"));
                        lines.push(format!("  selp.f32 {dst}, -1.0, {dst}, %p2;"));
                    }
                    DType::F64 => {
                        lines.push(format!("  setp.eq.f64 %p1, {a}, 0.0;"));
                        lines.push(format!("  selp.f64 {dst}, 0.0, 1.0, %p1;"));
                        lines.push(format!("  setp.lt.f64 %p2, {a}, 0.0;"));
                        lines.push(format!("  selp.f64 {dst}, -1.0, {dst}, %p2;"));
                    }
                }
                return Ok(dst);
            }
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
            if storage_mode == Some(ScopedStorageMode::LeakyRelu) {
                if *op != crate::BinaryOp::Mul {
                    return Err(PtxError::Unsupported(
                        "public LeakyReLU requires only its slope * input branch".into(),
                    ));
                }
                match ty {
                    DType::Bool => lines.push(format!("  and.b32 {dst}, {a}, {b};")),
                    DType::I8 | DType::I16 | DType::I32 => {
                        lines.push(format!("  mul.lo.s32 {dst}, {a}, {b};"));
                    }
                    DType::U8 | DType::U16 | DType::U32 => {
                        lines.push(format!("  mul.lo.u32 {dst}, {a}, {b};"));
                    }
                    DType::I64 => lines.push(format!("  mul.lo.s64 {dst}, {a}, {b};")),
                    DType::U64 => lines.push(format!("  mul.lo.u64 {dst}, {a}, {b};")),
                    DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                        let a = emit_select_predicate_value(lines, a, ty, 30);
                        let b = emit_select_predicate_value(lines, b, ty, 31);
                        let a = if ty == DType::F64 {
                            a
                        } else {
                            lines.push(format!("  cvt.rn.f64.f32 %fd29, {a};"));
                            "%fd29".into()
                        };
                        let b = if ty == DType::F64 {
                            b
                        } else {
                            lines.push(format!("  cvt.rn.f64.f32 %fd30, {b};"));
                            "%fd30".into()
                        };
                        lines.push(format!("  mul.rn.f64 %fd28, {a}, {b};"));
                        match ty {
                            DType::F16 => {
                                lines.push("  cvt.rn.f32.f64 %f31, %fd28;".into());
                                lines.push(format!("  cvt.rn.f16.f32 {dst}, %f31;"));
                            }
                            DType::BF16 => {
                                lines.push("  cvt.rn.f32.f64 %f31, %fd28;".into());
                                lines.push("  mov.b32 %r91, %f31;".into());
                                lines.push("  and.b32 %r92, %r91, 0x7f800000;".into());
                                lines.push("  setp.eq.u32 %p6, %r92, 0x7f800000;".into());
                                lines.push("  and.b32 %r92, %r91, 0x007fffff;".into());
                                lines.push("  setp.ne.u32 %p7, %r92, 0;".into());
                                lines.push("  and.pred %p6, %p6, %p7;".into());
                                lines.push("  shr.u32 %r93, %r91, 16;".into());
                                lines.push("  and.b32 %r94, %r93, 0x7f;".into());
                                lines.push("  setp.eq.u32 %p7, %r94, 0;".into());
                                lines.push("  or.b32 %r94, %r93, 1;".into());
                                lines.push("  selp.b32 %r93, %r94, %r93, %p7;".into());
                                lines.push("  shr.u32 %r92, %r91, 16;".into());
                                lines.push("  and.b32 %r92, %r92, 1;".into());
                                lines.push("  add.u32 %r92, %r92, 0x7fff;".into());
                                lines.push("  add.u32 %r91, %r91, %r92;".into());
                                lines.push("  shr.u32 %r91, %r91, 16;".into());
                                lines.push("  selp.b32 %r91, %r93, %r91, %p6;".into());
                                lines.push(format!("  mov.b32 {dst}, %r91;"));
                            }
                            DType::F32 => {
                                lines.push(format!("  cvt.rn.f32.f64 {dst}, %fd28;"));
                            }
                            DType::F64 => lines.push(format!("  mov.f64 {dst}, %fd28;")),
                            _ => unreachable!(),
                        }
                    }
                }
                return Ok(dst);
            }
            if matches!(
                storage_mode,
                Some(
                    ScopedStorageMode::Mul
                        | ScopedStorageMode::Add
                        | ScopedStorageMode::Sub
                        | ScopedStorageMode::Div
                        | ScopedStorageMode::SubBool
                )
            ) && matches!(*op, crate::BinaryOp::Mul | crate::BinaryOp::Add | crate::BinaryOp::Sub)
            {
                let is_add = *op == crate::BinaryOp::Add;
                let is_sub = *op == crate::BinaryOp::Sub;
                match ty {
                    DType::Bool if is_add => lines.push(format!("  or.b32 {dst}, {a}, {b};")),
                    DType::Bool => return Err(PtxError::Unsupported("raw Bool binary is not scoped Sub".into())),
                    DType::I8 | DType::I16 | DType::I32 => {
                        if is_add { lines.push(format!("  add.s32 {dst}, {a}, {b};")); }
                        else if is_sub { lines.push(format!("  sub.s32 {dst}, {a}, {b};")); }
                        else { lines.push(format!("  mul.lo.s32 {dst}, {a}, {b};")); }
                    }
                    DType::U8 | DType::U16 | DType::U32 => {
                        if is_add { lines.push(format!("  add.u32 {dst}, {a}, {b};")); }
                        else if is_sub { lines.push(format!("  sub.u32 {dst}, {a}, {b};")); }
                        else { lines.push(format!("  mul.lo.u32 {dst}, {a}, {b};")); }
                    }
                    DType::I64 => lines.push(format!("  {}.s64 {dst}, {a}, {b};", if is_add { "add" } else if is_sub { "sub" } else { "mul.lo" })),
                    DType::U64 => lines.push(format!("  {}.u64 {dst}, {a}, {b};", if is_add { "add" } else if is_sub { "sub" } else { "mul.lo" })),
                    DType::F16 | DType::BF16 | DType::F32 => {
                        lines.push(format!("  cvt.rn.f64.f32 %fd29, {a};"));
                        lines.push(format!("  cvt.rn.f64.f32 %fd30, {b};"));
                        lines.push(format!("  {}.rn.f64 {dst}, %fd29, %fd30;", if is_add { "add" } else if is_sub { "sub" } else { "mul" }));
                    }
                    DType::F64 => {
                        lines.push(format!("  {}.rn.f64 {dst}, {a}, {b};", if is_add { "add" } else if is_sub { "sub" } else { "mul" }));
                    }
                }
                return Ok(dst);
            }
            if storage_mode == Some(ScopedStorageMode::Abs)
                && *op == crate::BinaryOp::Mul
            {
                let mnemonic = match ty {
                    DType::Bool => {
                        lines.push(format!("  and.b32 {dst}, {a}, {b};"));
                        return Ok(dst);
                    }
                    DType::I8 | DType::I16 | DType::I32 => "mul.lo.s32",
                    DType::U8 | DType::U16 | DType::U32 => "mul.lo.u32",
                    DType::I64 => "mul.lo.s64",
                    DType::U64 => "mul.lo.u64",
                    // F16/BF16 values and their Sign factor are decoded into
                    // F32 registers; the sole narrowing occurs at Store.
                    DType::F16 | DType::BF16 | DType::F32 => "mul.rn.f32",
                    DType::F64 => "mul.rn.f64",
                };
                lines.push(format!("  {mnemonic} {dst}, {a}, {b};"));
                return Ok(dst);
            }
            if storage_mode == Some(ScopedStorageMode::Extrema) {
                if !matches!(*op, crate::BinaryOp::Maximum | crate::BinaryOp::Minimum) {
                    return Err(PtxError::Unsupported(
                        "scoped extrema root does not match its public operation".into(),
                    ));
                }
                let raw_a = a;
                let raw_b = b;
                let a = emit_select_predicate_value(lines, raw_a.clone(), ty, 30);
                let b = emit_select_predicate_value(lines, raw_b.clone(), ty, 31);
                let predicate_dtype = match ty {
                    DType::F16 | DType::BF16 | DType::F32 => "f32",
                    DType::F64 => "f64",
                    dtype => ptx_type(dtype),
                };
                // Ordered predicates select rhs only when it strictly wins.
                // Equality, signed-zero ties, and every unordered NaN case
                // retain lhs and its exact stored payload.
                let predicate = if *op == crate::BinaryOp::Maximum { "lt" } else { "gt" };
                lines.push(format!("  setp.{predicate}.{predicate_dtype} %p1, {a}, {b};"));
                let select_type = match ty {
                    DType::F16 | DType::BF16 | DType::Bool | DType::I8 | DType::U8 | DType::I16 | DType::U16 | DType::I32 | DType::U32 => "b32",
                    DType::I64 | DType::U64 => "b64",
                    DType::F32 => "f32",
                    DType::F64 => "f64",
                };
                lines.push(format!("  selp.{select_type} {dst}, {raw_b}, {raw_a}, %p1;"));
                return Ok(dst);
            }
            if matches!(*op, crate::BinaryOp::Maximum | crate::BinaryOp::Minimum) {
                if matches!(ty, DType::F16 | DType::BF16) {
                    return Err(PtxError::Unsupported(
                        "ordered maximum/minimum for narrow float lacks an exact PTX path".into(),
                    ));
                }
                let predicate = if *op == crate::BinaryOp::Maximum { "lt" } else { "gt" };
                lines.push(format!("  setp.{predicate}.{} %p1, {a}, {b};", ptx_type(ty)));
                lines.push(format!("  selp.{} {dst}, {b}, {a}, %p1;", ptx_type(ty)));
                return Ok(dst);
            }
            let mnemonic = match op {
                crate::BinaryOp::Add => "add",
                crate::BinaryOp::Sub => "sub",
                crate::BinaryOp::Mul => "mul",
                crate::BinaryOp::Div if ty.is_float() => "div",
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
            if matches!(
                storage_mode,
                Some(
                    ScopedStorageMode::Select
                        | ScopedStorageMode::Relu
                        | ScopedStorageMode::LeakyRelu
                        | ScopedStorageMode::Clamp
                )
            ) {
                if storage_mode == Some(ScopedStorageMode::Relu)
                    && *op != crate::CompareOp::Lt
                {
                    return Err(PtxError::Unsupported(
                        "public ReLU requires ordered zero < input".into(),
                    ));
                }
                if storage_mode == Some(ScopedStorageMode::LeakyRelu)
                    && *op != crate::CompareOp::Lt
                {
                    return Err(PtxError::Unsupported(
                        "public LeakyReLU requires ordered input < zero".into(),
                    ));
                }
                let operand_dtype = n.sources()[0]
                    .ty()
                    .ok_or_else(|| PtxError::Unsupported("untyped public Select predicate operand".into()))?
                    .scalar;
                let a = emit_select_predicate_value(lines, a, operand_dtype, 30);
                let b = emit_select_predicate_value(lines, b, operand_dtype, 31);
                let predicate_dtype = match operand_dtype {
                    DType::F16 | DType::BF16 | DType::F32 => "f32",
                    DType::F64 => "f64",
                    dtype => ptx_type(dtype),
                };
                match op {
                    crate::CompareOp::Eq => {
                        lines.push(format!("  setp.eq.{predicate_dtype} %p1, {a}, {b};"));
                    }
                    crate::CompareOp::Ne => {
                        // Ordered equality plus inversion is source's
                        // unordered-not-equal: NaN selects the true payload.
                        lines.push(format!("  setp.eq.{predicate_dtype} %p1, {a}, {b};"));
                        lines.push("  not.pred %p1, %p1;".into());
                    }
                    crate::CompareOp::Lt => {
                        lines.push(format!("  setp.lt.{predicate_dtype} %p1, {a}, {b};"));
                    }
                    _ => return Err(PtxError::Unsupported("public Select predicate is not an admitted comparison".into())),
                }
                lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
                return Ok(dst);
            }
            if storage_mode == Some(ScopedStorageMode::InclusiveLt) {
                match op {
                    crate::CompareOp::Lt => {
                        let operand_dtype = n.sources()[0]
                            .ty()
                            .ok_or_else(|| PtxError::Unsupported("untyped public inclusive operand".into()))?
                            .scalar;
                        let predicate_dtype = match operand_dtype {
                            DType::F16 | DType::BF16 | DType::F32 => "f32",
                            DType::F64 => "f64",
                            dtype => ptx_type(dtype),
                        };
                        lines.push(format!("  setp.lt.{predicate_dtype} %p1, {a}, {b};"));
                        lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
                    }
                    crate::CompareOp::Ne => {
                        // The validator admits only the public Bool cast and
                        // a scalar UOp Const Bool(true), so this is literal
                        // `not(ordered_lt)`, including NaN -> true.
                        if n.sources()[0].ty().map(|ty| ty.scalar) != Some(DType::Bool)
                            || n.sources()[1].ty().map(|ty| ty.scalar) != Some(DType::Bool)
                        {
                            return Err(PtxError::Unsupported("inclusive comparison needs Bool inversion".into()));
                        }
                        lines.push(format!("  setp.eq.u8 %p1, {a}, {b};"));
                        lines.push("  not.pred %p1, %p1;".into());
                        lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
                    }
                    _ => return Err(PtxError::Unsupported("scoped inclusive predicate does not match its root plan".into())),
                }
                return Ok(dst);
            }
            if storage_mode == Some(ScopedStorageMode::LogicalNot) {
                if *op != crate::CompareOp::Ne
                    || n.sources()[0].ty().map(|ty| ty.scalar) != Some(DType::Bool)
                    || n.sources()[1].ty().map(|ty| ty.scalar) != Some(DType::Bool)
                {
                    return Err(PtxError::Unsupported("scoped logical-not does not match its Bool Ne root".into()));
                }
                // The root proof requires the RHS to be the exact scalar
                // UOp Const Bool(true). Keep the literal Ne rather than
                // folding it into the cast so provenance remains observable.
                lines.push(format!("  setp.eq.u8 %p1, {a}, {b};"));
                lines.push("  not.pred %p1, %p1;".into());
                lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
                return Ok(dst);
            }
            if matches!(storage_mode, Some(ScopedStorageMode::Eq | ScopedStorageMode::Ne | ScopedStorageMode::OrderedLt)) {
                let expected = if storage_mode == Some(ScopedStorageMode::Eq) {
                    crate::CompareOp::Eq
                } else if storage_mode == Some(ScopedStorageMode::Ne) {
                    crate::CompareOp::Ne
                } else {
                    crate::CompareOp::Lt
                };
                if *op != expected {
                    return Err(PtxError::Unsupported("scoped predicate does not match its root plan".into()));
                }
                let operand_dtype = n.sources()[0]
                    .ty()
                    .ok_or_else(|| PtxError::Unsupported("untyped public Eq operand".into()))?
                    .scalar;
                let predicate_dtype = match operand_dtype {
                    // F16/BF16 loads and exact source casts are decoded into
                    // F32 registers. Equality needs no output rounding, but
                    // it must compare those logical values, not their bits.
                    DType::F16 | DType::BF16 | DType::F32 => "f32",
                    DType::F64 => "f64",
                    dtype => ptx_type(dtype),
                };
                if expected == crate::CompareOp::Eq {
                    lines.push(format!("  setp.eq.{predicate_dtype} %p1, {a}, {b};"));
                    lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
                } else if expected == crate::CompareOp::Ne {
                    // `setp.eq` is ordered (false for NaN); complementing
                    // it is the exact unordered-not-equal truth table: NaN
                    // is true, while both zero signs compare equal/false.
                    lines.push(format!("  setp.eq.{predicate_dtype} %p1, {a}, {b};"));
                    lines.push("  not.pred %p1, %p1;".into());
                    lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
                } else {
                    // Ordered CMPLT is false for either NaN and retains the
                    // literal source operand order (Lt or reversed Gt).
                    lines.push(format!("  setp.lt.{predicate_dtype} %p1, {a}, {b};"));
                    lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
                }
                return Ok(dst);
            }
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
        UOpKind::GraphLogical(crate::LogicalOp::Not)
            if storage_mode == Some(ScopedStorageMode::NegBool) =>
        {
            let a = child(0)?;
            lines.push(format!("  setp.eq.u8 %p1, {a}, 0;"));
            lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
        }
        UOpKind::GraphLogical(crate::LogicalOp::Not)
            if storage_mode == Some(ScopedStorageMode::SubBool) =>
        {
            let a = child(0)?;
            lines.push(format!("  setp.eq.u8 %p1, {a}, 0;"));
            lines.push(format!("  selp.u32 {dst}, 1, 0, %p1;"));
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            let (p, a, b) = (child(0)?, child(1)?, child(2)?);
            lines.push(format!("  setp.ne.u32 %p2, {p}, 0;"));
            if matches!(
                storage_mode,
                Some(
                    ScopedStorageMode::Select
                        | ScopedStorageMode::Relu
                        | ScopedStorageMode::LeakyRelu
                        | ScopedStorageMode::Clamp
                )
            ) {
                let select_type = match ty {
                    DType::F16 | DType::BF16 | DType::Bool | DType::I8 | DType::U8 | DType::I16 | DType::U16 | DType::I32 | DType::U32 => "b32",
                    DType::I64 | DType::U64 => "b64",
                    DType::F32 => "f32",
                    DType::F64 => "f64",
                };
                lines.push(format!("  selp.{select_type} {dst}, {a}, {b}, %p2;"));
            } else {
                lines.push(format!("  selp.{} {dst}, {a}, {b}, %p2;", ptx_type(ty)));
            }
        }
        _ => return Err(PtxError::Unsupported(format!("{:?}", n.kind()))),
    };
    Ok(dst)
}

struct ReductionSpec<'a> {
    input_shape: &'a Shape,
    output_shape: &'a Shape,
    axes: &'a [usize],
    keepdim: bool,
    kind: crate::ReduceKind,
    mean: bool,
    value: &'a UOp,
}

#[derive(Clone, Copy)]
enum ReductionAccumulator {
    F32,
    F64,
    I32,
    U32,
    I64,
    U64,
}

fn reduction_accumulator(
    output: DType,
    value: DType,
    kind: crate::ReduceKind,
    mean: bool,
) -> Result<ReductionAccumulator, PtxError> {
    // CpuBackend stores every reduction result at `output` precision, but its
    // Scalar oracle accumulates F32/F16/BF16 through f64.  The tuple below is
    // therefore a typed ABI decision, not an inference from rendered PTX.
    if matches!(kind, crate::ReduceKind::Max | crate::ReduceKind::Min) {
        return match (output, value) {
            (DType::F16 | DType::BF16 | DType::F32, DType::F16 | DType::BF16 | DType::F32) => {
                Ok(ReductionAccumulator::F32)
            }
            (DType::F64, DType::F64) => Ok(ReductionAccumulator::F64),
            (
                DType::Bool | DType::I8 | DType::I16 | DType::I32,
                DType::Bool | DType::I8 | DType::I16 | DType::I32,
            ) => Ok(ReductionAccumulator::I32),
            (DType::U8 | DType::U16 | DType::U32, DType::U8 | DType::U16 | DType::U32) => {
                Ok(ReductionAccumulator::U32)
            }
            (DType::I64, DType::I64) => Ok(ReductionAccumulator::I64),
            (DType::U64, DType::U64) => Ok(ReductionAccumulator::U64),
            _ => Err(PtxError::Unsupported(format!(
                "reduction {kind:?} output {output:?} from {value:?} is outside the exact PTX subset",
            ))),
        };
    }
    match (mean, output, value) {
        // CPU reduces F32 through its f64 Scalar representation before the
        // final F32 quantization. Keep that accumulation width on PTX too.
        (false, DType::F32, DType::F32)
        | (_, DType::F16, DType::F16)
        | (_, DType::BF16, DType::BF16)
        | (
            true,
            DType::F32,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::I32
            | DType::U32
            | DType::I64
            | DType::U64,
        )
        | (true, DType::F32, DType::F32) => Ok(ReductionAccumulator::F32),
        (_, DType::F64, DType::F64) => Ok(ReductionAccumulator::F64),
        (
            false,
            DType::Bool | DType::I8 | DType::I16 | DType::I32,
            DType::Bool | DType::I8 | DType::I16 | DType::I32,
        ) => Ok(ReductionAccumulator::I32),
        (false, DType::U8 | DType::U16 | DType::U32, DType::U8 | DType::U16 | DType::U32) => {
            Ok(ReductionAccumulator::U32)
        }
        (false, DType::I64, DType::I64) => Ok(ReductionAccumulator::I64),
        (false, DType::U64, DType::U64) => Ok(ReductionAccumulator::U64),
        _ => Err(PtxError::Unsupported(format!(
            "reduction {:?} output from {value:?} is outside the exact PTX subset",
            if mean { "mean" } else { "sum" },
        ))),
    }
}

fn reduction_spec(store: &UOp) -> Result<Option<ReductionSpec<'_>>, PtxError> {
    let Some(finalize) = store
        .sources()
        .get(1)
        .filter(|node| matches!(node.kind(), UOpKind::ReduceFinalize))
    else {
        return Ok(None);
    };
    let update = finalize
        .sources()
        .first()
        .ok_or_else(|| PtxError::Unsupported("reduction finalize without update".into()))?;
    let init = update
        .sources()
        .first()
        .ok_or_else(|| PtxError::Unsupported("reduction update without init".into()))?;
    let UArg::Reduction {
        input_shape,
        output_shape,
        axes,
        keepdim,
        kind,
        mean,
    } = init.arg()
    else {
        return Err(PtxError::Unsupported("reduction metadata".into()));
    };
    let value = update
        .sources()
        .get(1)
        .ok_or_else(|| PtxError::Unsupported("reduction producer".into()))?;
    Ok(Some(ReductionSpec {
        input_shape,
        output_shape,
        axes,
        keepdim: *keepdim,
        kind: *kind,
        mean: *mean,
        value,
    }))
}

fn render_reduction(
    renderer: &PtxRenderer,
    store: &UOp,
    buffers: &[PtxBufferAbi],
    out_id: u64,
    extent: usize,
    reduction: ReductionSpec<'_>,
) -> Result<RenderedPtx, PtxError> {
    let out = buffers
        .iter()
        .find(|buffer| buffer.id == out_id)
        .ok_or_else(|| PtxError::Unsupported("reduction output missing ABI".into()))?;
    let value_dtype = reduction
        .value
        .ty()
        .ok_or_else(|| PtxError::Unsupported("untyped reduction producer".into()))?
        .scalar;
    let accumulator =
        reduction_accumulator(out.dtype, value_dtype, reduction.kind, reduction.mean)?;
    let extrema = matches!(
        reduction.kind,
        crate::ReduceKind::Max | crate::ReduceKind::Min
    );
    let product = matches!(reduction.kind, crate::ReduceKind::Product);
    if (matches!(out.dtype, DType::F16) || matches!(value_dtype, DType::F16)) && renderer.sm < 53 {
        return Err(PtxError::Unsupported(
            "F16 reduction conversion requires sm_53 or newer".into(),
        ));
    }
    if reduction.axes.windows(2).any(|axes| axes[0] >= axes[1])
        || reduction
            .axes
            .iter()
            .any(|axis| *axis >= reduction.input_shape.rank())
    {
        return Err(PtxError::Unsupported("invalid reduction axes".into()));
    }
    let expected_output = Shape::new(
        reduction
            .input_shape
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(axis, dimension)| {
                if reduction.axes.contains(&axis) {
                    reduction.keepdim.then_some(1)
                } else {
                    Some(*dimension)
                }
            })
            .collect::<Vec<_>>(),
    );
    if &expected_output != reduction.output_shape
        || extent
            != reduction
                .output_shape
                .numel()
                .map_err(|_| PtxError::Overflow)?
    {
        return Err(PtxError::Unsupported(
            "inconsistent static reduction geometry".into(),
        ));
    }
    let reduction_len = reduction.axes.iter().try_fold(1usize, |length, axis| {
        length
            .checked_mul(reduction.input_shape.dims()[*axis])
            .ok_or(PtxError::Overflow)
    })?;
    let reduction_len_u32 = u32::try_from(reduction_len).map_err(|_| PtxError::Overflow)?;
    if extrema && reduction_len == 0 {
        return Err(PtxError::Unsupported(
            "empty extrema has no PTX identity".into(),
        ));
    }
    let entry = format!("rg_reduce_e{extent}_r{reduction_len}_b{}", buffers.len());
    let mut lines = vec![
        format!("// {PTX_RENDERER_VERSION} ABI {PTX_ABI_VERSION}"),
        ".version 7.0".into(),
        format!(".target sm_{}", renderer.sm),
        ".address_size 64".into(),
        "".into(),
        format!(".visible .entry {entry}("),
    ];
    for index in 0..buffers.len() {
        lines.push(format!("  .param .u64 p{index},"));
    }
    lines.extend([
        "  .param .u64 extent".into(),
        ")".into(),
        "{".into(),
        "  .reg .pred %p<8>;".into(),
        "  .reg .b32 %r<96>;".into(),
        "  .reg .b64 %rd<96>;".into(),
        "  .reg .f32 %f<96>;".into(),
        "  .reg .f64 %fd<96>;".into(),
    ]);
    for index in 0..buffers.len() {
        lines.push(format!("  ld.param.u64 %rd{}0, [p{index}];", index + 1));
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
    for (index, buffer) in buffers.iter().enumerate() {
        ids.insert(buffer.id, index);
    }
    match accumulator {
        ReductionAccumulator::F32 | ReductionAccumulator::F64 => lines.push(
            if extrema && matches!(reduction.kind, crate::ReduceKind::Max) {
                "  mov.b64 %fd60, 0xfff0000000000000;"
            } else if extrema {
                "  mov.b64 %fd60, 0x7ff0000000000000;"
            } else if product {
                "  mov.f64 %fd60, 0d3ff0000000000000;"
            } else {
                "  mov.f64 %fd60, 0d0000000000000000;"
            }
            .into(),
        ),
        ReductionAccumulator::I32 | ReductionAccumulator::U32 => lines.push(
            if extrema {
                "  mov.u32 %r60, 0;"
            } else if product {
                "  mov.u32 %r60, 1;"
            } else {
                "  mov.u32 %r60, 0;"
            }
            .into(),
        ),
        ReductionAccumulator::I64 | ReductionAccumulator::U64 => lines.push(
            if extrema {
                "  mov.u64 %rd60, 0;"
            } else if product {
                "  mov.u64 %rd60, 1;"
            } else {
                "  mov.u64 %rd60, 0;"
            }
            .into(),
        ),
    }
    let mut map = BTreeMap::new();
    if reduction_len != 0 {
        lines.extend([
            "  mov.u32 %r5, 0;".into(),
            "REDUCE:".into(),
            format!("  setp.ge.u32 %p1, %r5, {reduction_len_u32};"),
            "  @%p1 bra REDUCE_DONE;".into(),
        ]);
        lines.extend(reduction_index_ptx(
            reduction.input_shape,
            reduction.output_shape,
            reduction.axes,
            reduction.keepdim,
        )?);
        let value = emit(reduction.value, &ids, &mut lines, &mut map, "%r4", true, None)?;
        if extrema {
            let convert = match value_dtype {
                DType::Bool | DType::U8 | DType::U16 | DType::U32 => "u32",
                DType::I8 | DType::I16 | DType::I32 => "s32",
                DType::I64 => "s64",
                DType::U64 => "u64",
                DType::F16 | DType::BF16 | DType::F32 => "f32",
                DType::F64 => "f64",
            };
            if value_dtype == DType::F64 {
                lines.push(format!("  mov.f64 %fd61, {value};"));
            } else {
                lines.push(format!("  cvt.rn.f64.{convert} %fd61, {value};"));
            }
            lines.push(format!(
                "  setp.{}.f64 %p2, %fd61, %fd60;",
                if matches!(reduction.kind, crate::ReduceKind::Max) {
                    "gt"
                } else {
                    "lt"
                }
            ));
            lines.push("  selp.f64 %fd60, %fd61, %fd60, %p2;".into());
            match accumulator {
                ReductionAccumulator::F32 | ReductionAccumulator::F64 => {}
                ReductionAccumulator::I32 | ReductionAccumulator::U32 => {
                    lines.push(format!("  selp.b32 %r60, {value}, %r60, %p2;"));
                }
                ReductionAccumulator::I64 | ReductionAccumulator::U64 => {
                    lines.push(format!("  selp.b64 %rd60, {value}, %rd60, %p2;"));
                }
            }
        } else {
            match accumulator {
                ReductionAccumulator::F32 => {
                    let convert = match value_dtype {
                        DType::Bool | DType::U8 | DType::U16 | DType::U32 => "u32",
                        DType::I8 | DType::I16 | DType::I32 => "s32",
                        DType::I64 => "s64",
                        DType::U64 => "u64",
                        DType::F16 | DType::BF16 | DType::F32 => "f32",
                        _ => unreachable!(),
                    };
                    lines.push(format!("  cvt.rn.f64.{convert} %fd61, {value};"));
                    lines.push(
                        if product {
                            "  mul.rn.f64 %fd60, %fd60, %fd61;"
                        } else {
                            "  add.rn.f64 %fd60, %fd60, %fd61;"
                        }
                        .into(),
                    );
                }
                ReductionAccumulator::F64 => {
                    lines.push(format!(
                        "  {}.rn.f64 %fd60, %fd60, {value};",
                        if product { "mul" } else { "add" }
                    ));
                }
                ReductionAccumulator::I32 => {
                    if product && out.dtype == DType::Bool {
                        lines.push(format!("  and.b32 %r60, %r60, {value};"));
                    } else {
                        lines.push(format!(
                            "  {}.s32 %r60, %r60, {value};",
                            if product { "mul.lo" } else { "add" }
                        ));
                    }
                }
                ReductionAccumulator::U32 => {
                    lines.push(format!(
                        "  {}.u32 %r60, %r60, {value};",
                        if product { "mul.lo" } else { "add" }
                    ));
                }
                ReductionAccumulator::I64 => {
                    lines.push(format!(
                        "  {}.s64 %rd60, %rd60, {value};",
                        if product { "mul.lo" } else { "add" }
                    ));
                }
                ReductionAccumulator::U64 => {
                    lines.push(format!(
                        "  {}.u64 %rd60, %rd60, {value};",
                        if product { "mul.lo" } else { "add" }
                    ));
                }
            }
        }
        lines.extend([
            "  add.u32 %r5, %r5, 1;".into(),
            "  bra REDUCE;".into(),
            "REDUCE_DONE:".into(),
        ]);
    }
    let result = if reduction.mean && reduction_len == 0 {
        match accumulator {
            ReductionAccumulator::F32 => {
                lines.push("  mov.b32 %f60, 0x7fc00000;".into());
                "%f60"
            }
            ReductionAccumulator::F64 => {
                lines.push("  mov.b64 %fd60, 0x7ff8000000000000;".into());
                "%fd60"
            }
            _ => return Err(PtxError::Unsupported("integer mean output".into())),
        }
    } else if reduction.mean {
        match accumulator {
            ReductionAccumulator::F32 => {
                lines.push(format!(
                    "  mov.b64 %fd61, 0x{:016x};",
                    (reduction_len as f64).to_bits()
                ));
                lines.push("  div.rn.f64 %fd60, %fd60, %fd61;".into());
                lines.push("  cvt.rn.f32.f64 %f60, %fd60;".into());
                "%f60"
            }
            ReductionAccumulator::F64 => {
                lines.push(format!(
                    "  mov.b64 %fd61, 0x{:016x};",
                    (reduction_len as f64).to_bits()
                ));
                lines.push("  div.rn.f64 %fd60, %fd60, %fd61;".into());
                "%fd60"
            }
            _ => return Err(PtxError::Unsupported("integer mean output".into())),
        }
    } else {
        match accumulator {
            ReductionAccumulator::F32 => {
                lines.push("  cvt.rn.f32.f64 %f60, %fd60;".into());
                "%f60"
            }
            ReductionAccumulator::F64 => "%fd60",
            ReductionAccumulator::I32 | ReductionAccumulator::U32 => "%r60",
            ReductionAccumulator::I64 | ReductionAccumulator::U64 => "%rd60",
        }
    };
    let result = match out.dtype {
        DType::F16 => {
            lines.push(format!("  cvt.rn.f16.f32 %r60, {result};"));
            "%r60"
        }
        DType::BF16 => {
            // Preserve representable NaN sign/payload bits and force a low
            // BF16 payload bit only when truncation would produce infinity.
            // Non-NaNs retain the ordinary wrapping ties-to-even path.
            lines.push(format!("  mov.b32 %r60, {result};"));
            lines.push("  and.b32 %r61, %r60, 0x7f800000;".into());
            lines.push("  setp.eq.u32 %p6, %r61, 0x7f800000;".into());
            lines.push("  and.b32 %r61, %r60, 0x007fffff;".into());
            lines.push("  setp.ne.u32 %p7, %r61, 0;".into());
            lines.push("  and.pred %p6, %p6, %p7;".into());
            lines.push("  shr.u32 %r62, %r60, 16;".into());
            lines.push("  and.b32 %r63, %r62, 0x7f;".into());
            lines.push("  setp.eq.u32 %p7, %r63, 0;".into());
            lines.push("  or.b32 %r63, %r62, 1;".into());
            lines.push("  selp.b32 %r62, %r63, %r62, %p7;".into());
            lines.push("  shr.u32 %r61, %r60, 16;".into());
            lines.push("  and.b32 %r61, %r61, 1;".into());
            lines.push("  add.u32 %r61, %r61, 0x7fff;".into());
            lines.push("  add.u32 %r61, %r60, %r61;".into());
            lines.push("  shr.u32 %r61, %r61, 16;".into());
            lines.push("  selp.b32 %r60, %r62, %r61, %p6;".into());
            "%r60"
        }
        _ => result,
    };
    let output_index = ids[&out_id] + 1;
    lines.push(format!(
        "  mul.wide.u32 %rd31, %r3, {};",
        out.dtype.itemsize()
    ));
    lines.push(format!("  add.u64 %rd31, %rd{output_index}0, %rd31;"));
    lines.push(format!(
        "  st.global.{} [%rd31], {result};",
        ptx_type(out.dtype)
    ));
    lines.extend(["DONE:".into(), "  ret;".into(), "}".into()]);
    let source = lines.join("\n") + "\n";
    Ok(RenderedPtx {
        source_map: map,
        cache_key: stable_key(&(PTX_RENDERER_VERSION, renderer.sm, &source, buffers)),
        source,
        buffers: buffers.to_vec(),
        extent,
        entry,
        launch: PtxLaunchGeometry::Linear,
        semantic_program: Some(KernelSemanticProgram::UOp(Arc::new(UOp::sink(vec![
            store.clone(),
        ])))),
    })
}

fn reduction_index_ptx(
    input: &Shape,
    output: &Shape,
    axes: &[usize],
    keepdim: bool,
) -> Result<Vec<String>, PtxError> {
    let mut lines = vec!["  mov.u32 %r4, 0;".into()];
    let mut output_axis = 0usize;
    let mut reduction_axis = 0usize;
    for (axis, dimension) in input.dims().iter().copied().enumerate() {
        let (linear, divisor) = if axes.contains(&axis) {
            let divisor = axes[reduction_axis + 1..]
                .iter()
                .try_fold(1usize, |value, next| {
                    value
                        .checked_mul(input.dims()[*next])
                        .ok_or(PtxError::Overflow)
                })?;
            reduction_axis += 1;
            ("%r5", divisor)
        } else {
            let output_axis_for_input = if keepdim { axis } else { output_axis };
            let divisor = output.dims()[output_axis_for_input + 1..]
                .iter()
                .try_fold(1usize, |value, next| {
                    value.checked_mul(*next).ok_or(PtxError::Overflow)
                })?;
            output_axis += 1;
            ("%r3", divisor)
        };
        lines.push(format!("  div.u32 %r61, {linear}, {divisor};"));
        lines.push(format!("  rem.u32 %r61, %r61, {dimension};"));
        lines.push(format!("  mul.lo.u32 %r4, %r4, {dimension};"));
        lines.push("  add.u32 %r4, %r4, %r61;".into());
    }
    Ok(lines)
}

fn broadcast_offset(
    input: &[usize],
    output: &[usize],
    linear: &str,
) -> Result<Vec<String>, PtxError> {
    if input.len() > output.len() {
        return Err(PtxError::Unsupported("broadcast rank".into()));
    };
    let mut lines = vec![format!("  mul.wide.u32 %rd28, {linear}, {};", 1)];
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
            lines.push(format!("  div.u32 %r20, {linear}, {divisor};"));
            lines.push(format!("  rem.u32 %r20, %r20, {d};"));
            lines.push(format!("  mul.wide.u32 %rd27, %r20, {scale};"));
            lines.push("  add.u64 %rd28, %rd28, %rd27;".into());
        }
    }
    lines.push(format!("  mul.lo.u64 %rd28, %rd28, {};", 1));
    Ok(lines)
}
fn view_offset(view: &crate::uop::AffineView) -> Result<Vec<String>, PtxError> {
    view.validate_read()
        .map_err(|_| PtxError::InvalidBinding("invalid signed affine view".into()))?;
    let mut lines = vec![
        "  mov.u64 %rd26, %rd28;".into(),
        format!("  mov.s64 %rd28, {};", view.offset),
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
        lines.push(format!("  mad.lo.s64 %rd28, %rd27, {stride}, %rd28;"));
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
        launch: PtxLaunchGeometry::Linear,
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
        rendered.validate()?;
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
        rendered.validate()?;
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
        for (index, (want, got)) in self.rendered.buffers.iter().zip(bindings).enumerate() {
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
            let pointer = got.buffer.device_ptr()?;
            self.rendered.validate_pointer_alignment(index, pointer)?;
            words.push(pointer)
        }
        words.push(self.rendered.extent as u64);
        let mut args: Vec<*mut c_void> = words.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        self.function.launch(
            self.rendered.launch_config(self.block_size)?,
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
        rendered.validate()?;
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
        for (index, (want, got)) in self.rendered.buffers.iter().zip(bindings).enumerate() {
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
            let pointer = got.buffer.device_ptr()?;
            self.rendered.validate_pointer_alignment(index, pointer)?;
            words.push(pointer);
        }
        words.push(self.rendered.extent as u64);
        let mut args: Vec<*mut c_void> = words.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        self.function.launch(
            self.rendered.launch_config(self.block_size)?,
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
        for (index, (want, got)) in self.rendered.buffers.iter().zip(bindings).enumerate() {
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
            let pointer = got.buffer.device_ptr()?;
            self.rendered.validate_pointer_alignment(index, pointer)?;
            words.push(pointer);
        }
        words.push(self.rendered.extent as u64);
        let config = self.rendered.launch_config(self.block_size)?;
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
        rendered.validate()?;
        let block_size = rendered.effective_block_size(block_size)?;
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
        rendered.validate()?;
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
        rendered.validate()?;
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
    use crate::{Backend, CpuBackend, Driver, Graph, Scalar, TensorData, UOp, UType};
    use std::{
        collections::HashMap,
        sync::{Arc, Barrier},
    };

    fn concurrent_rendered(key: &str) -> RenderedPtx {
        RenderedPtx {
            source: ".version 7.0".into(),
            source_map: BTreeMap::new(),
            buffers: vec![],
            extent: 0,
            cache_key: key.into(),
            entry: "kernel".into(),
            launch: PtxLaunchGeometry::Linear,
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
    fn extrema_ptx_has_a_scoped_lhs_preserving_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, predicate, select, store) in [
            (DType::Bool, "setp.lt.u8", "selp.b32", "st.global.u8"),
            (DType::I8, "setp.lt.s8", "selp.b32", "st.global.s8"),
            (DType::U8, "setp.lt.u8", "selp.b32", "st.global.u8"),
            (DType::I16, "setp.lt.s16", "selp.b32", "st.global.s16"),
            (DType::U16, "setp.lt.u16", "selp.b32", "st.global.u16"),
            (DType::I32, "setp.lt.s32", "selp.b32", "st.global.s32"),
            (DType::U32, "setp.lt.u32", "selp.b32", "st.global.u32"),
            (DType::I64, "setp.lt.s64", "selp.b64", "st.global.s64"),
            (DType::U64, "setp.lt.u64", "selp.b64", "st.global.u64"),
            (DType::F16, "setp.lt.f32", "selp.b32", "st.global.b16"),
            (DType::BF16, "setp.lt.f32", "selp.b32", "st.global.b16"),
            (DType::F32, "setp.lt.f32", "selp.f32", "st.global.f32"),
            (DType::F64, "setp.lt.f64", "selp.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 3], dtype);
            let maximum = graph.maximum(lhs, rhs).unwrap();
            let minimum = graph.minimum(lhs, rhs).unwrap();
            let max_first = renderer
                .render(&crate::lower_graph_elementwise(&graph, maximum).unwrap())
                .unwrap();
            let max_second = renderer
                .render(&crate::lower_graph_elementwise(&graph, maximum).unwrap())
                .unwrap();
            let minimum = renderer
                .render(&crate::lower_graph_elementwise(&graph, minimum).unwrap())
                .unwrap()
                .source;
            assert_eq!(graph.shape(maximum).unwrap(), &crate::Shape::from([2, 3]));
            assert_eq!(graph.dtype(maximum).unwrap(), dtype);
            assert!(max_first.source.contains(PTX_RENDERER_VERSION), "{dtype:?} version");
            assert!(max_first.source.contains(predicate), "{dtype:?} maximum predicate");
            assert!(minimum.contains(&predicate.replacen("lt", "gt", 1)), "{dtype:?} minimum predicate");
            assert!(max_first.source.contains(select), "{dtype:?} maximum raw lhs select");
            assert!(minimum.contains(select), "{dtype:?} minimum raw lhs select");
            assert!(max_first.source.contains(store), "{dtype:?} maximum typed store");
            assert!(minimum.contains(store), "{dtype:?} minimum typed store");
            assert!(!max_first.source.contains("max."), "{dtype:?} no native max");
            assert!(!minimum.contains("min."), "{dtype:?} no native min");
            assert!(matches!(&max_first.semantic_program, Some(KernelSemanticProgram::UOp(_))));
            assert_eq!(max_first.source, max_second.source, "{dtype:?} deterministic source");
            assert_eq!(max_first.cache_key, max_second.cache_key, "{dtype:?} deterministic key");
        }

        // Same-kind I64/U64 remains exact, while the mixed source-LUB bridge
        // intentionally converts both operands to F32 before ordered choice.
        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.maximum(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert_eq!(bridge.dtype(output).unwrap(), DType::F32);
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("setp.lt.f32"));

        // A source LUB Cast to F16 reaches the exact typed boundary before
        // ordered comparison and raw lhs/rhs payload selection.
        let mut narrow_cast = Graph::new();
        let lhs = narrow_cast.input_dtype("lhs", [1], DType::I16);
        let rhs = narrow_cast.input_dtype("rhs", [1], DType::F16);
        let output = narrow_cast.minimum(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap())
            .unwrap();
        assert_eq!(narrow_cast.dtype(output).unwrap(), DType::F16);
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(rendered.source.contains("setp.gt.f32"));
        assert!(rendered.source.contains("selp.b32"));

        // Scalar/empty descriptors and the existing 0.5/0.5 equal-tie VJP
        // contract remain graph-owned; the renderer only admits the forward
        // root and keeps CUDA's tagged UOp semantics as its parity authority.
        let mut scalar = Graph::new();
        let lhs = scalar.input_dtype("lhs", [], DType::F64);
        let rhs = scalar.input_dtype("rhs", [], DType::F64);
        let output = scalar.maximum(lhs, rhs).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&scalar, output).unwrap()).unwrap().extent, 1);
        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 2], DType::BF16);
        let rhs = empty.input_dtype("rhs", [1, 2], DType::BF16);
        let output = empty.minimum(lhs, rhs).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        let mut vjp = Graph::new();
        let lhs = vjp.input_dtype_requires_grad("lhs", [2, 1], DType::F32, true);
        let rhs = vjp.input_dtype_requires_grad("rhs", [1, 3], DType::F32, true);
        let output = vjp.maximum(lhs, rhs).unwrap();
        let loss = vjp.sum_all(output).unwrap();
        let lhs_gradient = vjp.grad(loss, lhs).unwrap();
        let rhs_gradient = vjp.grad(loss, rhs).unwrap();
        assert_eq!(vjp.dtype(lhs_gradient).unwrap(), DType::F32);
        assert_eq!(vjp.dtype(rhs_gradient).unwrap(), DType::F32);

        // Non-LUB casts, affine operands, unrelated compounds, and the F16
        // architecture gate cannot inherit this strict binary-root admission.
        let mut non_lub = Graph::new();
        let lhs = non_lub.input_dtype("lhs", [1], DType::I64);
        let rhs = non_lub.input_dtype("rhs", [1], DType::U64);
        let lhs = non_lub.cast(lhs, DType::F64).unwrap();
        let rhs = non_lub.cast(rhs, DType::F64).unwrap();
        let output = non_lub.maximum(lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&non_lub, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut viewed = Graph::new();
        let raw_lhs = viewed.input_dtype("lhs", [1, 2], DType::F16);
        let lhs = viewed.permute(raw_lhs, [1, 0]).unwrap();
        let rhs = viewed.input_dtype("rhs", [2, 1], DType::F16);
        let output = viewed.maximum(lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut compound = Graph::new();
        let lhs = compound.input_dtype("lhs", [1], DType::F16);
        let rhs = compound.input_dtype("rhs", [1], DType::F16);
        let zero = compound.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::F16));
        let lhs = compound.add(lhs, zero).unwrap();
        let output = compound.maximum(lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&compound, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut gate = Graph::new();
        let lhs = gate.input_dtype("lhs", [1], DType::F16);
        let rhs = gate.input_dtype("rhs", [1], DType::F16);
        let output = gate.maximum(lhs, rhs).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn clamp_ptx_has_strict_lower_upper_and_two_stage_roots() {
        let renderer = PtxRenderer::new(80).unwrap();
        for dtype in [
            DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16,
            DType::I32, DType::U32, DType::I64, DType::U64, DType::F16,
            DType::BF16, DType::F32, DType::F64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2, 1], dtype);
            let min = graph.input_dtype("min", [1, 3], dtype);
            let max = graph.input_dtype("max", [2, 3], dtype);
            let lower = graph.clamp(input, Some(min), None).unwrap();
            let upper = graph.clamp(input, None, Some(max)).unwrap();
            let both = graph.clamp(input, Some(min), Some(max)).unwrap();
            for output in [lower, upper, both] {
                let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                assert_eq!(graph.shape(output).unwrap(), &crate::Shape::from([2, 3]));
                assert_eq!(graph.dtype(output).unwrap(), dtype);
                assert!(first.source.contains(PTX_RENDERER_VERSION));
                assert!(first.source.contains("setp.lt"));
                assert!(first.source.contains("selp."));
                assert!(matches!(&first.semantic_program, Some(KernelSemanticProgram::UOp(_))));
                assert_eq!(first.source, second.source, "{dtype:?} deterministic source");
                assert_eq!(first.cache_key, second.cache_key, "{dtype:?} deterministic key");
            }
        }

        // Per-stage source LUB uses the intentional I64/U64-to-F32 bridge;
        // the lower result is a typed value before the upper comparison.
        let mut bridge = Graph::new();
        let input = bridge.input_dtype("input", [1], DType::I64);
        let min = bridge.input_dtype("min", [1], DType::U64);
        let max = bridge.input_dtype("max", [1], DType::F32);
        let output = bridge.clamp(input, Some(min), Some(max)).unwrap();
        let rendered = renderer.render(&crate::lower_graph_elementwise(&bridge, output).unwrap()).unwrap();
        assert_eq!(bridge.dtype(output).unwrap(), DType::F32);
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));

        // Empty/scalar domains and the graph-owned Select VJP are admitted
        // without changing their routing semantics.
        let mut scalar = Graph::new();
        let input = scalar.input_dtype_requires_grad("input", [], DType::F32, true);
        let min = scalar.input_dtype("min", [], DType::F32);
        let output = scalar.clamp(input, Some(min), None).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&scalar, output).unwrap()).unwrap().extent, 1);
        let loss = scalar.sum_all(output).unwrap();
        let gradient = scalar.grad(loss, input).unwrap();
        assert_eq!(scalar.dtype(gradient).unwrap(), DType::F32);
        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 1], DType::BF16);
        let max = empty.input_dtype("max", [1, 3], DType::BF16);
        let output = empty.clamp(input, None, Some(max)).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        let mut rejected = Graph::new();
        let input = rejected.input_dtype("input", [1], DType::F16);
        let min = rejected.input_dtype("min", [1], DType::F16);
        let output = rejected.clamp(input, Some(min), None).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&rejected, output).unwrap()), Err(PtxError::Unsupported(_))));
        let nodes = rejected.node_count();
        assert!(rejected.clamp(input, None, None).is_err());
        assert_eq!(rejected.node_count(), nodes);
    }

    #[test]
    fn captured_uniform_threefry_ptx_is_owner_scoped_and_matches_cpu_bytes() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let cache = ConcurrentPtxCache::new();
        for (dtype, shape, seed) in [
            (DType::F16, vec![5], 1337),
            (DType::BF16, vec![5], 1337),
            (DType::F32, vec![5], 1337),
            (DType::F64, vec![4], 0xffff_ffff),
        ] {
            let mut graph = Graph::new();
            let output = graph
                .uniform(shape.clone(), -1.25, 2.5, dtype, seed)
                .unwrap();
            let expected = CpuBackend
                .execute(&graph, output, &HashMap::new())
                .unwrap()
                .to_le_bytes()
                .unwrap();
            let lowered = crate::kernel::lower_graph_random(&graph, output).unwrap();
            let rendered = PtxRenderer::new(80).unwrap().render(&lowered).unwrap();
            assert_eq!(rendered.buffers.len(), 1);
            assert!(rendered.buffers[0].mutable);
            assert!(rendered.source.contains("captured-threefry"));
            assert!(rendered.source.contains("div.u64"));
            let kernel = cache.get_or_load(&primary, rendered.clone(), 64).unwrap();
            let output_lease = primary
                .allocate(NonZeroUsize::new(expected.len()).unwrap())
                .unwrap();
            kernel
                .launch(
                    &stream,
                    &[PtxBinding {
                        buffer: output_lease.view(),
                        dtype,
                        mutable: true,
                    }],
                    true,
                )
                .unwrap();
            let mut actual = vec![0; expected.len()];
            output_lease.view().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected, "{dtype:?} {shape:?}");
            assert!(Arc::ptr_eq(
                &kernel,
                &cache.get_or_load(&primary, rendered, 64).unwrap()
            ));
        }
        let mut empty = Graph::new();
        let output = empty.rand([0], DType::F32, 7).unwrap();
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&crate::kernel::lower_graph_random(&empty, output).unwrap())
            .unwrap();
        assert_eq!(rendered.extent, 0);
        let mut f16 = Graph::new();
        let f16_output = f16.rand([1], DType::F16, 1).unwrap();
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::kernel::lower_graph_random(&f16, f16_output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn captured_normal_and_randint_ptx_render_exact_plan_control_flow() {
        let mut graph = Graph::new();
        let normal = graph.randn([3], DType::F32, 17).unwrap();
        let randint = graph.randint([3], -7, 19, DType::I32, 23).unwrap();
        let renderer = PtxRenderer::new(80).unwrap();
        let normal = renderer
            .render(&crate::kernel::lower_graph_random(&graph, normal).unwrap())
            .unwrap();
        let randint = renderer
            .render(&crate::kernel::lower_graph_random(&graph, randint).unwrap())
            .unwrap();
        assert!(normal.source.contains("cos.approx.f32"));
        assert!(normal.source.contains("lg2.approx.f32"));
        assert!(normal.source.contains("sqrt.rn.f32"));
        assert!(normal.source.contains("mul.wide.u32 %rd12, %r3, 2;"));
        assert!(randint.source.contains("cvt.rzi.s32.f64"));
        assert!(randint.source.contains("st.global.s32"));
        assert_ne!(normal.cache_key, randint.cache_key);
    }

    #[test]
    fn signed_affine_view_ptx_keeps_negative_stride_in_address_identity() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::F32);
        let flipped = graph
            .stride(
                input,
                vec![
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    crate::Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                ],
            )
            .unwrap();
        let output = graph.neg(flipped).unwrap();
        let lowered = crate::lower_graph_elementwise(&graph, output).unwrap();
        let rendered = PtxRenderer::new(80).unwrap().render(&lowered).unwrap();
        assert!(rendered.source.contains("mad.lo.s64"));
        assert!(rendered.source.contains("mov.s64 %rd28"));
        let tensor = TensorData::from_scalars(
            [2, 3],
            DType::F32,
            [1., 2., 3., 4., 5., 6.].map(crate::Scalar::F),
        )
        .unwrap();
        assert_eq!(
            crate::kernel::execute_elementwise(
                &graph,
                output,
                &HashMap::from([("x".into(), tensor.clone())])
            )
            .unwrap()
            .to_le_bytes()
            .unwrap(),
            CpuBackend
                .execute(&graph, output, &HashMap::from([("x".into(), tensor)]))
                .unwrap()
                .to_le_bytes()
                .unwrap(),
        );
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
                view: view.into(),
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
                    view: view.clone().into(),
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
    fn static_reduction_ptx_has_serial_geometry_and_source_map() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3, 4], DType::F32);
        let producer = graph.neg(input).unwrap();
        let output = graph
            .reduce(producer, crate::ReduceKind::Sum, Some(vec![2, 0]), true)
            .unwrap();
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&crate::lower_graph_reduction(&graph, output).unwrap())
            .unwrap();
        assert_eq!(rendered.extent, 3);
        assert!(rendered.source.contains("REDUCE:"));
        assert!(rendered.source.contains("setp.ge.u32 %p1, %r5, 8;"));
        assert!(rendered.source.contains("neg.f32"));
        assert!(!rendered.source_map.is_empty());
        let repeat = PtxRenderer::new(80)
            .unwrap()
            .render(&crate::lower_graph_reduction(&graph, output).unwrap())
            .unwrap();
        assert_eq!(rendered.cache_key, repeat.cache_key);
        assert_eq!(rendered.buffers, repeat.buffers);
    }

    #[test]
    fn mock_static_reductions_match_cpu_for_fused_f32_and_f64() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let cache = ConcurrentPtxCache::new();
        for (name, dtype, shape, values, axes, keepdim, kind) in [
            (
                "f32 fused sum",
                DType::F32,
                vec![2, 3],
                vec![1.0, 2.0, 3.0, -4.0, 5.0, 6.0],
                vec![1],
                false,
                crate::ReduceKind::Sum,
            ),
            (
                "f64 fused keepdim mean",
                DType::F64,
                vec![2, 2, 2],
                vec![1.0, 3.0, 5.0, 7.0, -2.0, 4.0, 6.0, 8.0],
                vec![0, 2],
                true,
                crate::ReduceKind::Mean,
            ),
            (
                "f32 fused product",
                DType::F32,
                vec![2, 3],
                vec![1.0, 2.0, 3.0, -4.0, 5.0, 6.0],
                vec![1],
                false,
                crate::ReduceKind::Product,
            ),
            (
                "f32 max ignores nan and retains first signed-zero tie",
                DType::F32,
                vec![2, 3],
                vec![-0.0, 0.0, f64::NAN, -1.0, f64::NEG_INFINITY, -2.0],
                vec![1],
                false,
                crate::ReduceKind::Max,
            ),
            (
                "f32 min ignores nan and retains first signed-zero tie",
                DType::F32,
                vec![2, 3],
                vec![-0.0, 0.0, f64::NAN, 1.0, f64::INFINITY, 2.0],
                vec![1],
                false,
                crate::ReduceKind::Min,
            ),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", shape.clone(), dtype);
            let producer = graph.neg(input).unwrap();
            let output = graph.reduce(producer, kind, Some(axes), keepdim).unwrap();
            let tensor = TensorData::from_scalars(
                shape.clone(),
                dtype,
                values.iter().copied().map(crate::Scalar::F),
            )
            .unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("x".into(), tensor.clone())]),
                )
                .unwrap();
            let rendered = PtxRenderer::new(80)
                .unwrap()
                .render(&crate::lower_graph_reduction(&graph, output).unwrap())
                .unwrap();
            let input_lease = primary
                .allocate(NonZeroUsize::new(tensor.to_le_bytes().unwrap().len()).unwrap())
                .unwrap();
            let output_lease = primary
                .allocate(NonZeroUsize::new(expected.to_le_bytes().unwrap().len()).unwrap())
                .unwrap();
            input_lease
                .view()
                .copy_from(0, &tensor.to_le_bytes().unwrap())
                .unwrap();
            let kernel = cache.get_or_load(&primary, rendered.clone(), 32).unwrap();
            let bindings = rendered
                .buffers
                .iter()
                .map(|abi| PtxBinding {
                    buffer: if abi.mutable {
                        output_lease.view()
                    } else {
                        input_lease.view()
                    },
                    dtype: abi.dtype,
                    mutable: abi.mutable,
                })
                .collect::<Vec<_>>();
            kernel.launch(&stream, &bindings, true).unwrap();
            let mut actual = vec![0; expected.to_le_bytes().unwrap().len()];
            output_lease.view().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name}");
        }
        assert_eq!(mock.generic_kernel_count(), 5);
    }

    #[test]
    fn mock_static_reductions_match_cpu_for_wide_integer_and_bool_contracts() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let cache = ConcurrentPtxCache::new();
        let cases = [
            (
                "i32 wrapping sum",
                DType::I32,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::I(i32::MAX as i64),
                    crate::Scalar::I(1),
                    crate::Scalar::I(i32::MIN as i64),
                    crate::Scalar::I(-1),
                ],
                "add.s32",
            ),
            (
                "u32 wrapping sum",
                DType::U32,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::U(u32::MAX as u64),
                    crate::Scalar::U(1),
                    crate::Scalar::U(u32::MAX as u64),
                    crate::Scalar::U(2),
                ],
                "add.u32",
            ),
            (
                "i64 wrapping sum",
                DType::I64,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::I(i64::MAX),
                    crate::Scalar::I(1),
                    crate::Scalar::I(i64::MIN),
                    crate::Scalar::I(-1),
                ],
                "add.s64",
            ),
            (
                "u64 wrapping sum",
                DType::U64,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::U(u64::MAX),
                    crate::Scalar::U(1),
                    crate::Scalar::U(u64::MAX),
                    crate::Scalar::U(2),
                ],
                "add.u64",
            ),
            (
                "bool sum counts true",
                DType::Bool,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::Bool(true),
                    crate::Scalar::Bool(false),
                    crate::Scalar::Bool(true),
                    crate::Scalar::Bool(true),
                ],
                "add.s32",
            ),
            (
                "i8 wrapping product",
                DType::I8,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::I(64),
                    crate::Scalar::I(4),
                    crate::Scalar::I(-2),
                    crate::Scalar::I(3),
                ],
                "mul.lo.s32",
            ),
            (
                "i16 wrapping product",
                DType::I16,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::I(256),
                    crate::Scalar::I(256),
                    crate::Scalar::I(-2),
                    crate::Scalar::I(3),
                ],
                "mul.lo.s32",
            ),
            (
                "i32 wrapping product",
                DType::I32,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::I(i32::MAX as i64),
                    crate::Scalar::I(2),
                    crate::Scalar::I(-1),
                    crate::Scalar::I(3),
                ],
                "mul.lo.s32",
            ),
            (
                "i64 wrapping product",
                DType::I64,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::I(i64::MAX),
                    crate::Scalar::I(2),
                    crate::Scalar::I(-1),
                    crate::Scalar::I(3),
                ],
                "mul.lo.s64",
            ),
            (
                "u8 wrapping product",
                DType::U8,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::U(128),
                    crate::Scalar::U(2),
                    crate::Scalar::U(255),
                    crate::Scalar::U(3),
                ],
                "mul.lo.u32",
            ),
            (
                "u16 wrapping product",
                DType::U16,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::U(256),
                    crate::Scalar::U(256),
                    crate::Scalar::U(65535),
                    crate::Scalar::U(3),
                ],
                "mul.lo.u32",
            ),
            (
                "u32 wrapping product",
                DType::U32,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::U(u32::MAX as u64),
                    crate::Scalar::U(2),
                    crate::Scalar::U(3),
                    crate::Scalar::U(5),
                ],
                "mul.lo.u32",
            ),
            (
                "u64 wrapping product",
                DType::U64,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::U(u64::MAX),
                    crate::Scalar::U(2),
                    crate::Scalar::U(3),
                    crate::Scalar::U(5),
                ],
                "mul.lo.u64",
            ),
            (
                "bool product is and",
                DType::Bool,
                crate::ReduceKind::Product,
                vec![
                    crate::Scalar::Bool(true),
                    crate::Scalar::Bool(false),
                    crate::Scalar::Bool(true),
                    crate::Scalar::Bool(true),
                ],
                "and.b32",
            ),
            (
                "i64 minimum compares through f64 and keeps first tie",
                DType::I64,
                crate::ReduceKind::Min,
                vec![
                    crate::Scalar::I(i64::MIN),
                    crate::Scalar::I(i64::MIN + 1),
                    crate::Scalar::I(-2),
                    crate::Scalar::I(3),
                ],
                "cvt.rn.f64.s64",
            ),
            (
                "u64 maximum compares through f64 high-bit ties",
                DType::U64,
                crate::ReduceKind::Max,
                vec![
                    crate::Scalar::U(1_u64 << 63),
                    crate::Scalar::U((1_u64 << 63) + 1),
                    crate::Scalar::U(3),
                    crate::Scalar::U(2),
                ],
                "cvt.rn.f64.u64",
            ),
            (
                "bool minimum is and",
                DType::Bool,
                crate::ReduceKind::Min,
                vec![
                    crate::Scalar::Bool(true),
                    crate::Scalar::Bool(false),
                    crate::Scalar::Bool(true),
                    crate::Scalar::Bool(true),
                ],
                "setp.lt.f64",
            ),
            (
                "bool maximum is or",
                DType::Bool,
                crate::ReduceKind::Max,
                vec![
                    crate::Scalar::Bool(false),
                    crate::Scalar::Bool(true),
                    crate::Scalar::Bool(false),
                    crate::Scalar::Bool(false),
                ],
                "setp.gt.f64",
            ),
            (
                "i8 sum sign extends into i32",
                DType::I8,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::I(i8::MIN as i64),
                    crate::Scalar::I(-1),
                    crate::Scalar::I(i8::MAX as i64),
                    crate::Scalar::I(2),
                ],
                "ld.global.s8",
            ),
            (
                "i16 sum sign extends into i32",
                DType::I16,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::I(i16::MIN as i64),
                    crate::Scalar::I(-1),
                    crate::Scalar::I(i16::MAX as i64),
                    crate::Scalar::I(2),
                ],
                "ld.global.s16",
            ),
            (
                "u8 sum zero extends into u32",
                DType::U8,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::U(u8::MAX as u64),
                    crate::Scalar::U(1),
                    crate::Scalar::U(u8::MAX as u64),
                    crate::Scalar::U(2),
                ],
                "ld.global.u8",
            ),
            (
                "u16 sum zero extends into u32",
                DType::U16,
                crate::ReduceKind::Sum,
                vec![
                    crate::Scalar::U(u16::MAX as u64),
                    crate::Scalar::U(1),
                    crate::Scalar::U(u16::MAX as u64),
                    crate::Scalar::U(2),
                ],
                "ld.global.u16",
            ),
            (
                "u64 mean promotes through f64",
                DType::U64,
                crate::ReduceKind::Mean,
                vec![
                    crate::Scalar::U(u64::MAX),
                    crate::Scalar::U(1),
                    crate::Scalar::U(1_u64 << 63),
                    crate::Scalar::U(3),
                ],
                "cvt.rn.f64.u64",
            ),
            (
                "i16 mean promotes through f64",
                DType::I16,
                crate::ReduceKind::Mean,
                vec![
                    crate::Scalar::I(i16::MIN as i64),
                    crate::Scalar::I(1),
                    crate::Scalar::I(i16::MAX as i64),
                    crate::Scalar::I(-2),
                ],
                "cvt.rn.f64.s32",
            ),
        ];
        for (name, dtype, kind, values, instruction) in cases {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [2, 2], dtype);
            let output = graph.reduce(input, kind, Some(vec![1]), true).unwrap();
            let tensor = TensorData::from_scalars([2, 2], dtype, values).unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("x".into(), tensor.clone())]),
                )
                .unwrap();
            let rendered = PtxRenderer::new(80)
                .unwrap()
                .render(&crate::lower_graph_reduction(&graph, output).unwrap())
                .unwrap();
            assert!(rendered.source.contains(instruction), "{name}");
            let input_lease = primary
                .allocate(NonZeroUsize::new(tensor.to_le_bytes().unwrap().len()).unwrap())
                .unwrap();
            let output_lease = primary
                .allocate(NonZeroUsize::new(expected.to_le_bytes().unwrap().len()).unwrap())
                .unwrap();
            input_lease
                .view()
                .copy_from(0, &tensor.to_le_bytes().unwrap())
                .unwrap();
            let kernel = cache.get_or_load(&primary, rendered.clone(), 32).unwrap();
            let bindings = rendered
                .buffers
                .iter()
                .map(|abi| PtxBinding {
                    buffer: if abi.mutable {
                        output_lease.view()
                    } else {
                        input_lease.view()
                    },
                    dtype: abi.dtype,
                    mutable: abi.mutable,
                })
                .collect::<Vec<_>>();
            kernel.launch(&stream, &bindings, true).unwrap();
            let mut actual = vec![0; expected.to_le_bytes().unwrap().len()];
            output_lease.view().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name}");
        }
    }

    #[test]
    fn mock_static_reductions_preserve_raw_f16_and_bf16_storage_contracts() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let cache = ConcurrentPtxCache::new();
        for (name, dtype, words, load_marker, store_marker) in [
            (
                "f16 subnormal signed-zero infinity",
                DType::F16,
                vec![0x3c00_u16, 0x0001, 0x8000, 0x7c00],
                "cvt.rn.f32.f16",
                "cvt.rn.f16.f32",
            ),
            (
                "bf16 raw nan and signed zero",
                DType::BF16,
                vec![0x3f80_u16, 0x7fc1, 0x8000, 0x0001],
                "shl.b32",
                "selp.b32 %r60, %r62, %r61, %p6",
            ),
        ] {
            for kind in [
                crate::ReduceKind::Sum,
                crate::ReduceKind::Mean,
                crate::ReduceKind::Product,
                crate::ReduceKind::Min,
                crate::ReduceKind::Max,
            ] {
                let mut graph = Graph::new();
                let input = graph.input_dtype("x", [2, 2], dtype);
                let output = graph.reduce(input, kind, Some(vec![1]), true).unwrap();
                let bytes = words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>();
                let tensor = TensorData::from_le_bytes([2, 2], dtype, &bytes).unwrap();
                let expected = CpuBackend
                    .execute(
                        &graph,
                        output,
                        &HashMap::from([("x".into(), tensor.clone())]),
                    )
                    .unwrap();
                let rendered = PtxRenderer::new(80)
                    .unwrap()
                    .render(&crate::lower_graph_reduction(&graph, output).unwrap())
                    .unwrap();
                assert!(rendered.source.contains(load_marker), "{name} {kind:?}");
                assert!(rendered.source.contains(store_marker), "{name} {kind:?}");
                if dtype == DType::BF16 {
                    assert!(rendered.source.contains("and.b32 %r61, %r60, 0x7f800000"));
                    assert!(rendered.source.contains("or.b32 %r63, %r62, 1"));
                }
                let input_lease = primary
                    .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                    .unwrap();
                let output_lease = primary
                    .allocate(NonZeroUsize::new(expected.to_le_bytes().unwrap().len()).unwrap())
                    .unwrap();
                input_lease.view().copy_from(0, &bytes).unwrap();
                let kernel = cache.get_or_load(&primary, rendered.clone(), 32).unwrap();
                let bindings = rendered
                    .buffers
                    .iter()
                    .map(|abi| PtxBinding {
                        buffer: if abi.mutable {
                            output_lease.view()
                        } else {
                            input_lease.view()
                        },
                        dtype: abi.dtype,
                        mutable: abi.mutable,
                    })
                    .collect::<Vec<_>>();
                kernel.launch(&stream, &bindings, true).unwrap();
                let mut actual = vec![0; expected.to_le_bytes().unwrap().len()];
                output_lease.view().copy_to(0, &mut actual).unwrap();
                assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name} {kind:?}");
            }
        }
    }

    #[test]
    fn empty_static_reduction_has_defined_results_and_rejects_extrema_pre_driver() {
        use std::num::NonZeroUsize;
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = primary(&mock);
        let stream = primary.stream().unwrap();
        let cache = ConcurrentPtxCache::new();
        for (kind, expected) in [
            (crate::ReduceKind::Sum, vec![0_u8; 8]),
            (
                crate::ReduceKind::Mean,
                vec![0, 0, 192, 127, 0, 0, 192, 127],
            ),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [2, 0], DType::F32);
            let output = graph.reduce(input, kind, Some(vec![1]), false).unwrap();
            let rendered = PtxRenderer::new(80)
                .unwrap()
                .render(&crate::lower_graph_reduction(&graph, output).unwrap())
                .unwrap();
            assert_eq!(rendered.extent, 2);
            if matches!(kind, crate::ReduceKind::Mean) {
                assert!(!rendered.source.contains("div.rn"));
            }
            let input_lease = primary.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
            let output_lease = primary.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
            let kernel = cache.get_or_load(&primary, rendered.clone(), 32).unwrap();
            let bindings = rendered
                .buffers
                .iter()
                .map(|abi| PtxBinding {
                    buffer: if abi.mutable {
                        output_lease.view()
                    } else {
                        input_lease.view()
                    },
                    dtype: abi.dtype,
                    mutable: abi.mutable,
                })
                .collect::<Vec<_>>();
            kernel.launch(&stream, &bindings, true).unwrap();
            let mut actual = vec![0; 8];
            output_lease.view().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
        let mut zero_graph = Graph::new();
        let zero_input = zero_graph.input_dtype("x", [0, 2], DType::F32);
        let zero_output = zero_graph
            .reduce(zero_input, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let zero_rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&crate::lower_graph_reduction(&zero_graph, zero_output).unwrap())
            .unwrap();
        assert_eq!(zero_rendered.extent, 0);
        let zero_kernel = cache
            .get_or_load(&primary, zero_rendered.clone(), 32)
            .unwrap();
        let zero_buffer = primary.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        let zero_bindings = zero_rendered
            .buffers
            .iter()
            .map(|abi| PtxBinding {
                buffer: zero_buffer.view(),
                dtype: abi.dtype,
                mutable: abi.mutable,
            })
            .collect::<Vec<_>>();
        let before_zero_launch = mock.calls().len();
        zero_kernel.launch(&stream, &zero_bindings, true).unwrap();
        assert_eq!(mock.calls().len(), before_zero_launch);
        let before = mock.calls().len();
        let mut f16_graph = Graph::new();
        let f16_input = f16_graph.input_dtype("x", [2, 2], DType::F16);
        let f16_sum = f16_graph
            .reduce(f16_input, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::lower_graph_reduction(&f16_graph, f16_sum).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 0], DType::I8);
        assert!(matches!(
            graph.reduce(input, crate::ReduceKind::Min, Some(vec![1]), false),
            Err(crate::Error::EmptyReduction { .. })
        ));
        assert_eq!(
            mock.calls().len(),
            before,
            "renderer rejection is pre-driver"
        );
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
        ] {
            assert!(matches!(
                renderer.render(&unary_kernel(dtype, op, crate::Shape::new(vec![4]))),
                Err(PtxError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn public_reciprocal_has_a_scoped_f64_oracle_storage_path() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, load, store) in [
            (DType::F16, "cvt.rn.f32.f16", "cvt.rn.f16.f32"),
            (DType::BF16, "shl.b32", "selp.b32 %r91"),
            (DType::F32, "ld.global.f32", "st.global.f32"),
            (DType::F64, "ld.global.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let output = graph.reciprocal(input).unwrap();
            assert!(matches!(graph.op(output).unwrap(), crate::Op::Unary { op: crate::UnaryOp::Reciprocal, input: source }
                if *source == input));
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains(load), "{dtype:?} load");
            assert!(first.source.contains("div.rn.f64"), "{dtype:?} F64 division");
            assert!(first.source.contains(store), "{dtype:?} storage rounding");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

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
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let output = graph.reciprocal(input).unwrap();
            let crate::Op::Unary { input: cast, .. } = graph.op(output).unwrap() else {
                panic!("nonfloat Reciprocal must retain its raw terminal ALU");
            };
            assert!(matches!(graph.op(*cast).unwrap(), crate::Op::Cast { input: source, dtype: DType::F32 }
                if *source == input));
            let rendered = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(rendered.source.contains("cvt.f32"), "{dtype:?} public cast");
            assert!(rendered.source.contains("div.rn.f64"), "{dtype:?} F64 division");
            assert!(rendered.source.contains("st.global.f32"), "{dtype:?} F32 result");
        }

        // The root exception is deliberately exact: F16 keeps its established
        // ISA gate, and a compound reciprocal graph cannot inherit admission.
        let mut f16 = Graph::new();
        let input = f16.input_dtype("x", [1], DType::F16);
        let output = f16.reciprocal(input).unwrap();
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::lower_graph_elementwise(&f16, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
        let mut compound = Graph::new();
        let input = compound.input_dtype("x", [1], DType::F16);
        let reciprocal = compound.reciprocal(input).unwrap();
        let combined = compound.add(input, reciprocal).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&compound, combined).unwrap()),
            Err(PtxError::Unsupported(_))
        ));

        // Reciprocal's existing source VJP remains a separate composition;
        // this forward-only root exception neither changes nor accidentally
        // admits that compound graph.
        let mut vjp = Graph::new();
        let input = vjp.input_dtype("x", [], DType::F32);
        let output = vjp.reciprocal(input).unwrap();
        let gradient = vjp.grad(vjp.sum_all(output).unwrap(), input).unwrap();
        assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);
    }

    #[test]
    fn public_sqrt_has_a_scoped_f64_oracle_storage_path() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, load, store) in [
            (DType::F16, "cvt.rn.f32.f16", "cvt.rn.f16.f32"),
            (DType::BF16, "shl.b32", "selp.b32 %r91"),
            (DType::F32, "ld.global.f32", "st.global.f32"),
            (DType::F64, "ld.global.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let output = graph.sqrt(input).unwrap();
            assert!(matches!(graph.op(output).unwrap(), crate::Op::Unary { op: crate::UnaryOp::Sqrt, input: source }
                if *source == input));
            let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(first.source.contains(load), "{dtype:?} load");
            assert!(first.source.contains("sqrt.rn.f64"), "{dtype:?} exact F64 sqrt");
            assert!(!first.source.contains("sqrt.approx"), "{dtype:?} no approximate sqrt");
            assert!(first.source.contains(store), "{dtype:?} storage rounding");
            assert!(matches!(&first.semantic_program, Some(KernelSemanticProgram::UOp(_))));
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }
        for dtype in [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32, DType::U32, DType::I64, DType::U64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let output = graph.sqrt(input).unwrap();
            let crate::Op::Unary { input: cast, .. } = graph.op(output).unwrap() else {
                panic!("nonfloat Sqrt must retain its raw terminal ALU");
            };
            assert!(matches!(graph.op(*cast).unwrap(), crate::Op::Cast { input: source, dtype: DType::F32 } if *source == input));
            let rendered = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("cvt.rn.f32"), "{dtype:?} public cast");
            assert!(rendered.source.contains("sqrt.rn.f64"), "{dtype:?} exact F64 sqrt");
            assert!(rendered.source.contains("st.global.f32"), "{dtype:?} F32 result");
        }

        // The root exception retains the existing F16 target gate, scalar and
        // empty descriptors, and graph-owned result-typed-two VJP structure.
        let mut f16 = Graph::new();
        let input = f16.input_dtype("x", [1], DType::F16);
        let output = f16.sqrt(input).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&f16, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut floor = Graph::new();
        let input = floor.input_dtype("x", [1], DType::F64);
        let output = floor.sqrt(input).unwrap();
        let floor_source = PtxRenderer::new(20).unwrap().render(&crate::lower_graph_elementwise(&floor, output).unwrap()).unwrap().source;
        assert!(floor_source.contains(".target sm_20"));
        assert!(floor_source.contains("sqrt.rn.f64"));
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F64);
        let output = scalar.sqrt(input).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&scalar, output).unwrap()).unwrap().extent, 1);
        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0], DType::BF16);
        let output = empty.sqrt(input).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);
        let mut vjp = Graph::new();
        let input = vjp.input_dtype_requires_grad("x", [], DType::F32, true);
        let output = vjp.sqrt(input).unwrap();
        let gradient = vjp.grad(vjp.sum_all(output).unwrap(), input).unwrap();
        assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

        let mut compound = Graph::new();
        let input = compound.input_dtype("x", [1], DType::F32);
        let root = compound.sqrt(input).unwrap();
        let combined = compound.add(input, root).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&compound, combined).unwrap()), Err(PtxError::Unsupported(_))));
        let mut viewed = Graph::new();
        let input = viewed.input_dtype("x", [1, 1], DType::F32);
        let input = viewed.permute(input, [1, 0]).unwrap();
        let output = viewed.sqrt(input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_rsqrt_has_a_scoped_typed_sqrt_boundary() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, store) in [
            (DType::F16, "cvt.rn.f16.f32"), (DType::BF16, "selp.b32 %r91"),
            (DType::F32, "cvt.rn.f32.f64"), (DType::F64, "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let output = graph.rsqrt(input).unwrap();
            let crate::Op::Unary { op: crate::UnaryOp::Reciprocal, input: sqrt } = graph.op(output).unwrap() else { panic!("public rsqrt must end in Reciprocal") };
            assert!(matches!(graph.op(*sqrt).unwrap(), crate::Op::Unary { op: crate::UnaryOp::Sqrt, input: source } if *source == input));
            let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert_eq!(first.source.matches("sqrt.rn.f64").count(), 1, "{dtype:?} one typed sqrt boundary");
            assert!(first.source.contains("div.rn.f64"), "{dtype:?} reciprocal after sqrt");
            assert!(!first.source.contains("sqrt.approx"), "{dtype:?} no approximate sqrt");
            assert!(first.source.contains(store), "{dtype:?} typed final store");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} deterministic key");
            assert!(matches!(&first.semantic_program, Some(KernelSemanticProgram::UOp(_))));
        }
        for dtype in [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32, DType::U32, DType::I64, DType::U64] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let output = graph.rsqrt(input).unwrap();
            let crate::Op::Unary { input: sqrt, .. } = graph.op(output).unwrap() else { panic!("public rsqrt must end in Reciprocal") };
            let crate::Op::Unary { input: cast, .. } = graph.op(*sqrt).unwrap() else { panic!("public rsqrt must contain Sqrt") };
            assert!(matches!(graph.op(*cast).unwrap(), crate::Op::Cast { input: source, dtype: DType::F32 } if *source == input));
            let rendered = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(rendered.source.contains("sqrt.rn.f64"), "{dtype:?} sqrt");
            assert!(rendered.source.contains("div.rn.f64"), "{dtype:?} reciprocal");
            assert!(rendered.source.contains("st.global.f32"), "{dtype:?} F32 result");
        }
        let mut f16 = Graph::new();
        let input = f16.input_dtype("x", [1], DType::F16);
        let output = f16.rsqrt(input).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&f16, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0], DType::BF16);
        let output = empty.rsqrt(input).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);
        let mut vjp = Graph::new();
        let input = vjp.input_dtype_requires_grad("x", [], DType::F32, true);
        let output = vjp.rsqrt(input).unwrap();
        let gradient = vjp.grad(vjp.sum_all(output).unwrap(), input).unwrap();
        assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);
    }

    #[test]
    fn public_mul_has_a_scoped_typed_storage_path() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, multiply, store) in [
            (DType::Bool, "and.b32", "st.global.u8"),
            (DType::I8, "mul.lo.s32", "st.global.s8"),
            (DType::U8, "mul.lo.u32", "st.global.u8"),
            (DType::I16, "mul.lo.s32", "st.global.s16"),
            (DType::U16, "mul.lo.u32", "st.global.u16"),
            (DType::I32, "mul.lo.s32", "st.global.s32"),
            (DType::U32, "mul.lo.u32", "st.global.u32"),
            (DType::I64, "mul.lo.s64", "st.global.s64"),
            (DType::U64, "mul.lo.u64", "st.global.u64"),
            (DType::F16, "mul.rn.f64", "cvt.rn.f16.f32"),
            (DType::BF16, "mul.rn.f64", "selp.b32 %r91"),
            (DType::F32, "mul.rn.f64", "cvt.rn.f32.f64"),
            (DType::F64, "mul.rn.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 3], dtype);
            let output = graph.mul(lhs, rhs).unwrap();
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains(multiply), "{dtype:?} multiply");
            assert!(first.source.contains(store), "{dtype:?} store");
            assert!(first.source.contains(PTX_RENDERER_VERSION));
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

        // The source-specific I64/U64 meet inserts two F32 Casts before the
        // F64 working multiply. The public root carries no other arithmetic.
        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.mul(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("mul.rn.f64"));

        let mut narrow_cast = Graph::new();
        let lhs = narrow_cast.input_dtype("lhs", [1], DType::I16);
        let rhs = narrow_cast.input_dtype("rhs", [1], DType::F16);
        let output = narrow_cast.mul(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s16"));
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(rendered.source.contains("cvt.rn.f64.f32"));
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 1], DType::F32);
        let rhs = empty.input_dtype("rhs", [1, 3], DType::F32);
        let output = empty.mul(lhs, rhs).unwrap();
        assert_eq!(
            renderer
                .render(&crate::lower_graph_elementwise(&empty, output).unwrap())
                .unwrap()
                .extent,
            0
        );

        // The exception is root-scoped: a narrow compound and a direct raw
        // binary that lacks public source-LUB casts stay fail-closed.
        let mut compound = Graph::new();
        let lhs = compound.input_dtype("lhs", [1], DType::F16);
        let rhs = compound.input_dtype("rhs", [1], DType::F16);
        let product = compound.mul(lhs, rhs).unwrap();
        let combined = compound.add(product, lhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&compound, combined).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
        let mut raw = Graph::new();
        let lhs = raw.input_dtype("lhs", [1], DType::I64);
        let rhs = raw.input_dtype("rhs", [1], DType::U64);
        let raw_product = raw.binary(crate::BinaryOp::Mul, lhs, rhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&raw, raw_product).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn public_add_has_a_scoped_typed_storage_path() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, operation, store) in [
            (DType::Bool, "or.b32", "st.global.u8"),
            (DType::I8, "add.s32", "st.global.s8"),
            (DType::U8, "add.u32", "st.global.u8"),
            (DType::I16, "add.s32", "st.global.s16"),
            (DType::U16, "add.u32", "st.global.u16"),
            (DType::I32, "add.s32", "st.global.s32"),
            (DType::U32, "add.u32", "st.global.u32"),
            (DType::I64, "add.s64", "st.global.s64"),
            (DType::U64, "add.u64", "st.global.u64"),
            (DType::F16, "add.rn.f64", "cvt.rn.f16.f32"),
            (DType::BF16, "add.rn.f64", "selp.b32 %r91"),
            (DType::F32, "add.rn.f64", "cvt.rn.f32.f64"),
            (DType::F64, "add.rn.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 3], dtype);
            let output = graph.add(lhs, rhs).unwrap();
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains(operation), "{dtype:?} Add operation");
            assert!(first.source.contains(store), "{dtype:?} Add storage");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} Add key");
        }

        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.add(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("add.rn.f64"));

        let mut narrow_cast = Graph::new();
        let lhs = narrow_cast.input_dtype("lhs", [1], DType::U16);
        let rhs = narrow_cast.input_dtype("rhs", [1], DType::F16);
        let output = narrow_cast.add(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.u16"));
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 1], DType::F32);
        let rhs = empty.input_dtype("rhs", [1, 3], DType::F32);
        let output = empty.add(lhs, rhs).unwrap();
        assert_eq!(
            renderer
                .render(&crate::lower_graph_elementwise(&empty, output).unwrap())
                .unwrap()
                .extent,
            0
        );

        let mut compound = Graph::new();
        let lhs = compound.input_dtype("lhs", [1], DType::F16);
        let rhs = compound.input_dtype("rhs", [1], DType::F16);
        let sum = compound.add(lhs, rhs).unwrap();
        let combined = compound.mul(sum, lhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&compound, combined).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn public_sub_has_ordered_scoped_roots() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, operation, store) in [
            (DType::I8, "sub.s32", "st.global.s8"),
            (DType::U8, "sub.u32", "st.global.u8"),
            (DType::I16, "sub.s32", "st.global.s16"),
            (DType::U16, "sub.u32", "st.global.u16"),
            (DType::I32, "sub.s32", "st.global.s32"),
            (DType::U32, "sub.u32", "st.global.u32"),
            (DType::I64, "sub.s64", "st.global.s64"),
            (DType::U64, "sub.u64", "st.global.u64"),
            (DType::F16, "sub.rn.f64", "cvt.rn.f16.f32"),
            (DType::BF16, "sub.rn.f64", "selp.b32 %r91"),
            (DType::F32, "sub.rn.f64", "cvt.rn.f32.f64"),
            (DType::F64, "sub.rn.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 3], dtype);
            let output = graph.sub(lhs, rhs).unwrap();
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains(operation), "{dtype:?} Sub operation");
            assert!(first.source.contains(store), "{dtype:?} Sub storage");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} Sub key");
        }

        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.sub(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("sub.rn.f64"));

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 1], DType::F32);
        let rhs = empty.input_dtype("rhs", [1, 3], DType::F32);
        let output = empty.sub(lhs, rhs).unwrap();
        assert_eq!(
            renderer
                .render(&crate::lower_graph_elementwise(&empty, output).unwrap())
                .unwrap()
                .extent,
            0
        );

        let mut boolean = Graph::new();
        let lhs = boolean.input_dtype("lhs", [2, 1], DType::Bool);
        let rhs = boolean.input_dtype("rhs", [1, 3], DType::Bool);
        let output = boolean.sub(lhs, rhs).unwrap();
        assert!(matches!(boolean.op(output).unwrap(), crate::Op::Binary { op: crate::BinaryOp::Add, lhs: left, rhs: right }
            if *left == lhs && matches!(boolean.op(*right).unwrap(), crate::Op::Logical { op: crate::LogicalOp::Not, .. })));
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&boolean, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("setp.eq.u8"));
        assert!(rendered.source.contains("or.b32"));

        let mut raw_bool = Graph::new();
        let lhs = raw_bool.input_dtype("lhs", [1], DType::Bool);
        let rhs = raw_bool.input_dtype("rhs", [1], DType::Bool);
        let output = raw_bool.binary(crate::BinaryOp::Sub, lhs, rhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&raw_bool, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
        let mut swapped = Graph::new();
        let lhs = swapped.input_dtype("lhs", [1], DType::Bool);
        let rhs = swapped.input_dtype("rhs", [1], DType::Bool);
        let not_lhs = swapped.logical_not(lhs).unwrap();
        let output = swapped.binary(crate::BinaryOp::Add, not_lhs, rhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&swapped, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));

        let mut narrow = Graph::new();
        let lhs = narrow.input_dtype("lhs", [1], DType::I16);
        let rhs = narrow.input_dtype("rhs", [1], DType::F16);
        let output = narrow.sub(lhs, rhs).unwrap();
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::lower_graph_elementwise(&narrow, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn public_div_has_a_scoped_reciprocal_boundary() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, reciprocal_boundary, store) in [
            (DType::F16, "cvt.rn.f16.f32", "st.global.b16"),
            (DType::BF16, "shl.b32 %r91, %r91, 16", "st.global.b16"),
            (DType::F32, "cvt.rn.f32.f64", "st.global.f32"),
            (DType::F64, "mov.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 3], dtype);
            let output = graph.div(lhs, rhs).unwrap();
            assert!(matches!(graph.op(output).unwrap(), crate::Op::Binary { op: crate::BinaryOp::Mul, lhs: _, rhs }
                if matches!(graph.op(*rhs).unwrap(), crate::Op::Unary { op: crate::UnaryOp::Reciprocal, .. })));
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains("div.rn.f64 %fd30"), "{dtype:?} reciprocal");
            assert!(first.source.contains(reciprocal_boundary), "{dtype:?} reciprocal boundary");
            assert!(first.source.contains("mul.rn.f64"), "{dtype:?} final Mul");
            assert!(first.source.contains(store), "{dtype:?} store");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

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
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1], dtype);
            let rhs = graph.input_dtype("rhs", [1], dtype);
            let output = graph.div(lhs, rhs).unwrap();
            assert_eq!(graph.dtype(output).unwrap(), DType::F32);
            let rendered = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(rendered.source.contains("cvt.rn.f32"), "{dtype:?} lift");
            assert!(rendered.source.contains("div.rn.f64 %fd30"), "{dtype:?} reciprocal");
            assert!(rendered.source.contains("mul.rn.f64"), "{dtype:?} final Mul");
        }

        // The I64/U64 meet is explicitly F32 before the reciprocal, and an
        // empty broadcast remains a valid, zero-work root.
        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [0, 1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1, 3], DType::U64);
        let output = bridge.div(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert_eq!(rendered.extent, 0);
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));

        // The admitted forward root does not change Div's source-composed
        // VJP: its product/reciprocal gradient remains a separate graph.
        let mut vjp = Graph::new();
        let lhs = vjp.input_dtype("lhs", [], DType::F32);
        let rhs = vjp.input_dtype("rhs", [], DType::F32);
        let output = vjp.div(lhs, rhs).unwrap();
        let gradient = vjp.grad(vjp.sum_all(output).unwrap(), lhs).unwrap();
        assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

        // The exception is the ordered source graph only: raw DIV, swapped
        // reciprocal-Mul, affine views, and unrelated compounds stay closed.
        let mut raw = Graph::new();
        let lhs = raw.input_dtype("lhs", [1], DType::F16);
        let rhs = raw.input_dtype("rhs", [1], DType::F16);
        let output = raw.binary(crate::BinaryOp::Div, lhs, rhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&raw, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
        let mut swapped = Graph::new();
        let lhs = swapped.input_dtype("lhs", [1], DType::F16);
        let rhs = swapped.input_dtype("rhs", [1], DType::F16);
        let reciprocal = swapped.reciprocal(rhs).unwrap();
        let output = swapped.mul(reciprocal, lhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&swapped, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
        let mut gate = Graph::new();
        let lhs = gate.input_dtype("lhs", [1], DType::F16);
        let rhs = gate.input_dtype("rhs", [1], DType::F16);
        let output = gate.div(lhs, rhs).unwrap();
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::lower_graph_elementwise(&gate, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn public_eq_has_a_scoped_source_lub_predicate() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, predicate, store) in [
            (DType::Bool, "setp.eq.u8", "st.global.u8"),
            (DType::I8, "setp.eq.s8", "st.global.u8"),
            (DType::U8, "setp.eq.u8", "st.global.u8"),
            (DType::I16, "setp.eq.s16", "st.global.u8"),
            (DType::U16, "setp.eq.u16", "st.global.u8"),
            (DType::I32, "setp.eq.s32", "st.global.u8"),
            (DType::U32, "setp.eq.u32", "st.global.u8"),
            (DType::I64, "setp.eq.s64", "st.global.u8"),
            (DType::U64, "setp.eq.u64", "st.global.u8"),
            (DType::F16, "setp.eq.f32", "st.global.u8"),
            (DType::BF16, "setp.eq.f32", "st.global.u8"),
            (DType::F32, "setp.eq.f32", "st.global.u8"),
            (DType::F64, "setp.eq.f64", "st.global.u8"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 3], dtype);
            let output = graph.eq(lhs, rhs).unwrap();
            assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains(predicate), "{dtype:?} predicate");
            assert!(first.source.contains(store), "{dtype:?} Bool store");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

        // The source-only I64/U64 bridge makes their equality an F32
        // predicate after two typed casts; equal same-kind wide values retain
        // their direct integer predicate instead.
        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.eq(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("setp.eq.f32"));

        let mut narrow_cast = Graph::new();
        let lhs = narrow_cast.input_dtype("lhs", [1], DType::I16);
        let rhs = narrow_cast.input_dtype("rhs", [1], DType::F16);
        let output = narrow_cast.eq(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s16"));
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(rendered.source.contains("setp.eq.f32"));

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 1], DType::F32);
        let rhs = empty.input_dtype("rhs", [1, 3], DType::F32);
        let output = empty.eq(lhs, rhs).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        // Raw/narrow and ordered predicates, swapped compounds, and views do
        // not inherit Eq's public-root admission.
        let mut raw = Graph::new();
        let lhs = raw.input_dtype("lhs", [1], DType::I64);
        let rhs = raw.input_dtype("rhs", [1], DType::U64);
        let lhs = raw.cast(lhs, DType::F64).unwrap();
        let rhs = raw.cast(rhs, DType::F64).unwrap();
        let output = raw.compare(crate::CompareOp::Eq, lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&raw, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut ordered = Graph::new();
        let lhs = ordered.input_dtype("lhs", [1], DType::F16);
        let rhs = ordered.input_dtype("rhs", [1], DType::F16);
        let output = ordered.ne(lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&ordered, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut gate = Graph::new();
        let lhs = gate.input_dtype("lhs", [1], DType::F16);
        let rhs = gate.input_dtype("rhs", [1], DType::F16);
        let output = gate.eq(lhs, rhs).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_ne_has_an_unordered_scoped_source_lub_predicate() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, predicate) in [
            (DType::Bool, "setp.eq.u8"),
            (DType::I8, "setp.eq.s8"),
            (DType::U8, "setp.eq.u8"),
            (DType::I16, "setp.eq.s16"),
            (DType::U16, "setp.eq.u16"),
            (DType::I32, "setp.eq.s32"),
            (DType::U32, "setp.eq.u32"),
            (DType::I64, "setp.eq.s64"),
            (DType::U64, "setp.eq.u64"),
            (DType::F16, "setp.eq.f32"),
            (DType::BF16, "setp.eq.f32"),
            (DType::F32, "setp.eq.f32"),
            (DType::F64, "setp.eq.f64"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 3], dtype);
            let output = graph.ne(lhs, rhs).unwrap();
            assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains(predicate), "{dtype:?} predicate");
            // Complementing ordered equality is explicitly unordered Ne:
            // NaN is true while +0/-0 are false, matching `Scalar::compare`.
            assert!(first.source.contains("not.pred %p1, %p1"), "{dtype:?} unordered Ne");
            assert!(first.source.contains("st.global.u8"), "{dtype:?} Bool store");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.ne(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("setp.eq.f32"));

        let mut narrow_cast = Graph::new();
        let lhs = narrow_cast.input_dtype("lhs", [1], DType::U16);
        let rhs = narrow_cast.input_dtype("rhs", [1], DType::BF16);
        let output = narrow_cast.ne(lhs, rhs).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap())
            .unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.u16"));
        assert!(rendered.source.contains("selp.b32 %r91"));
        assert!(rendered.source.contains("setp.eq.f32"));

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 1], DType::F32);
        let rhs = empty.input_dtype("rhs", [1, 3], DType::F32);
        let output = empty.ne(lhs, rhs).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        let mut non_lub = Graph::new();
        let lhs = non_lub.input_dtype("lhs", [1], DType::I64);
        let rhs = non_lub.input_dtype("rhs", [1], DType::U64);
        let lhs = non_lub.cast(lhs, DType::F64).unwrap();
        let rhs = non_lub.cast(rhs, DType::F64).unwrap();
        let output = non_lub.compare(crate::CompareOp::Ne, lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&non_lub, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut ordered = Graph::new();
        let lhs = ordered.input_dtype("lhs", [1], DType::F16);
        let rhs = ordered.input_dtype("rhs", [1], DType::F16);
        let output = ordered.lt(lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&ordered, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut gate = Graph::new();
        let lhs = gate.input_dtype("lhs", [1], DType::F16);
        let rhs = gate.input_dtype("rhs", [1], DType::F16);
        let output = gate.ne(lhs, rhs).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_logical_not_has_a_scoped_typed_truthiness_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, predicate) in [
            (DType::Bool, "setp.eq.u8"),
            (DType::I8, "setp.eq.s8"),
            (DType::U8, "setp.eq.u8"),
            (DType::I16, "setp.eq.s16"),
            (DType::U16, "setp.eq.u16"),
            (DType::I32, "setp.eq.s32"),
            (DType::U32, "setp.eq.u32"),
            (DType::I64, "setp.eq.s64"),
            (DType::U64, "setp.eq.u64"),
            (DType::F16, "setp.eq.f32"),
            (DType::BF16, "setp.eq.f32"),
            (DType::F32, "setp.eq.f32"),
            (DType::F64, "setp.eq.f64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2, 1], dtype);
            let output = graph.logical_not(input).unwrap();
            assert!(matches!(graph.op(output).unwrap(), crate::Op::Compare { op: crate::CompareOp::Ne, lhs, rhs }
                if matches!(graph.op(*lhs).unwrap(), crate::Op::Cast { input: source, dtype: DType::Bool } if *source == input)
                && matches!(graph.op(*rhs).unwrap(), crate::Op::Constant(_))));
            let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            // Ordered equality then inversion makes both zero signs false at
            // the cast boundary and preserves NaN/infinity as truthy.
            assert!(first.source.contains(predicate), "{dtype:?} truthiness predicate");
            assert!(first.source.contains("mov.b8") && first.source.contains("0x01"), "{dtype:?} canonical true");
            assert!(first.source.matches("not.pred %p1, %p1").count() >= 2, "{dtype:?} cast and Ne inversions");
            assert!(first.source.contains("st.global.u8"), "{dtype:?} Bool store");
            assert!(first.source.contains(PTX_RENDERER_VERSION));
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} deterministic key");
        }

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::F64);
        let output = scalar.logical_not(input).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&scalar, output).unwrap()).unwrap().extent, 1);
        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0], DType::BF16);
        let output = empty.logical_not(input).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        let mut gate = Graph::new();
        let input = gate.input_dtype("input", [1], DType::F16);
        let output = gate.logical_not(input).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));

        // A runtime true, a raw Ne shell, and an affine view cannot inherit
        // the public literal cast/Const provenance exception.
        let mut runtime = Graph::new();
        let input = runtime.input_dtype("input", [1], DType::F32);
        let boolean = runtime.cast(input, DType::Bool).unwrap();
        let truth = runtime.input_dtype("truth", [1], DType::Bool);
        let output = runtime.compare(crate::CompareOp::Ne, boolean, truth).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&runtime, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut raw = Graph::new();
        let input = raw.input_dtype("input", [1], DType::Bool);
        let truth = raw.constant(crate::Scalar::Bool(true));
        let output = raw.compare(crate::CompareOp::Ne, input, truth).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&raw, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut viewed = Graph::new();
        let input = viewed.input_dtype("input", [1, 1], DType::F32);
        let input = viewed.permute(input, [1, 0]).unwrap();
        let output = viewed.logical_not(input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut finite = Graph::new();
        let input = finite.input_dtype("input", [1], DType::F32);
        let output = finite.isfinite(input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&finite, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_isinf_has_a_scoped_storage_bit_classification_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, required) in [
            (DType::Bool, "mov.u32"),
            (DType::I8, "mov.u32"),
            (DType::U8, "mov.u32"),
            (DType::I16, "mov.u32"),
            (DType::U16, "mov.u32"),
            (DType::I32, "mov.u32"),
            (DType::U32, "mov.u32"),
            (DType::I64, "mov.u32"),
            (DType::U64, "mov.u32"),
            (DType::F16, "0x7c00"),
            (DType::BF16, "0x7f80"),
            (DType::F32, "0x7f800000"),
            (DType::F64, "0x7ff0000000000000"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2], dtype);
            let output = graph.isinf(input).unwrap();
            assert!(matches!(graph.op(output).unwrap(), crate::Op::Unary { op: crate::UnaryOp::IsInf, input: source } if *source == input));
            let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert!(first.source.contains(required), "{dtype:?} exact classifier");
            assert!(first.source.contains("st.global.u8"), "{dtype:?} Bool store");
            assert!(first.source.contains(PTX_RENDERER_VERSION));
            assert!(matches!(&first.semantic_program, Some(KernelSemanticProgram::UOp(_))));
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} deterministic key");
        }

        // The masked exact encodings classify both infinity signs while
        // rejecting all exponent-all-ones NaNs, including signaling/payload
        // variants, without decoding a narrow lane through a float compare.
        let mut f16 = Graph::new();
        let input = f16.input_dtype("input", [1], DType::F16);
        let output = f16.isinf(input).unwrap();
        let source = renderer.render(&crate::lower_graph_elementwise(&f16, output).unwrap()).unwrap().source;
        assert!(source.contains("and.b32") && source.contains("0x7fff") && source.contains("0x7c00"));
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&f16, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::F64);
        let output = scalar.isinf(input).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&scalar, output).unwrap()).unwrap().extent, 1);
        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0], DType::BF16);
        let output = empty.isinf(input).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        // The direct raw IsInf predicate is public-equivalent. Casted/raw
        // compounds, IsNan/IsFinite, sign-select composition, and views do
        // not inherit this one-load root exception.
        let mut mixed = Graph::new();
        let input = mixed.input_dtype("input", [1], DType::F32);
        let input = mixed.cast(input, DType::F64).unwrap();
        let output = mixed.unary(crate::UnaryOp::IsInf, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&mixed, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut isnan = Graph::new();
        let input = isnan.input_dtype("input", [1], DType::F32);
        let output = isnan.isnan(input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&isnan, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut finite = Graph::new();
        let input = finite.input_dtype("input", [1], DType::F32);
        let output = finite.isfinite(input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&finite, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut signs = Graph::new();
        let input = signs.input_dtype("input", [1], DType::F32);
        let output = signs.isinf_with_signs(input, true, false).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&signs, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut viewed = Graph::new();
        let input = viewed.input_dtype("input", [1, 1], DType::F32);
        let input = viewed.permute(input, [1, 0]).unwrap();
        let output = viewed.isinf(input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut vjp = Graph::new();
        let input = vjp.input_dtype_requires_grad("input", [], DType::F32, true);
        let output = vjp.isinf(input).unwrap();
        assert!(matches!(vjp.grad(output, input), Err(crate::Error::NoGradient(_))));
    }

    #[test]
    fn public_less_and_greater_share_an_ordered_scoped_cmplt_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for greater in [false, true] {
            for (dtype, predicate) in [
                (DType::Bool, "setp.lt.u8"),
                (DType::I8, "setp.lt.s8"),
                (DType::U8, "setp.lt.u8"),
                (DType::I16, "setp.lt.s16"),
                (DType::U16, "setp.lt.u16"),
                (DType::I32, "setp.lt.s32"),
                (DType::U32, "setp.lt.u32"),
                (DType::I64, "setp.lt.s64"),
                (DType::U64, "setp.lt.u64"),
                (DType::F16, "setp.lt.f32"),
                (DType::BF16, "setp.lt.f32"),
                (DType::F32, "setp.lt.f32"),
                (DType::F64, "setp.lt.f64"),
            ] {
                let mut graph = Graph::new();
                let lhs = graph.input_dtype("lhs", [2, 1], dtype);
                let rhs = graph.input_dtype("rhs", [1, 3], dtype);
                let output = if greater { graph.gt(lhs, rhs).unwrap() } else { graph.lt(lhs, rhs).unwrap() };
                assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
                let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                assert!(first.source.contains(predicate), "{greater} {dtype:?} predicate");
                assert!(first.source.contains("st.global.u8"), "{greater} {dtype:?} Bool store");
                assert_eq!(first.cache_key, second.cache_key, "{greater} {dtype:?} key");
            }
        }

        // Greater is source-literal reversed Less, not a raw Gt predicate.
        let mut orientation = Graph::new();
        let lhs = orientation.input_dtype("lhs", [1], DType::F32);
        let rhs = orientation.input_dtype("rhs", [1], DType::F32);
        let output = orientation.gt(lhs, rhs).unwrap();
        assert!(matches!(orientation.op(output).unwrap(), crate::Op::Compare { op: crate::CompareOp::Lt, lhs: left, rhs: right }
            if *left == rhs && *right == lhs));

        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.lt(lhs, rhs).unwrap();
        let rendered = renderer.render(&crate::lower_graph_elementwise(&bridge, output).unwrap()).unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("setp.lt.f32"));

        let mut narrow_cast = Graph::new();
        let lhs = narrow_cast.input_dtype("lhs", [1], DType::I16);
        let rhs = narrow_cast.input_dtype("rhs", [1], DType::F16);
        let output = narrow_cast.lt(lhs, rhs).unwrap();
        let rendered = renderer.render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap()).unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s16"));
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(rendered.source.contains("setp.lt.f32"));

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 1], DType::F32);
        let rhs = empty.input_dtype("rhs", [1, 3], DType::F32);
        let output = empty.gt(lhs, rhs).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        // A raw Gt, source-inexact LUB casts, and Le remain outside this
        // ordered-CMPLT root exception.
        let mut raw_gt = Graph::new();
        let lhs = raw_gt.input_dtype("lhs", [1], DType::F16);
        let rhs = raw_gt.input_dtype("rhs", [1], DType::F16);
        let output = raw_gt.compare(crate::CompareOp::Gt, lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&raw_gt, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut non_lub = Graph::new();
        let lhs = non_lub.input_dtype("lhs", [1], DType::I64);
        let rhs = non_lub.input_dtype("rhs", [1], DType::U64);
        let lhs = non_lub.cast(lhs, DType::F64).unwrap();
        let rhs = non_lub.cast(rhs, DType::F64).unwrap();
        let output = non_lub.compare(crate::CompareOp::Lt, lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&non_lub, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut le = Graph::new();
        let lhs = le.input_dtype("lhs", [1], DType::F16);
        let rhs = le.input_dtype("rhs", [1], DType::F16);
        let output = le.le(lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&le, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut gate = Graph::new();
        let lhs = gate.input_dtype("lhs", [1], DType::F16);
        let rhs = gate.input_dtype("rhs", [1], DType::F16);
        let output = gate.lt(lhs, rhs).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_inclusive_comparisons_have_a_scoped_not_ordered_lt_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for greater_or_equal in [false, true] {
            for (dtype, predicate) in [
                (DType::Bool, "setp.lt.u8"),
                (DType::I8, "setp.lt.s8"),
                (DType::U8, "setp.lt.u8"),
                (DType::I16, "setp.lt.s16"),
                (DType::U16, "setp.lt.u16"),
                (DType::I32, "setp.lt.s32"),
                (DType::U32, "setp.lt.u32"),
                (DType::I64, "setp.lt.s64"),
                (DType::U64, "setp.lt.u64"),
                (DType::F16, "setp.lt.f32"),
                (DType::BF16, "setp.lt.f32"),
                (DType::F32, "setp.lt.f32"),
                (DType::F64, "setp.lt.f64"),
            ] {
                let mut graph = Graph::new();
                let lhs = graph.input_dtype("lhs", [2, 1], dtype);
                let rhs = graph.input_dtype("rhs", [1, 3], dtype);
                let output = if greater_or_equal { graph.ge(lhs, rhs).unwrap() } else { graph.le(lhs, rhs).unwrap() };
                let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
                assert!(first.source.contains(predicate), "{greater_or_equal} {dtype:?} ordered Lt");
                assert!(first.source.contains("mov.b8") && first.source.contains("0x01"), "{greater_or_equal} {dtype:?} Const Bool(true)");
                // Inverting ordered Lt makes NaN true while both zero signs
                // remain equal/true, exactly as tinygrad's literal Not path.
                assert!(first.source.contains("not.pred %p1, %p1"), "{greater_or_equal} {dtype:?} inversion");
                assert!(first.source.contains("st.global.u8"), "{greater_or_equal} {dtype:?} Bool store");
                assert_eq!(first.cache_key, second.cache_key, "{greater_or_equal} {dtype:?} key");
            }
        }

        // Le is `!(rhs < lhs)` whereas Ge is `!(lhs < rhs)`; the root proof
        // intentionally preserves that literal operand orientation.
        for greater_or_equal in [false, true] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1], DType::F32);
            let rhs = graph.input_dtype("rhs", [1], DType::F32);
            let output = if greater_or_equal { graph.ge(lhs, rhs).unwrap() } else { graph.le(lhs, rhs).unwrap() };
            let crate::Op::Compare { op: crate::CompareOp::Ne, lhs: outer, .. } = graph.op(output).unwrap() else { panic!("public inclusive outer Ne") };
            let crate::Op::Cast { input: inner, dtype: DType::Bool } = graph.op(*outer).unwrap() else { panic!("public inclusive Bool Cast") };
            let crate::Op::Compare { op: crate::CompareOp::Lt, lhs: left, rhs: right } = graph.op(*inner).unwrap() else { panic!("public inclusive inner Lt") };
            assert_eq!((*left, *right), if greater_or_equal { (lhs, rhs) } else { (rhs, lhs) });
        }

        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let output = bridge.ge(lhs, rhs).unwrap();
        let rendered = renderer.render(&crate::lower_graph_elementwise(&bridge, output).unwrap()).unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));

        let mut empty = Graph::new();
        let lhs = empty.input_dtype("lhs", [0, 1], DType::F32);
        let rhs = empty.input_dtype("rhs", [1, 3], DType::F32);
        let output = empty.le(lhs, rhs).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        // The scalar role and exact true bits are part of the root proof:
        // runtime truth buffers, false/wrong constants, raw Le, and F16 below
        // its ISA gate remain outside the exception.
        let mut runtime_truth = Graph::new();
        let lhs = runtime_truth.input_dtype("lhs", [1], DType::F16);
        let rhs = runtime_truth.input_dtype("rhs", [1], DType::F16);
        let ordered = runtime_truth.lt(lhs, rhs).unwrap();
        let boolean = runtime_truth.cast(ordered, DType::Bool).unwrap();
        let truth = runtime_truth.input_dtype("truth", [], DType::Bool);
        let output = runtime_truth.ne(boolean, truth).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&runtime_truth, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut false_truth = Graph::new();
        let lhs = false_truth.input_dtype("lhs", [1], DType::F16);
        let rhs = false_truth.input_dtype("rhs", [1], DType::F16);
        let ordered = false_truth.lt(lhs, rhs).unwrap();
        let boolean = false_truth.cast(ordered, DType::Bool).unwrap();
        let truth = false_truth.constant(TensorData::scalar_with_dtype(Scalar::Bool(false), DType::Bool));
        let output = false_truth.ne(boolean, truth).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&false_truth, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut wrong_truth = Graph::new();
        let lhs = wrong_truth.input_dtype("lhs", [1], DType::F16);
        let rhs = wrong_truth.input_dtype("rhs", [1], DType::F16);
        let ordered = wrong_truth.lt(lhs, rhs).unwrap();
        let boolean = wrong_truth.cast(ordered, DType::Bool).unwrap();
        let truth = wrong_truth.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I8));
        let output = wrong_truth.ne(boolean, truth).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&wrong_truth, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut raw = Graph::new();
        let lhs = raw.input_dtype("lhs", [1], DType::F16);
        let rhs = raw.input_dtype("rhs", [1], DType::F16);
        let output = raw.compare(crate::CompareOp::Le, lhs, rhs).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&raw, output).unwrap()), Err(PtxError::Unsupported(_))));
        let mut gate = Graph::new();
        let lhs = gate.input_dtype("lhs", [1], DType::F16);
        let rhs = gate.input_dtype("rhs", [1], DType::F16);
        let output = gate.ge(lhs, rhs).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_select_has_a_scoped_direct_mask_three_way_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, selection, store) in [
            (DType::Bool, "selp.b32", "st.global.u8"),
            (DType::I8, "selp.b32", "st.global.s8"),
            (DType::U8, "selp.b32", "st.global.u8"),
            (DType::I16, "selp.b32", "st.global.s16"),
            (DType::U16, "selp.b32", "st.global.u16"),
            (DType::I32, "selp.b32", "st.global.s32"),
            (DType::U32, "selp.b32", "st.global.u32"),
            (DType::I64, "selp.b64", "st.global.s64"),
            (DType::U64, "selp.b64", "st.global.u64"),
            // Raw b32 selection preserves direct F16/BF16 payloads and -0.
            (DType::F16, "selp.b32", "st.global.b16"),
            (DType::BF16, "selp.b32", "st.global.b16"),
            (DType::F32, "selp.f32", "st.global.f32"),
            (DType::F64, "selp.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let condition = graph.input_dtype("condition", [2, 1], DType::Bool);
            let on_true = graph.input_dtype("on_true", [1, 3], dtype);
            let on_false = graph.input_dtype("on_false", [2, 3], dtype);
            let output = graph.select(condition, on_true, on_false).unwrap();
            let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            assert_eq!(graph.shape(output).unwrap(), &crate::Shape::from([2, 3]));
            assert!(first.source.contains("setp.ne.u32 %p2"), "{dtype:?} Bool condition");
            assert!(first.source.contains(selection), "{dtype:?} typed select");
            assert!(first.source.contains(store), "{dtype:?} typed store");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

        // The source I64/U64 payload bridge is exactly F32 before the select,
        // including its logical cast boundaries.
        let mut bridge = Graph::new();
        let condition = bridge.input_dtype("condition", [1], DType::Bool);
        let on_true = bridge.input_dtype("on_true", [1], DType::I64);
        let on_false = bridge.input_dtype("on_false", [1], DType::U64);
        let output = bridge.select(condition, on_true, on_false).unwrap();
        let rendered = renderer.render(&crate::lower_graph_elementwise(&bridge, output).unwrap()).unwrap();
        assert_eq!(bridge.dtype(output).unwrap(), DType::F32);
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("selp.f32"));

        // A casted narrow branch is rounded/encoded before the raw select;
        // the direct F16 branch can therefore retain its original payload.
        let mut narrow_cast = Graph::new();
        let condition = narrow_cast.input_dtype("condition", [1], DType::Bool);
        let on_true = narrow_cast.input_dtype("on_true", [1], DType::I16);
        let on_false = narrow_cast.input_dtype("on_false", [1], DType::F16);
        let output = narrow_cast.select(condition, on_true, on_false).unwrap();
        let rendered = renderer.render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap()).unwrap();
        assert_eq!(narrow_cast.dtype(output).unwrap(), DType::F16);
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(rendered.source.contains("selp.b32"));

        let mut empty = Graph::new();
        let condition = empty.input_dtype("condition", [0, 1], DType::Bool);
        let on_true = empty.input_dtype("on_true", [1, 3], DType::F32);
        let on_false = empty.input_dtype("on_false", [0, 3], DType::F32);
        let output = empty.select(condition, on_true, on_false).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        // Select's VJP keeps the condition nondifferentiable and routes only
        // the payload gradient; the scoped forward admission adds no new VJP.
        let mut vjp = Graph::new();
        let condition = vjp.input_dtype("condition", [], DType::Bool);
        let on_true = vjp.input_dtype("on_true", [], DType::F32);
        let on_false = vjp.input_dtype("on_false", [], DType::F32);
        let output = vjp.select(condition, on_true, on_false).unwrap();
        let loss = vjp.sum_all(output).unwrap();
        let gradient = vjp.grad(loss, on_true).unwrap();
        assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

        // Source-inexact casts, views, non-Bool conditions, and F16 below
        // sm_53 remain outside this direct-mask root. Predicate roots are
        // covered separately by the strict proof below.
        let mut non_lub = Graph::new();
        let condition = non_lub.input_dtype("condition", [1], DType::Bool);
        let raw_true = non_lub.input_dtype("on_true", [1], DType::I64);
        let raw_false = non_lub.input_dtype("on_false", [1], DType::U64);
        let on_true = non_lub.cast(raw_true, DType::F64).unwrap();
        let on_false = non_lub.cast(raw_false, DType::F64).unwrap();
        let output = non_lub.select(condition, on_true, on_false).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&non_lub, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut non_bool = Graph::new();
        let condition = non_bool.input_dtype("condition", [1], DType::I8);
        let on_true = non_bool.input_dtype("on_true", [1], DType::F32);
        let on_false = non_bool.input_dtype("on_false", [1], DType::F32);
        let before = non_bool.node_count();
        assert!(non_bool.select(condition, on_true, on_false).is_err());
        assert_eq!(non_bool.node_count(), before);

        let mut viewed = Graph::new();
        let condition = viewed.input_dtype("condition", [1, 2], DType::Bool);
        let raw_true = viewed.input_dtype("on_true", [2, 1], DType::F16);
        let on_true = viewed.permute(raw_true, [1, 0]).unwrap();
        let on_false = viewed.input_dtype("on_false", [1, 2], DType::F16);
        let output = viewed.select(condition, on_true, on_false).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut gate = Graph::new();
        let condition = gate.input_dtype("condition", [1], DType::Bool);
        let on_true = gate.input_dtype("on_true", [1], DType::F16);
        let on_false = gate.input_dtype("on_false", [1], DType::F16);
        let output = gate.select(condition, on_true, on_false).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_select_reuses_only_scoped_comparison_value_conditions() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (label, predicate) in [
            ("eq", "setp.eq"),
            ("ne", "setp.eq"),
            ("lt", "setp.lt"),
            ("gt", "setp.lt"),
            ("le", "setp.lt"),
            ("ge", "setp.lt"),
        ] {
            for dtype in [DType::Bool, DType::I8, DType::U16, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32, DType::F64] {
                let mut graph = Graph::new();
                let lhs = graph.input_dtype("lhs", [2, 1], dtype);
                let rhs = graph.input_dtype("rhs", [1, 3], dtype);
                let condition = match label {
                    "eq" => graph.eq(lhs, rhs).unwrap(),
                    "ne" => graph.ne(lhs, rhs).unwrap(),
                    "lt" => graph.lt(lhs, rhs).unwrap(),
                    "gt" => graph.gt(lhs, rhs).unwrap(),
                    "le" => graph.le(lhs, rhs).unwrap(),
                    "ge" => graph.ge(lhs, rhs).unwrap(),
                    _ => unreachable!(),
                };
                let on_true = graph.input_dtype("on_true", [1, 3], DType::F16);
                let on_false = graph.input_dtype("on_false", [2, 3], DType::F16);
                let output = graph.select(condition, on_true, on_false).unwrap();
                let first = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                let second = renderer.render(&crate::lower_graph_elementwise(&graph, output).unwrap()).unwrap();
                assert!(first.source.contains(predicate), "{label} {dtype:?} predicate");
                assert!(first.source.contains("selp.b32"), "{label} {dtype:?} payload bits");
                assert!(first.source.contains("st.global.b16"), "{label} {dtype:?} payload store");
                if matches!(label, "ne" | "le" | "ge") {
                    assert!(first.source.contains("not.pred %p1, %p1"), "{label} {dtype:?} NaN-aware inversion");
                }
                assert_eq!(first.cache_key, second.cache_key, "{label} {dtype:?} key");
            }
        }

        // A predicate and payload may each use their public source-LUB casts;
        // the I64/U64 condition bridge remains F32 and the selected F16 value
        // still crosses its own typed payload boundary before raw selection.
        let mut bridge = Graph::new();
        let lhs = bridge.input_dtype("lhs", [1], DType::I64);
        let rhs = bridge.input_dtype("rhs", [1], DType::U64);
        let condition = bridge.lt(lhs, rhs).unwrap();
        let on_true = bridge.input_dtype("on_true", [1], DType::I16);
        let on_false = bridge.input_dtype("on_false", [1], DType::F16);
        let output = bridge.select(condition, on_true, on_false).unwrap();
        let rendered = renderer.render(&crate::lower_graph_elementwise(&bridge, output).unwrap()).unwrap();
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(rendered.source.contains("selp.b32"));

        // NaN and either signed zero are governed by the exact predicate
        // emission above, while the selected direct narrow payload is copied
        // bitwise. Raw Le/Ge, arbitrary logical/nested masks, affine views,
        // and source-inexact payload casts stay closed until each owns an
        // equally strict whole-root proof.
        let mut raw_le = Graph::new();
        let lhs = raw_le.input_dtype("lhs", [1], DType::F16);
        let rhs = raw_le.input_dtype("rhs", [1], DType::F16);
        let condition = raw_le.compare(crate::CompareOp::Le, lhs, rhs).unwrap();
        let on_true = raw_le.input_dtype("on_true", [1], DType::F16);
        let on_false = raw_le.input_dtype("on_false", [1], DType::F16);
        let output = raw_le.select(condition, on_true, on_false).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&raw_le, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut nested = Graph::new();
        let base = nested.input_dtype("base", [1], DType::Bool);
        let nested_true = nested.input_dtype("nested_true", [1], DType::Bool);
        let nested_false = nested.input_dtype("nested_false", [1], DType::Bool);
        let condition = nested.select(base, nested_true, nested_false).unwrap();
        let on_true = nested.input_dtype("on_true", [1], DType::F16);
        let on_false = nested.input_dtype("on_false", [1], DType::F16);
        let output = nested.select(condition, on_true, on_false).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&nested, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut logical = Graph::new();
        let lhs = logical.input_dtype("lhs", [1], DType::Bool);
        let rhs = logical.input_dtype("rhs", [1], DType::Bool);
        let condition = logical.logical_and(lhs, rhs).unwrap();
        let on_true = logical.input_dtype("on_true", [1], DType::F16);
        let on_false = logical.input_dtype("on_false", [1], DType::F16);
        let output = logical.select(condition, on_true, on_false).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&logical, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut reversed_scalar_relu = Graph::new();
        let input = reversed_scalar_relu.input_dtype("input", [1], DType::F16);
        let zero = reversed_scalar_relu.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::F16));
        let condition = reversed_scalar_relu.lt(input, zero).unwrap();
        let output = reversed_scalar_relu.select(condition, input, zero).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&reversed_scalar_relu, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut viewed = Graph::new();
        let lhs = viewed.input_dtype("lhs", [2, 1], DType::F16);
        let rhs = viewed.input_dtype("rhs", [1, 2], DType::F16);
        let condition = viewed.lt(lhs, rhs).unwrap();
        let raw_true = viewed.input_dtype("on_true", [2, 1], DType::F16);
        let on_true = viewed.permute(raw_true, [1, 0]).unwrap();
        let on_false = viewed.input_dtype("on_false", [1, 2], DType::F16);
        let output = viewed.select(condition, on_true, on_false).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut gate = Graph::new();
        let lhs = gate.input_dtype("lhs", [1], DType::F16);
        let rhs = gate.input_dtype("rhs", [1], DType::F16);
        let condition = gate.lt(lhs, rhs).unwrap();
        let on_true = gate.input_dtype("on_true", [1], DType::F16);
        let on_false = gate.input_dtype("on_false", [1], DType::F16);
        let output = gate.select(condition, on_true, on_false).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn public_relu_has_a_scoped_typed_scalar_zero_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, predicate, select, store, zero) in [
            (DType::Bool, "setp.lt.u8", "selp.b32", "st.global.u8", "mov.b8"),
            (DType::I8, "setp.lt.s8", "selp.b32", "st.global.s8", "mov.b8"),
            (DType::U8, "setp.lt.u8", "selp.b32", "st.global.u8", "mov.b8"),
            (DType::I16, "setp.lt.s16", "selp.b32", "st.global.s16", "mov.b16"),
            (DType::U16, "setp.lt.u16", "selp.b32", "st.global.u16", "mov.b16"),
            (DType::I32, "setp.lt.s32", "selp.b32", "st.global.s32", "mov.b32"),
            (DType::U32, "setp.lt.u32", "selp.b32", "st.global.u32", "mov.b32"),
            (DType::I64, "setp.lt.s64", "selp.b64", "st.global.s64", "mov.b64"),
            (DType::U64, "setp.lt.u64", "selp.b64", "st.global.u64", "mov.b64"),
            (DType::F16, "setp.lt.f32", "selp.b32", "st.global.b16", "mov.b16"),
            (DType::BF16, "setp.lt.f32", "selp.b32", "st.global.b16", "mov.b16"),
            (DType::F32, "setp.lt.f32", "selp.f32", "st.global.f32", "mov.b32"),
            (DType::F64, "setp.lt.f64", "selp.f64", "st.global.f64", "mov.b64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2, 1], dtype);
            let output = graph.relu(input).unwrap();
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            assert_eq!(graph.shape(output).unwrap(), &crate::Shape::from([2, 1]));
            assert!(first.source.contains(PTX_RENDERER_VERSION), "{dtype:?} version");
            assert!(first.source.contains(predicate), "{dtype:?} ordered zero < input");
            assert!(first.source.contains(select), "{dtype:?} raw payload select");
            assert!(first.source.contains(store), "{dtype:?} typed store");
            assert!(first.source.contains(zero) && first.source.contains("0x00"), "{dtype:?} canonical zero Const");
            assert!(matches!(&first.semantic_program, Some(KernelSemanticProgram::UOp(_))));
            assert_eq!(first.source, second.source, "{dtype:?} source");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

        // Scalar and empty descriptors retain the same complete root. The
        // ordered predicate sends -0, +0 and NaN to the raw canonical-zero
        // false payload, while positive infinity retains the input payload.
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::F64);
        let output = scalar.relu(input).unwrap();
        assert_eq!(
            renderer
                .render(&crate::lower_graph_elementwise(&scalar, output).unwrap())
                .unwrap()
                .extent,
            1
        );
        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 2], DType::BF16);
        let output = empty.relu(input).unwrap();
        assert_eq!(
            renderer
                .render(&crate::lower_graph_elementwise(&empty, output).unwrap())
                .unwrap()
                .extent,
            0
        );

        // The Select VJP owns ReLU's strict boundary routing: predicates are
        // nondifferentiable and only the selected input branch receives the
        // cotangent. This admission changes no autograd construction.
        let mut vjp = Graph::new();
        let input = vjp.input_dtype_requires_grad("input", [], DType::F32, true);
        let output = vjp.relu(input).unwrap();
        let gradient = vjp.grad(vjp.sum_all(output).unwrap(), input).unwrap();
        assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

        // Strict rejection: reversed predicate/payloads, noncanonical or
        // runtime zeros, raw Unary ReLU, and affine inputs cannot inherit the
        // public scalar-root admission.
        let mut reversed = Graph::new();
        let input = reversed.input_dtype("input", [1], DType::F16);
        let zero = reversed.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::F16));
        let condition = reversed.lt(input, zero).unwrap();
        let output = reversed.select(condition, input, zero).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&reversed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut swapped = Graph::new();
        let input = swapped.input_dtype("input", [1], DType::F16);
        let zero = swapped.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::F16));
        let condition = swapped.gt(input, zero).unwrap();
        let output = swapped.select(condition, zero, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&swapped, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut negative_zero = Graph::new();
        let input = negative_zero.input_dtype("input", [1], DType::F16);
        let zero = negative_zero.constant(
            TensorData::from_storage(crate::Shape::new([]), crate::Storage::F16(vec![0x8000]))
                .unwrap(),
        );
        let condition = negative_zero.gt(input, zero).unwrap();
        let output = negative_zero.select(condition, input, zero).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&negative_zero, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut nonzero = Graph::new();
        let input = nonzero.input_dtype("input", [1], DType::F16);
        let zero = nonzero.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::F16));
        let condition = nonzero.gt(input, zero).unwrap();
        let output = nonzero.select(condition, input, zero).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&nonzero, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut runtime_zero = Graph::new();
        let input = runtime_zero.input_dtype("input", [1], DType::F16);
        let zero = runtime_zero.input_dtype("zero", [1], DType::F16);
        let condition = runtime_zero.gt(input, zero).unwrap();
        let output = runtime_zero.select(condition, input, zero).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&runtime_zero, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut raw = Graph::new();
        let input = raw.input_dtype("input", [1], DType::F16);
        let output = raw.unary(crate::UnaryOp::Relu, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&raw, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut viewed = Graph::new();
        let raw_input = viewed.input_dtype("input", [1, 2], DType::F16);
        let input = viewed.permute(raw_input, [1, 0]).unwrap();
        let output = viewed.relu(input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut gate = Graph::new();
        let input = gate.input_dtype("input", [1], DType::F16);
        let output = gate.relu(input).unwrap();
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&crate::lower_graph_elementwise(&gate, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
    }

    #[test]
    fn public_leaky_relu_has_a_scoped_mul_select_root() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, predicate, branch, select, store) in [
            (DType::Bool, "setp.lt.u8", "and.b32", "selp.b32", "st.global.u8"),
            (DType::I8, "setp.lt.s8", "mul.lo.s32", "selp.b32", "st.global.s8"),
            (DType::U8, "setp.lt.u8", "mul.lo.u32", "selp.b32", "st.global.u8"),
            (DType::I16, "setp.lt.s16", "mul.lo.s32", "selp.b32", "st.global.s16"),
            (DType::U16, "setp.lt.u16", "mul.lo.u32", "selp.b32", "st.global.u16"),
            (DType::I32, "setp.lt.s32", "mul.lo.s32", "selp.b32", "st.global.s32"),
            (DType::U32, "setp.lt.u32", "mul.lo.u32", "selp.b32", "st.global.u32"),
            (DType::I64, "setp.lt.s64", "mul.lo.s64", "selp.b64", "st.global.s64"),
            (DType::U64, "setp.lt.u64", "mul.lo.u64", "selp.b64", "st.global.u64"),
            (DType::F16, "setp.lt.f32", "mul.rn.f64", "selp.b32", "st.global.b16"),
            (DType::BF16, "setp.lt.f32", "mul.rn.f64", "selp.b32", "st.global.b16"),
            (DType::F32, "setp.lt.f32", "mul.rn.f64", "selp.f32", "st.global.f32"),
            (DType::F64, "setp.lt.f64", "mul.rn.f64", "selp.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2, 1], dtype);
            let slope = graph.input_dtype("slope", [1, 3], dtype);
            let output = graph.leaky_relu(input, slope).unwrap();
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            assert_eq!(graph.shape(output).unwrap(), &crate::Shape::from([2, 3]));
            assert!(first.source.contains(PTX_RENDERER_VERSION), "{dtype:?} version");
            assert!(first.source.contains("mov.b") && first.source.contains("0x00"), "{dtype:?} canonical zero Const");
            assert!(first.source.contains(predicate), "{dtype:?} input < zero");
            assert!(first.source.contains(branch), "{dtype:?} slope * input");
            assert!(first.source.contains(select), "{dtype:?} typed branch select");
            assert!(first.source.contains(store), "{dtype:?} typed store");
            assert!(matches!(&first.semantic_program, Some(KernelSemanticProgram::UOp(_))));
            assert_eq!(first.source, second.source, "{dtype:?} source");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} key");
        }

        // The source LUB bridge is part of the root: both wide integer
        // operands become F32 before the product, while the predicate keeps
        // the original input storage dtype and canonical zero.
        let mut bridge = Graph::new();
        let input = bridge.input_dtype("input", [2, 1], DType::I64);
        let slope = bridge.input_dtype("slope", [1, 3], DType::U64);
        let output = bridge.leaky_relu(input, slope).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&bridge, output).unwrap())
            .unwrap();
        assert_eq!(bridge.dtype(output).unwrap(), DType::F32);
        assert!(rendered.source.contains("cvt.rn.f32.s64"));
        assert!(rendered.source.contains("cvt.rn.f32.u64"));
        assert!(rendered.source.contains("setp.lt.s64"));
        assert!(rendered.source.contains("mul.rn.f64"));

        // A nonfloat-to-narrow source LUB value crosses its F16 storage
        // boundary before the F64 product, then the product crosses the
        // distinct final F16 boundary before raw selection.
        let mut narrow_cast = Graph::new();
        let input = narrow_cast.input_dtype("input", [1], DType::I16);
        let slope = narrow_cast.input_dtype("slope", [1], DType::F16);
        let output = narrow_cast.leaky_relu(input, slope).unwrap();
        let rendered = renderer
            .render(&crate::lower_graph_elementwise(&narrow_cast, output).unwrap())
            .unwrap();
        assert_eq!(narrow_cast.dtype(output).unwrap(), DType::F16);
        assert!(rendered.source.contains("cvt.rn.f16.f32"));
        assert!(rendered.source.contains("cvt.rn.f64.f32"));
        assert!(rendered.source.contains("mul.rn.f64"));
        assert!(rendered.source.contains("selp.b32"));

        // Scalar and empty shapes retain the same complete root. The false
        // branch therefore preserves -0 and NaN payloads, while a negative
        // lane reaches the once-rounded slope product.
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("input", [], DType::F64);
        let slope = scalar.input_dtype("slope", [], DType::F64);
        let output = scalar.leaky_relu(input, slope).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&scalar, output).unwrap()).unwrap().extent, 1);
        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0, 2], DType::BF16);
        let slope = empty.input_dtype("slope", [1, 2], DType::BF16);
        let output = empty.leaky_relu(input, slope).unwrap();
        assert_eq!(renderer.render(&crate::lower_graph_elementwise(&empty, output).unwrap()).unwrap().extent, 0);

        // Select owns the strict boundary VJP; the predicate is
        // nondifferentiable and both broadcastable value branches retain
        // their normal sum-to routing.
        let mut vjp = Graph::new();
        let input = vjp.input_dtype_requires_grad("input", [2, 1], DType::F32, true);
        let slope = vjp.input_dtype_requires_grad("slope", [1, 3], DType::F32, true);
        let output = vjp.leaky_relu(input, slope).unwrap();
        let loss = vjp.sum_all(output).unwrap();
        let input_gradient = vjp.grad(loss, input).unwrap();
        let slope_gradient = vjp.grad(loss, slope).unwrap();
        assert_eq!(vjp.dtype(input_gradient).unwrap(), DType::F32);
        assert_eq!(vjp.dtype(slope_gradient).unwrap(), DType::F32);

        // Only the literal shared-input root is admitted: reversed scalar
        // tests, wrong zero payloads, swapped branches, arbitrary products,
        // affine views, and the F16 pre-SM53 path stay fail-closed.
        let mut reversed = Graph::new();
        let input = reversed.input_dtype("input", [1], DType::F16);
        let slope = reversed.input_dtype("slope", [1], DType::F16);
        let zero = reversed.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::F16));
        let condition = reversed.gt(input, zero).unwrap();
        let scaled = reversed.mul(slope, input).unwrap();
        let output = reversed.select(condition, scaled, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&reversed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut wrong_zero = Graph::new();
        let input = wrong_zero.input_dtype("input", [1], DType::F16);
        let slope = wrong_zero.input_dtype("slope", [1], DType::F16);
        let zero = wrong_zero.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::F16));
        let condition = wrong_zero.lt(input, zero).unwrap();
        let scaled = wrong_zero.mul(slope, input).unwrap();
        let output = wrong_zero.select(condition, scaled, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&wrong_zero, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut negative_zero = Graph::new();
        let input = negative_zero.input_dtype("input", [1], DType::F16);
        let slope = negative_zero.input_dtype("slope", [1], DType::F16);
        let zero = negative_zero.constant(
            TensorData::from_storage(crate::Shape::new([]), crate::Storage::F16(vec![0x8000]))
                .unwrap(),
        );
        let condition = negative_zero.lt(input, zero).unwrap();
        let scaled = negative_zero.mul(slope, input).unwrap();
        let output = negative_zero.select(condition, scaled, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&negative_zero, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut runtime_zero = Graph::new();
        let input = runtime_zero.input_dtype("input", [1], DType::F16);
        let slope = runtime_zero.input_dtype("slope", [1], DType::F16);
        let zero = runtime_zero.input_dtype("zero", [1], DType::F16);
        let condition = runtime_zero.lt(input, zero).unwrap();
        let scaled = runtime_zero.mul(slope, input).unwrap();
        let output = runtime_zero.select(condition, scaled, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&runtime_zero, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut swapped = Graph::new();
        let input = swapped.input_dtype("input", [1], DType::F16);
        let slope = swapped.input_dtype("slope", [1], DType::F16);
        let zero = swapped.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::F16));
        let condition = swapped.lt(input, zero).unwrap();
        let scaled = swapped.mul(slope, input).unwrap();
        let output = swapped.select(condition, input, scaled).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&swapped, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut unrelated = Graph::new();
        let input = unrelated.input_dtype("input", [1], DType::F16);
        let slope = unrelated.input_dtype("slope", [1], DType::F16);
        let other = unrelated.input_dtype("other", [1], DType::F16);
        let zero = unrelated.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::F16));
        let condition = unrelated.lt(input, zero).unwrap();
        let scaled = unrelated.mul(slope, other).unwrap();
        let output = unrelated.select(condition, scaled, input).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&unrelated, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut viewed = Graph::new();
        let raw_input = viewed.input_dtype("input", [1, 2], DType::F16);
        let input = viewed.permute(raw_input, [1, 0]).unwrap();
        let slope = viewed.input_dtype("slope", [2, 1], DType::F16);
        let output = viewed.leaky_relu(input, slope).unwrap();
        assert!(matches!(renderer.render(&crate::lower_graph_elementwise(&viewed, output).unwrap()), Err(PtxError::Unsupported(_))));

        let mut gate = Graph::new();
        let input = gate.input_dtype("input", [1], DType::F16);
        let slope = gate.input_dtype("slope", [1], DType::F16);
        let output = gate.leaky_relu(input, slope).unwrap();
        assert!(matches!(PtxRenderer::new(52).unwrap().render(&crate::lower_graph_elementwise(&gate, output).unwrap()), Err(PtxError::Unsupported(_))));
    }

    #[test]
    fn sign_has_a_versioned_operation_scoped_narrow_storage_abi() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, load, predicate, result) in [
            (DType::Bool, "ld.global.u8", "mov.u32", "st.global.u8"),
            (DType::I8, "ld.global.s8", "setp.lt.s8", "st.global.s8"),
            (DType::U8, "ld.global.u8", "setp.ne.u8", "st.global.u8"),
            (DType::I16, "ld.global.s16", "setp.lt.s16", "st.global.s16"),
            (DType::U16, "ld.global.u16", "setp.ne.u16", "st.global.u16"),
            (DType::I32, "ld.global.s32", "setp.lt.s32", "st.global.s32"),
            (DType::U32, "ld.global.u32", "setp.ne.u32", "st.global.u32"),
            (DType::I64, "ld.global.s64 %rd", "setp.lt.s64", "st.global.s64"),
            (DType::U64, "ld.global.u64 %rd", "setp.ne.u64", "st.global.u64"),
            (DType::F16, "cvt.rn.f32.f16", "setp.eq.f32", "st.global.b16"),
            (DType::BF16, "shl.b32", "setp.eq.f32", "st.global.b16"),
            (DType::F32, "ld.global.f32", "setp.eq.f32", "st.global.f32"),
            (DType::F64, "ld.global.f64", "setp.eq.f64", "st.global.f64"),
        ] {
            let kernel = unary_kernel(dtype, crate::UnaryOp::Sign, crate::Shape::new(vec![4]));
            let first = renderer.render(&kernel).unwrap();
            let second = renderer.render(&kernel).unwrap();
            assert!(first.source.contains(load), "{dtype:?} load");
            assert!(first.source.contains(predicate), "{dtype:?} predicate");
            assert!(first.source.contains(result), "{dtype:?} store");
            assert!(first.source.contains(PTX_RENDERER_VERSION));
            assert_eq!(first.source, second.source, "{dtype:?} source");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} cache key");
        }

        let f16 = renderer
            .render(&unary_kernel(DType::F16, crate::UnaryOp::Sign, crate::Shape::new(vec![1])))
            .unwrap()
            .source;
        assert!(f16.contains("selp.b32 %r"));
        assert!(f16.contains("0xbc00"));
        assert!(f16.contains("0x3c00"));
        let bf16 = renderer
            .render(&unary_kernel(DType::BF16, crate::UnaryOp::Sign, crate::Shape::new(vec![1])))
            .unwrap()
            .source;
        assert!(bf16.contains("0xbf80"));
        assert!(bf16.contains("0x3f80"));

        // The exception is root-scoped: merely being a narrow elementwise
        // kernel does not admit another operation, and F16 still observes its
        // explicit ISA gate before cache/module publication.
        assert!(matches!(
            renderer.render(&unary_kernel(DType::F16, crate::UnaryOp::Neg, crate::Shape::new(vec![1]))),
            Err(PtxError::Unsupported(_))
        ));
        assert!(matches!(
            PtxRenderer::new(52)
                .unwrap()
                .render(&unary_kernel(DType::F16, crate::UnaryOp::Sign, crate::Shape::new(vec![1]))),
            Err(PtxError::Unsupported(_))
        ));

        // Public Abs is admitted only as tinygrad's exact shared-input
        // Sign-times-Mul DAG.  The wider multiply and final narrow stores are
        // explicit, preserving signed minima, -0, NaN, and infinities.
        for (dtype, multiply, tail) in [
            (DType::Bool, "and.b32", "st.global.u8"),
            (DType::I8, "mul.lo.s32", "st.global.s8"),
            (DType::U8, "mul.lo.u32", "st.global.u8"),
            (DType::I16, "mul.lo.s32", "st.global.s16"),
            (DType::U16, "mul.lo.u32", "st.global.u16"),
            (DType::I32, "mul.lo.s32", "st.global.s32"),
            (DType::U32, "mul.lo.u32", "st.global.u32"),
            (DType::I64, "mul.lo.s64", "st.global.s64"),
            (DType::U64, "mul.lo.u64", "st.global.u64"),
            (DType::F16, "mul.rn.f32", "cvt.rn.f16.f32"),
            (DType::BF16, "mul.rn.f32", "selp.b32 %r91"),
            (DType::F32, "mul.rn.f32", "st.global.f32"),
            (DType::F64, "mul.rn.f64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let abs = graph.abs(input).unwrap();
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, abs).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, abs).unwrap())
                .unwrap();
            assert!(first.source.contains(multiply), "{dtype:?} Abs multiply");
            assert!(first.source.contains(tail), "{dtype:?} Abs storage");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} Abs cache key");
        }
        // Raw Unary Abs remains outside the public-composition-only admission.
        // Public Mul has its own strict root proof below.
        assert!(matches!(
            renderer.render(&unary_kernel(DType::F16, crate::UnaryOp::Abs, crate::Shape::new(vec![1]))),
            Err(PtxError::Unsupported(_))
        ));

        // Sign is discontinuous: its VJP remains an explicit typed zero and
        // therefore never needs this storage renderer to manufacture a
        // gradient kernel.
        let mut vjp = Graph::new();
        let input = vjp.input_dtype("x", [], DType::F32);
        let output = vjp.sign(input).unwrap();
        let gradient = vjp.grad(output, input).unwrap();
        assert!(matches!(
            &vjp.node(gradient).unwrap().op,
            &crate::Op::Constant(_)
        ));
    }

    #[test]
    fn public_neg_has_a_scoped_raw_storage_path() {
        let renderer = PtxRenderer::new(80).unwrap();
        for (dtype, operation, store) in [
            (DType::Bool, "setp.eq.u8", "st.global.u8"),
            (DType::I8, "neg.s32", "st.global.s8"),
            (DType::U8, "sub.u32", "st.global.u8"),
            (DType::I16, "neg.s32", "st.global.s16"),
            (DType::U16, "sub.u32", "st.global.u16"),
            (DType::I32, "neg.s32", "st.global.s32"),
            (DType::U32, "sub.u32", "st.global.u32"),
            (DType::I64, "neg.s64", "st.global.s64"),
            (DType::U64, "sub.u64", "st.global.u64"),
            (DType::F16, "xor.b32", "st.global.b16"),
            (DType::BF16, "xor.b32", "st.global.b16"),
            (DType::F32, "xor.b32", "st.global.f32"),
            (DType::F64, "xor.b64", "st.global.f64"),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], dtype);
            let output = graph.neg(input).unwrap();
            let first = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            let second = renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
            assert!(first.source.contains(operation), "{dtype:?} Neg operation");
            assert!(first.source.contains(store), "{dtype:?} Neg store");
            assert_eq!(first.cache_key, second.cache_key, "{dtype:?} Neg key");
        }
        let f16 = {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], DType::F16);
            let output = graph.neg(input).unwrap();
            renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap()
                .source
        };
        assert!(f16.contains("xor.b32"));
        assert!(!f16.contains("cvt.rn.f32.f16"));
        let mut legacy_f16 = Graph::new();
        let input = legacy_f16.input_dtype("x", [1], DType::F16);
        let output = legacy_f16.neg(input).unwrap();
        assert!(PtxRenderer::new(52)
            .unwrap()
            .render(&crate::lower_graph_elementwise(&legacy_f16, output).unwrap())
            .is_ok());
        let bf16 = {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [1], DType::BF16);
            let output = graph.neg(input).unwrap();
            renderer
                .render(&crate::lower_graph_elementwise(&graph, output).unwrap())
                .unwrap()
                .source
        };
        assert!(bf16.contains("0x8000"));
        assert!(!bf16.contains("shl.b32"));

        // The raw-bit exception is root-scoped: no other narrow unary or
        // compound expression inherits this admission.
        assert!(matches!(
            renderer.render(&unary_kernel(DType::F16, crate::UnaryOp::Abs, crate::Shape::new(vec![1]))),
            Err(PtxError::Unsupported(_))
        ));
        let mut compound = Graph::new();
        let input = compound.input_dtype("x", [1], DType::F16);
        let negated = compound.neg(input).unwrap();
        let combined = compound.add(input, negated).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&compound, combined).unwrap()),
            Err(PtxError::Unsupported(_))
        ));

        // Numeric Neg retains its source-composed reverse rule; Bool is a
        // logical predicate and remains outside differentiable paths.
        let mut vjp = Graph::new();
        let input = vjp.input_dtype("x", [], DType::F32);
        let output = vjp.neg(input).unwrap();
        let gradient = vjp.grad(output, input).unwrap();
        assert!(matches!(
            &vjp.node(gradient).unwrap().op,
            &crate::Op::Unary { op: crate::UnaryOp::Neg, .. }
        ));
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
