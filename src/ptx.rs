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

pub const PTX_RENDERER_VERSION: &str = "rustgrad-ptx-elementwise-v11";
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
    // Reciprocal, Mul, Add, and Sub roots; each has a source-proven, typed storage
    // boundary.
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
/// must first prove the whole root is an unadorned Sign kernel.
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
    Mul,
    Add,
    Sub,
    SubBool,
}

/// Validates the only roots allowed to use the narrow-storage ABI. The Abs
/// arm is the exact public `x * x.sign()` DAG; Reciprocal accepts either its
/// direct floating ALU root or its exact public nonfloat `Cast(F32)` root;
/// Mul/Add/Sub delegate to a two-input source-LUB proof. None is a generic unary,
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
        && !matches!(
            value.sources().get(1).map(|node| node.kind()),
            Some(UOpKind::GraphUnary(crate::UnaryOp::Sign))
        )
    {
        return scoped_binary_plan(store, sm, crate::BinaryOp::Mul, ScopedStorageMode::Mul);
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
    if (reciprocal_direct
        && (!input_dtype.is_float() || input_dtype != dtype))
        || (reciprocal_cast && (input_dtype.is_float() || dtype != DType::F32))
        || (!reciprocal_direct
            && !reciprocal_cast
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

/// A PTX Mul exception is intentionally a whole-root proof, not a generic
/// binary admission. Each operand is a direct public input or its one source
/// LUB Cast, and both index descriptors are the exact output broadcast domain.
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
    if output_dtype == DType::F16 && sm < 53 {
        return Err(PtxError::Unsupported(
            "F16 scoped Mul conversion requires sm_53 or newer".into(),
        ));
    }
    reject_sign_storage_dtype(output_dtype)?;
    Ok(Some(mode))
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
        && mode != Some(ScopedStorageMode::Mul)
        && mode != Some(ScopedStorageMode::Add)
        && mode != Some(ScopedStorageMode::Sub)
    {
        return value;
    }
    let reciprocal = matches!(
        mode,
        Some(ScopedStorageMode::Reciprocal | ScopedStorageMode::ReciprocalCast)
    );
    let scoped_binary = matches!(
        mode,
        Some(ScopedStorageMode::Mul | ScopedStorageMode::Add | ScopedStorageMode::Sub)
    );
    match dtype {
        DType::F16 => {
            if reciprocal || scoped_binary {
                lines.push(format!("  cvt.rn.f32.f64 %f31, {value};"));
                lines.push("  cvt.rn.f16.f32 %r91, %f31;".into());
                return "%r91".into();
            }
            lines.push(format!("  cvt.rn.f16.f32 %r91, {value};"));
            "%r91".into()
        }
        DType::BF16 => {
            if reciprocal || scoped_binary {
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
        DType::F32 if reciprocal || scoped_binary => {
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
        _ if matches!(
            storage_mode,
            Some(ScopedStorageMode::Reciprocal | ScopedStorageMode::ReciprocalCast)
        )
            && matches!(n.kind(), UOpKind::GraphUnary(crate::UnaryOp::Reciprocal)) =>
        {
            format!("%fd{id}")
        }
        _ if matches!(
            storage_mode,
            Some(ScopedStorageMode::Mul | ScopedStorageMode::Add | ScopedStorageMode::Sub)
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
                    if storage_mode != Some(ScopedStorageMode::Neg) {
                        lines.push(format!("  cvt.rn.f32.f16 {dst}, %r{id};"));
                    }
                }
                DType::BF16 => {
                    lines.push(format!("  ld.global.b16 %r{id}, [%rd29];"));
                    if storage_mode != Some(ScopedStorageMode::Neg) {
                        lines.push(format!("  shl.b32 %r90, %r{id}, 16;"));
                        lines.push(format!("  mov.b32 {dst}, %r90;"));
                    }
                }
                _ => lines.push(format!("  ld.global.{} {dst}, [%rd29];", ptx_type(ty))),
            }
        }
        UOpKind::Cast if matches!(
            storage_mode,
            Some(ScopedStorageMode::Mul | ScopedStorageMode::Add | ScopedStorageMode::Sub)
        ) => {
            let a = child(0)?;
            let source = n.sources()[0]
                .ty()
                .ok_or_else(|| PtxError::Unsupported("untyped Mul Cast input".into()))?
                .scalar;
            emit_typed_binary_cast(lines, &dst, a, source, ty)?;
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
            if matches!(
                storage_mode,
                Some(
                    ScopedStorageMode::Mul
                        | ScopedStorageMode::Add
                        | ScopedStorageMode::Sub
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
            lines.push(format!("  selp.{} {dst}, {a}, {b}, %p2;", ptx_type(ty)));
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
    use crate::{Backend, CpuBackend, Driver, Graph, TensorData, UOp, UType};
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
    fn extrema_ptx_uses_ordered_predicate_select_or_rejects_narrow_float() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [1], DType::F32);
        let rhs = graph.input_dtype("rhs", [1], DType::F32);
        let maximum = graph.maximum(lhs, rhs).unwrap();
        let minimum = graph.minimum(lhs, rhs).unwrap();
        let renderer = PtxRenderer::new(80).unwrap();
        let maximum = renderer
            .render(&crate::lower_graph_elementwise(&graph, maximum).unwrap())
            .unwrap()
            .source;
        let minimum = renderer
            .render(&crate::lower_graph_elementwise(&graph, minimum).unwrap())
            .unwrap()
            .source;
        assert!(maximum.contains("setp.lt.f32"));
        assert!(minimum.contains("setp.gt.f32"));
        assert!(maximum.contains("selp.f32"));
        assert!(minimum.contains("selp.f32"));
        assert!(!maximum.contains("max."));
        assert!(!minimum.contains("min."));

        let mut narrow = Graph::new();
        let lhs = narrow.input_dtype("lhs", [1], DType::F16);
        let rhs = narrow.input_dtype("rhs", [1], DType::F16);
        let output = narrow.maximum(lhs, rhs).unwrap();
        assert!(matches!(
            renderer.render(&crate::lower_graph_elementwise(&narrow, output).unwrap()),
            Err(PtxError::Unsupported(_))
        ));
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
            (DType::F64, crate::UnaryOp::Sqrt),
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
