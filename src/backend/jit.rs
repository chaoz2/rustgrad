//! Internal graph-to-native CPU JIT execution boundary.
//!
//! Schedule integration should pass its realized `ScheduleItem` buffers through
//! this same validated ABI boundary later; this module intentionally does not
//! change scheduling or lazily realize graphs.
use super::{Backend, CpuBackend};
use crate::{
    CpuJit, Graph, JitBuffer, JitError, JitKernel, NodeId, Op, ScheduleItem, TensorData, VectorPlan,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JitFallback {
    Error,
    CpuOracle,
}

/// Crate-private typed tensor lookup used by prepared native kernels.
pub(crate) trait TensorValueStore {
    fn tensor(&self, id: u64, context: &str) -> Result<&TensorData, JitBackendError>;
}
impl TensorValueStore for BTreeMap<u64, TensorData> {
    fn tensor(&self, id: u64, context: &str) -> Result<&TensorData, JitBackendError> {
        self.get(&id).ok_or_else(|| {
            JitBackendError::Binding(format!("{context}: missing captured buffer {id}"))
        })
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JitBackendError {
    Unsupported(String),
    Binding(String),
    Native(String),
}
impl fmt::Display for JitBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "CPU JIT unsupported: {s}"),
            Self::Binding(s) => write!(f, "CPU JIT binding: {s}"),
            Self::Native(s) => write!(f, "CPU JIT native call: {s}"),
        }
    }
}
impl std::error::Error for JitBackendError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JitExecution {
    pub cache_key: String,
    pub native: bool,
    pub vector: VectorPlan,
    pub vector_main: usize,
    pub vector_tail: usize,
}
pub struct CpuJitBackend {
    fallback: JitFallback,
    vectorized: bool,
    cache: Mutex<HashMap<String, Arc<JitKernel>>>,
    schedule_modules: Mutex<HashMap<NativeScheduleWrapperKey, Vec<Arc<JitKernel>>>>,
    // Zero-domain work has no kernel to compile, but the validated skip is a
    // prepared plan whose cache ownership belongs to this backend.
    zero_domain_cache: Mutex<HashSet<u64>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct NativeScheduleWrapperKey {
    artifact: String,
    entries: Vec<String>,
}
pub(crate) struct PreparedScheduleItem {
    kernel: Arc<JitKernel>,
    pub(crate) native_cache_key: String,
    pub(crate) cache_hit: bool,
    pub(crate) vector: VectorPlan,
    schedule_cache_key: u64,
    native_layout: NativeScheduleLayout,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeScheduleModulePreparation {
    pub(crate) rendered_entry_count: usize,
    pub(crate) loaded_module_count: usize,
    pub(crate) durable_artifact_cache_hit_count: usize,
    pub(crate) durable_artifact_cache_miss_count: usize,
    pub(crate) compiler_invocation_count: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeScheduleLayout {
    pub(crate) matmul: Option<crate::cpu_jit::NativeMatmulLayouts>,
    pub(crate) retained_matmul_sources: BTreeMap<u64, u64>,
    pub(crate) elided_output_source: Option<u64>,
}

impl PreparedScheduleItem {
    pub(crate) fn abi(&self) -> &crate::KernelAbi {
        self.kernel.abi()
    }

    pub(crate) fn authenticates_layout(
        &self,
        item: &ScheduleItem,
        layout: &NativeScheduleLayout,
    ) -> bool {
        self.schedule_cache_key == item.cache_key && &self.native_layout == layout
    }

    pub(crate) fn retained_matmul_source(
        &self,
        item: &ScheduleItem,
        buffer: u64,
    ) -> Result<Option<u64>, JitBackendError> {
        if self.schedule_cache_key != item.cache_key {
            return Err(JitBackendError::Binding(
                "prepared schedule identity mismatch".into(),
            ));
        }
        let Some((plan, layouts)) = matmul_plan(item).zip(self.native_layout.matmul) else {
            return Ok(None);
        };
        let lhs = plan.lhs.index() as u64 == buffer
            && layouts.lhs == crate::cpu_jit::NativeMatmulOperandLayout::Transpose2d;
        let rhs = plan.rhs.index() as u64 == buffer
            && layouts.rhs == crate::cpu_jit::NativeMatmulOperandLayout::Transpose2d;
        let source = self
            .native_layout
            .retained_matmul_sources
            .get(&buffer)
            .copied();
        if source.is_some() != (lhs || rhs) {
            return Err(JitBackendError::Binding(
                "prepared matmul source layout mismatch".into(),
            ));
        }
        Ok(source)
    }

    pub(crate) fn elided_output_source(
        &self,
        item: &ScheduleItem,
    ) -> Result<Option<u64>, JitBackendError> {
        if self.schedule_cache_key != item.cache_key {
            return Err(JitBackendError::Binding(
                "prepared schedule identity mismatch".into(),
            ));
        }
        Ok(self.native_layout.elided_output_source)
    }
}

fn matmul_plan(item: &ScheduleItem) -> Option<&crate::MatmulKernelPlan> {
    match item.kernel.operation() {
        crate::Operation::Matmul(crate::MatmulValue::Serial(plan)) => Some(plan),
        crate::Operation::Matmul(crate::MatmulValue::Tiled(payload)) => Some(&payload.matmul),
        crate::Operation::Matmul(crate::MatmulValue::TensorCore(payload)) => Some(&payload.matmul),
        _ => None,
    }
}

fn canonical_transpose_layout(
    item: &ScheduleItem,
    node: NodeId,
    expected: &crate::Shape,
    vector: bool,
) -> Result<crate::cpu_jit::NativeMatmulOperandLayout, JitBackendError> {
    if vector || expected.rank() != 2 || expected.dims().contains(&0) {
        return Ok(crate::cpu_jit::NativeMatmulOperandLayout::Dense);
    }
    let binding = item
        .ordered_inputs()
        .iter()
        .find(|binding| binding.input_node == node)
        .ok_or_else(|| JitBackendError::Binding("matmul operand binding is absent".into()))?;
    let Some(view) = &binding.desc.view else {
        return Ok(crate::cpu_jit::NativeMatmulOperandLayout::Dense);
    };
    if is_canonical_rank2_transpose(view, expected)? {
        Ok(crate::cpu_jit::NativeMatmulOperandLayout::Transpose2d)
    } else {
        Ok(crate::cpu_jit::NativeMatmulOperandLayout::Dense)
    }
}

pub(crate) fn is_canonical_rank2_transpose(
    view: &crate::AffineView,
    expected: &crate::Shape,
) -> Result<bool, JitBackendError> {
    view.validate_read()
        .map_err(|error| JitBackendError::Binding(error.to_string()))?;
    if expected.rank() != 2 || expected.dims().contains(&0) {
        return Ok(false);
    }
    let rows = i64::try_from(expected.dims()[0]).map_err(|_| {
        JitBackendError::Binding("matmul transpose stride exceeds native address range".into())
    })?;
    Ok(view.logical_shape == *expected
        && view.source_shape.dims() == [expected.dims()[1], expected.dims()[0]]
        && view.offset == 0
        && view.strides.as_slice() == [1, rows])
}

fn canonical_scalar_transpose_copy(
    item: &ScheduleItem,
) -> Result<Option<(u64, u64, crate::AffineView)>, JitBackendError> {
    let [binding] = item.ordered_inputs() else {
        return Ok(None);
    };
    let output = item.primary_output();
    if output.shape.rank() != 2 || output.shape.dims().contains(&0) {
        return Ok(None);
    }
    let [store, end_range] = item.kernel.sources() else {
        return Ok(None);
    };
    if !matches!(item.kernel.operation(), crate::Operation::Sink)
        || !matches!(store.operation(), crate::Operation::Store)
        || !matches!(end_range.operation(), crate::Operation::EndRange)
    {
        return Ok(None);
    }
    let [output_index, value] = store.sources() else {
        return Ok(None);
    };
    let [input_index] = value.sources() else {
        return Ok(None);
    };
    let crate::Operation::Index(crate::IndexValue::Buffer {
        buffer: output_buffer,
        elements: output_elements,
        input_shape: output_input_shape,
        output_shape,
        addressing: crate::IndexAddressing::Broadcast,
    }) = output_index.operation()
    else {
        return Ok(None);
    };
    let crate::Operation::Index(crate::IndexValue::View {
        buffer: input_buffer,
        elements: input_elements,
        input_shape,
        output_shape: input_output_shape,
        view,
    }) = input_index.operation()
    else {
        return Ok(None);
    };
    let [_, input_coordinate] = input_index.sources() else {
        return Ok(None);
    };
    let [_, output_range] = output_index.sources() else {
        return Ok(None);
    };
    let [end_range_source] = end_range.sources() else {
        return Ok(None);
    };
    if !matches!(value.operation(), crate::Operation::Load)
        || !end_range_source.shares_node_with(output_range)
        || !input_coordinate.shares_node_with(output_range)
        || *output_buffer != output.id
        || *input_buffer != binding.desc.id
        || binding.input_node.index() as u64 != binding.desc.id
        || binding.desc.view.as_ref() != Some(view)
        || binding.desc.shape != view.source_shape
        || binding.desc.dtype != output.dtype
        || *output_elements
            != output.shape.numel().map_err(|error| {
                JitBackendError::Binding(format!("native transpose output shape: {error}"))
            })?
        || *input_elements
            != output.shape.numel().map_err(|error| {
                JitBackendError::Binding(format!("native transpose view shape: {error}"))
            })?
        || output_input_shape != &output.shape
        || output_shape != &output.shape
        || input_output_shape != &output.shape
        || input_shape != &view.logical_shape
        || item.node.index() as u64 != output.id
    {
        return Ok(None);
    }
    if !is_canonical_rank2_transpose(view, &output.shape)? {
        return Ok(None);
    }
    Ok(Some((binding.desc.id, output.id, view.clone())))
}

pub(crate) fn canonical_transpose_copy(
    item: &ScheduleItem,
) -> Result<Option<(u64, u64, crate::AffineView)>, JitBackendError> {
    if let crate::Operation::Movement(crate::MovementValue::Plan(plan)) = item.kernel.operation()
        && let crate::MovementKernelKind::AffineCopy { input, view } = &plan.kind
    {
        plan.validate()
            .map_err(|error| JitBackendError::Binding(error.to_string()))?;
        if plan.output.index() as u64 == item.primary_output().id
            && is_canonical_rank2_transpose(view, &plan.output_shape)?
        {
            return Ok(Some((
                input.node.index() as u64,
                item.primary_output().id,
                view.clone(),
            )));
        }
    }
    canonical_scalar_transpose_copy(item)
}

fn schedule_matmul_layouts(
    item: &ScheduleItem,
) -> Result<Option<crate::cpu_jit::NativeMatmulLayouts>, JitBackendError> {
    let Some(plan) = matmul_plan(item) else {
        return Ok(None);
    };
    let layouts = crate::cpu_jit::NativeMatmulLayouts {
        lhs: canonical_transpose_layout(item, plan.lhs, &plan.lhs_shape, plan.lhs_vector)?,
        rhs: canonical_transpose_layout(item, plan.rhs, &plan.rhs_shape, plan.rhs_vector)?,
    };
    if plan.lhs == plan.rhs && layouts.lhs != layouts.rhs {
        return Ok(Some(crate::cpu_jit::NativeMatmulLayouts::default()));
    }
    Ok(Some(layouts))
}

pub(crate) fn schedule_native_layout(
    item: &ScheduleItem,
) -> Result<NativeScheduleLayout, JitBackendError> {
    let matmul = schedule_matmul_layouts(item)?;
    let mut retained_matmul_sources = BTreeMap::new();
    if let Some((plan, layouts)) = matmul_plan(item).zip(matmul) {
        for (node, layout) in [(plan.lhs, layouts.lhs), (plan.rhs, layouts.rhs)] {
            if layout == crate::cpu_jit::NativeMatmulOperandLayout::Transpose2d {
                retained_matmul_sources.insert(node.index() as u64, node.index() as u64);
            }
        }
    }
    Ok(NativeScheduleLayout {
        matmul,
        retained_matmul_sources,
        elided_output_source: None,
    })
}

fn validate_native_layout(
    item: &ScheduleItem,
    layout: &NativeScheduleLayout,
) -> Result<(), JitBackendError> {
    match (matmul_plan(item), layout.matmul) {
        (Some(plan), Some(layouts)) => {
            let roles = [(plan.lhs, layouts.lhs), (plan.rhs, layouts.rhs)];
            for (node, operand_layout) in roles {
                let retained = layout
                    .retained_matmul_sources
                    .contains_key(&(node.index() as u64));
                if retained
                    != (operand_layout == crate::cpu_jit::NativeMatmulOperandLayout::Transpose2d)
                {
                    return Err(JitBackendError::Binding(
                        "native matmul retained-source inventory mismatch".into(),
                    ));
                }
            }
            if layout.retained_matmul_sources.keys().any(|buffer| {
                *buffer != plan.lhs.index() as u64 && *buffer != plan.rhs.index() as u64
            }) {
                return Err(JitBackendError::Binding(
                    "native matmul retained-source role is invalid".into(),
                ));
            }
        }
        (None, None) if layout.retained_matmul_sources.is_empty() => {}
        _ => {
            return Err(JitBackendError::Binding(
                "native matmul layout targets a non-matmul item".into(),
            ));
        }
    }
    if let Some(source) = layout.elided_output_source {
        let candidate = canonical_transpose_copy(item)?;
        if candidate.as_ref().map(|candidate| candidate.0) != Some(source) {
            return Err(JitBackendError::Binding(
                "native elided output source mismatch".into(),
            ));
        }
    }
    Ok(())
}

impl CpuJitBackend {
    pub fn new(fallback: JitFallback) -> Self {
        Self {
            fallback,
            vectorized: false,
            cache: Mutex::new(HashMap::new()),
            schedule_modules: Mutex::new(HashMap::new()),
            zero_domain_cache: Mutex::new(HashSet::new()),
        }
    }
    pub fn vectorized(mut self, enabled: bool) -> Self {
        self.vectorized = enabled;
        self
    }
    pub fn cache_len(&self) -> usize {
        self.cache.lock().expect("JIT cache lock").len()
            + self
                .zero_domain_cache
                .lock()
                .expect("zero-domain cache lock")
                .len()
    }

    pub(crate) fn prepare_zero_domain_schedule_item(
        &self,
        item: &ScheduleItem,
    ) -> Result<bool, JitBackendError> {
        let elements = item
            .primary_output()
            .shape
            .numel()
            .map_err(|error| JitBackendError::Binding(error.to_string()))?;
        if elements != 0 {
            return Err(JitBackendError::Binding(
                "zero-domain preparation received a non-empty output".into(),
            ));
        }
        let mut cache = self
            .zero_domain_cache
            .lock()
            .map_err(|_| JitBackendError::Native("zero-domain cache lock poisoned".into()))?;
        Ok(!cache.insert(item.cache_key))
    }
    fn render_kernel(
        &self,
        kernel: &crate::UOp,
    ) -> Result<(VectorPlan, crate::cpu_jit::RenderedC), JitBackendError> {
        let vector = if self.vectorized {
            CpuJit::vector_plan(kernel).map_err(|e| JitBackendError::Unsupported(e.to_string()))?
        } else {
            VectorPlan {
                lanes: 1,
                enabled: false,
                reason: "scalar policy disabled vector lanes".into(),
            }
        };
        let rendered = if self.vectorized {
            CpuJit::render_vectorized(kernel)
        } else {
            CpuJit::render(kernel)
        }
        .map_err(|e| JitBackendError::Unsupported(e.to_string()))?;
        Ok((vector, rendered))
    }

    fn render_schedule_kernel(
        &self,
        item: &ScheduleItem,
        layout: &NativeScheduleLayout,
    ) -> Result<
        (
            VectorPlan,
            crate::cpu_jit::RenderedC,
            Option<crate::cpu_jit::NativeMatmulLayouts>,
        ),
        JitBackendError,
    > {
        let layouts = layout.matmul;
        let (vector, rendered) = match (matmul_plan(item), layouts) {
            (Some(plan), Some(layouts)) => {
                let vector = if self.vectorized {
                    CpuJit::vector_plan(&item.kernel)
                        .map_err(|error| JitBackendError::Unsupported(error.to_string()))?
                } else {
                    VectorPlan {
                        lanes: 1,
                        enabled: false,
                        reason: "scalar policy disabled vector lanes".into(),
                    }
                };
                let rendered = CpuJit::render_matmul_with_layouts(plan, layouts)
                    .map_err(|error| JitBackendError::Unsupported(error.to_string()))?;
                (vector, rendered)
            }
            _ => self.render_kernel(&item.kernel)?,
        };
        Ok((vector, rendered, layouts))
    }

    fn compile_cached(
        &self,
        rendered: &crate::cpu_jit::RenderedC,
        cache_key: &str,
    ) -> Result<(Arc<JitKernel>, bool), JitBackendError> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| JitBackendError::Native("cache lock poisoned".into()))?;
        if let Some(compiled) = cache.get(cache_key) {
            return Ok((compiled.clone(), true));
        }
        let compiled = Arc::new(JitKernel::load(rendered).map_err(jit_error)?);
        cache.insert(cache_key.to_owned(), compiled.clone());
        Ok((compiled, false))
    }

    pub(crate) fn validate_schedule_item(
        &self,
        item: &ScheduleItem,
    ) -> Result<(), JitBackendError> {
        let layout = schedule_native_layout(item)?;
        validate_native_layout(item, &layout)?;
        let (_, rendered, _) = self.render_schedule_kernel(item, &layout)?;
        self.validate_rendered_schedule_item(item, &rendered)
    }

    fn validate_rendered_schedule_item(
        &self,
        item: &ScheduleItem,
        rendered: &crate::cpu_jit::RenderedC,
    ) -> Result<(), JitBackendError> {
        if !item.outputs.is_single() {
            return Err(JitBackendError::Unsupported(
                "native CPU JIT has no multi-output schedule ABI".into(),
            ));
        }
        item.validate_input_bindings()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        for (index, binding) in item.ordered_inputs().iter().enumerate() {
            if binding.abi_index != index {
                return Err(JitBackendError::Binding(
                    "non-contiguous schedule ABI index".into(),
                ));
            }
            let native = rendered
                .abi
                .buffers
                .iter()
                .find(|x| x.id == binding.desc.id)
                .ok_or_else(|| {
                    JitBackendError::Binding(format!(
                        "schedule buffer {} absent from native ABI",
                        binding.desc.id
                    ))
                })?;
            if native.dtype != binding.desc.dtype
                || native.elements.checked_mul(native.dtype.itemsize()) != Some(binding.desc.bytes)
                || native.mutable
            {
                return Err(JitBackendError::Binding(format!(
                    "schedule binding {index} mismatches native resource"
                )));
            }
        }
        for binding in item.ordered_quantized_inputs() {
            let native = rendered
                .abi
                .quantized_buffers
                .iter()
                .find(|resource| resource.id == binding.input_node.index() as u64)
                .ok_or_else(|| {
                    JitBackendError::Binding("quantized schedule resource absent from ABI".into())
                })?;
            if native.desc != binding.desc {
                return Err(JitBackendError::Binding(
                    "quantized schedule descriptor mismatches native ABI".into(),
                ));
            }
        }
        if rendered.abi.symbol_count != 0 {
            return Err(JitBackendError::Unsupported(
                "captured symbolic native ABI is not specialized".into(),
            ));
        }
        if rendered.abi.buffers.len() != item.ordered_inputs().len() + 1
            || rendered.abi.quantized_buffers.len() != item.ordered_quantized_inputs().len()
        {
            return Err(JitBackendError::Binding(
                "native ABI has unexpected resources".into(),
            ));
        }
        let output = rendered
            .abi
            .buffers
            .last()
            .ok_or_else(|| JitBackendError::Binding("native output missing".into()))?;
        let elements = item
            .primary_output()
            .shape
            .numel()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        if output.id != item.primary_output().id
            || output.dtype != item.primary_output().dtype
            || output.elements != elements
            || !output.mutable
        {
            return Err(JitBackendError::Binding(
                "native output descriptor mismatch".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn prepare_schedule_item(
        &self,
        item: &ScheduleItem,
    ) -> Result<PreparedScheduleItem, JitBackendError> {
        let native_layout = schedule_native_layout(item)?;
        self.prepare_schedule_item_with_layout(item, native_layout)
    }

    pub(crate) fn prepare_schedule_item_with_layout(
        &self,
        item: &ScheduleItem,
        native_layout: NativeScheduleLayout,
    ) -> Result<PreparedScheduleItem, JitBackendError> {
        validate_native_layout(item, &native_layout)?;
        let (vector, rendered, _) = self.render_schedule_kernel(item, &native_layout)?;
        self.validate_rendered_schedule_item(item, &rendered)?;
        let native_cache_key = format!("{}-schedule-{:016x}", rendered.cache_key, item.cache_key);
        let (kernel, cache_hit) = self.compile_cached(&rendered, &native_cache_key)?;
        Ok(PreparedScheduleItem {
            kernel,
            native_cache_key,
            cache_hit,
            vector,
            schedule_cache_key: item.cache_key,
            native_layout,
        })
    }

    pub(crate) fn prepare_schedule_module(
        &self,
        items: &[ScheduleItem],
        layouts: Vec<NativeScheduleLayout>,
    ) -> Result<(Vec<PreparedScheduleItem>, NativeScheduleModulePreparation), JitBackendError> {
        if items.len() != layouts.len() {
            return Err(JitBackendError::Binding(
                "native schedule module layout count mismatch".into(),
            ));
        }
        struct RenderedEntry {
            vector: VectorPlan,
            rendered: crate::cpu_jit::RenderedC,
            native_cache_key: String,
            layout: NativeScheduleLayout,
        }
        let entries = items
            .iter()
            .zip(layouts)
            .map(|(item, layout)| {
                validate_native_layout(item, &layout)?;
                let (vector, rendered, _) = self.render_schedule_kernel(item, &layout)?;
                self.validate_rendered_schedule_item(item, &rendered)?;
                let native_cache_key =
                    format!("{}-schedule-{:016x}", rendered.cache_key, item.cache_key);
                Ok(RenderedEntry {
                    vector,
                    rendered,
                    native_cache_key,
                    layout,
                })
            })
            .collect::<Result<Vec<_>, JitBackendError>>()?;
        if entries.is_empty() {
            return Ok((Vec::new(), NativeScheduleModulePreparation::default()));
        }
        let rendered = entries
            .iter()
            .map(|entry| entry.rendered.clone())
            .collect::<Vec<_>>();
        // The durable binary is source/compiler-addressed, but these wrappers
        // also carry the current schedule's concrete buffer ABI. Do not let an
        // isomorphic capture reuse wrappers authenticated for different IDs.
        let module_key = NativeScheduleWrapperKey {
            artifact: crate::cpu_jit::schedule_module_cache_key(&rendered),
            entries: entries
                .iter()
                .map(|entry| entry.native_cache_key.clone())
                .collect(),
        };
        let cached_module = self
            .schedule_modules
            .lock()
            .map_err(|_| JitBackendError::Native("schedule module cache lock poisoned".into()))?
            .get(&module_key)
            .cloned();
        let (kernels, load) = match cached_module {
            Some(kernels) => {
                if kernels.len() != rendered.len()
                    || kernels
                        .iter()
                        .zip(&rendered)
                        .any(|(kernel, rendered)| kernel.abi() != &rendered.abi)
                {
                    return Err(JitBackendError::Binding(
                        "cached native schedule module ABI mismatch".into(),
                    ));
                }
                (kernels, None)
            }
            None => {
                let (kernels, load) =
                    JitKernel::load_schedule_module(&rendered).map_err(jit_error)?;
                let kernels = kernels.into_iter().map(Arc::new).collect::<Vec<_>>();
                self.schedule_modules
                    .lock()
                    .map_err(|_| {
                        JitBackendError::Native("schedule module cache lock poisoned".into())
                    })?
                    .insert(module_key, kernels.clone());
                (kernels, Some(load))
            }
        };
        if kernels.len() != entries.len() {
            return Err(JitBackendError::Binding(
                "native schedule module entry count mismatch".into(),
            ));
        }
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| JitBackendError::Native("cache lock poisoned".into()))?;
        let mut prepared = Vec::with_capacity(entries.len());
        for ((item, entry), kernel) in items.iter().zip(entries).zip(kernels) {
            let cache_hit = cache
                .insert(entry.native_cache_key.clone(), kernel.clone())
                .is_some();
            prepared.push(PreparedScheduleItem {
                kernel,
                native_cache_key: entry.native_cache_key,
                cache_hit,
                vector: entry.vector,
                schedule_cache_key: item.cache_key,
                native_layout: entry.layout,
            });
        }
        Ok((
            prepared,
            NativeScheduleModulePreparation {
                rendered_entry_count: rendered.len(),
                loaded_module_count: 1,
                durable_artifact_cache_hit_count: load
                    .map(|load| usize::from(load.durable_cache_hit))
                    .unwrap_or(0),
                durable_artifact_cache_miss_count: load
                    .map(|load| usize::from(!load.durable_cache_hit))
                    .unwrap_or(0),
                compiler_invocation_count: load
                    .map(|load| load.compiler_invocation_count)
                    .unwrap_or(0),
            },
        ))
    }

    pub(crate) fn execute_prepared_schedule_item<V: TensorValueStore>(
        &self,
        item: &ScheduleItem,
        values: &V,
        quantized_values: &BTreeMap<u64, crate::QuantizedTensorData>,
        prepared: &PreparedScheduleItem,
    ) -> Result<(TensorData, JitExecution), JitBackendError> {
        if prepared.schedule_cache_key != item.cache_key {
            return Err(JitBackendError::Binding(
                "prepared schedule identity mismatch".into(),
            ));
        }
        prepared.retained_matmul_source(item, u64::MAX)?;
        let mut buffers = Vec::with_capacity(prepared.kernel.abi().buffers.len());
        for desc in &prepared.kernel.abi().buffers {
            if desc.id == item.primary_output().id {
                buffers.push(JitBuffer::zeroed(desc.dtype, desc.elements, true));
            } else {
                let value = values.tensor(desc.id, "prepared schedule")?;
                let binding = item
                    .ordered_inputs()
                    .iter()
                    .find(|binding| binding.desc.id == desc.id)
                    .ok_or_else(|| {
                        JitBackendError::Binding(format!(
                            "prepared schedule input {} has no binding",
                            desc.id
                        ))
                    })?;
                let retained_source = prepared.retained_matmul_source(item, desc.id)?;
                if retained_source.is_some_and(|source| source != desc.id) {
                    return Err(JitBackendError::Binding(
                        "cross-item retained matmul input requires prepared workspace".into(),
                    ));
                }
                let logical = if retained_source.is_some() {
                    std::borrow::Cow::Borrowed(value)
                } else {
                    crate::engine::direct_matmul_input(item, binding, value)
                        .map_err(JitBackendError::Binding)?
                };
                buffers.push(JitBuffer::from_tensor(logical.as_ref(), false));
            }
        }
        if prepared.kernel.abi().quantized_buffers.is_empty() {
            prepared.kernel.call(&mut buffers, &[]).map_err(jit_error)?;
        } else {
            let quantized = prepared
                .kernel
                .abi()
                .quantized_buffers
                .iter()
                .map(|desc| {
                    quantized_values.get(&desc.id).ok_or_else(|| {
                        JitBackendError::Binding(format!(
                            "missing packed captured buffer {}",
                            desc.id
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            prepared
                .kernel
                .call_with_quantized(&mut buffers, &quantized, &[])
                .map_err(jit_error)?;
        }
        let output_index = prepared
            .kernel
            .abi()
            .buffers
            .iter()
            .position(|x| x.id == item.primary_output().id)
            .ok_or_else(|| JitBackendError::Binding("native output missing".into()))?;
        let output_elements = item
            .primary_output()
            .shape
            .numel()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        let output = buffers
            .swap_remove(output_index)
            .into_tensor(item.primary_output().shape.clone())
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        Ok((output, prepared_execution(prepared, output_elements)))
    }

    pub(crate) fn execute_prepared_schedule_item_in_workspace(
        &self,
        item: &ScheduleItem,
        buffers: &mut [JitBuffer],
        slots: &[usize],
        borrowed: Option<&mut BTreeMap<usize, crate::cpu_jit::BorrowedJitBuffer<'_>>>,
        quantized_values: &BTreeMap<u64, crate::QuantizedTensorData>,
        prepared: &PreparedScheduleItem,
    ) -> Result<JitExecution, JitBackendError> {
        if prepared.schedule_cache_key != item.cache_key
            || slots.len() != prepared.kernel.abi().buffers.len()
        {
            return Err(JitBackendError::Binding(
                "prepared workspace schedule identity mismatch".into(),
            ));
        }
        prepared.retained_matmul_source(item, u64::MAX)?;
        let quantized = prepared
            .kernel
            .abi()
            .quantized_buffers
            .iter()
            .map(|desc| {
                quantized_values.get(&desc.id).ok_or_else(|| {
                    JitBackendError::Binding(format!("missing packed captured buffer {}", desc.id))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        match borrowed {
            Some(borrowed) => prepared
                .kernel
                .call_indexed_detached_borrowed(buffers, slots, borrowed, &quantized),
            None => prepared
                .kernel
                .call_indexed_detached(buffers, slots, &quantized),
        }
        .map_err(jit_error)?;
        let output_elements = item
            .primary_output()
            .shape
            .numel()
            .map_err(|error| JitBackendError::Binding(error.to_string()))?;
        Ok(prepared_execution(prepared, output_elements))
    }
    pub fn execute_native(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<(TensorData, JitExecution), JitBackendError> {
        let kernel = match graph
            .op(output)
            .map_err(|e| JitBackendError::Binding(e.to_string()))?
        {
            Op::Reduce {
                kind: crate::ReduceKind::Any | crate::ReduceKind::All,
                ..
            } => {
                return Err(JitBackendError::Unsupported(
                    "boolean reductions are CPU-oracle only".into(),
                ));
            }
            Op::PrefixScan { .. } | Op::TensorGuard { .. } => {
                return Err(JitBackendError::Unsupported(
                    "prefix scans are CPU-oracle only".into(),
                ));
            }
            Op::Sort { .. } => {
                return Err(JitBackendError::Unsupported(
                    "stable sort pairs are CPU-oracle only".into(),
                ));
            }
            Op::Reduce { .. } => crate::lower_graph_reduction(graph, output),
            Op::Matmul { .. } => crate::lower_graph_matmul(graph, output),
            Op::Bitcast { .. }
            | Op::Contiguous { .. }
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::Scatter { .. }
            | Op::ScatterPositions { .. }
            | Op::ScatterPositionsVjp { .. } => crate::lower_graph_movement(graph, output),
            _ => crate::lower_graph_elementwise(graph, output),
        }
        .map_err(|e| JitBackendError::Unsupported(e.to_string()))?;
        let (vector, rendered) = self.render_kernel(&kernel)?;
        let (compiled, _) = self.compile_cached(&rendered, &rendered.cache_key)?;
        let mut buffers = Vec::with_capacity(compiled.abi().buffers.len());
        for desc in &compiled.abi().buffers {
            let id = NodeId::from_index(desc.id as usize);
            let node = graph
                .nodes
                .get(id.index())
                .ok_or_else(|| JitBackendError::Binding(format!("unknown buffer {}", desc.id)))?;
            let value = match &node.op {
                Op::Input { name } => inputs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| JitBackendError::Binding(format!("missing input {name}")))?,
                Op::Constant(v) => v.clone(),
                _ if id == output => TensorData::from_scalars(
                    node.shape.clone(),
                    node.dtype,
                    (0..node
                        .shape
                        .numel()
                        .map_err(|e| JitBackendError::Binding(e.to_string()))?)
                        .map(|_| crate::Scalar::I(0)),
                )
                .map_err(|e| JitBackendError::Binding(e.to_string()))?,
                _ => {
                    return Err(JitBackendError::Binding(format!(
                        "buffer {} is not an input, constant, or output",
                        desc.id
                    )));
                }
            };
            buffers.push(JitBuffer::from_tensor(&value, desc.mutable));
        }
        compiled.call(&mut buffers, &[]).map_err(jit_error)?;
        let out_index = compiled
            .abi()
            .buffers
            .iter()
            .position(|b| b.id == output.index() as u64)
            .ok_or_else(|| JitBackendError::Binding("output missing from ABI".into()))?;
        let shape = graph
            .shape(output)
            .map_err(|e| JitBackendError::Binding(e.to_string()))?
            .clone();
        let output_elements = shape
            .numel()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        let result = buffers
            .swap_remove(out_index)
            .into_tensor(shape)
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        Ok((
            result,
            JitExecution {
                cache_key: rendered.cache_key,
                native: true,
                vector_main: if vector.enabled {
                    output_elements / vector.lanes * vector.lanes
                } else {
                    0
                },
                vector_tail: if vector.enabled {
                    output_elements % vector.lanes
                } else {
                    output_elements
                },
                vector,
            },
        ))
    }
    pub fn execute(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<(TensorData, JitExecution), JitBackendError> {
        match self.execute_native(graph, output, inputs) {
            Ok(x) => Ok(x),
            Err(_e) if self.fallback == JitFallback::CpuOracle => Ok((
                CpuBackend
                    .execute(graph, output, inputs)
                    .map_err(|x| JitBackendError::Native(x.to_string()))?,
                JitExecution {
                    cache_key: "cpu-fallback".into(),
                    native: false,
                    vector: VectorPlan {
                        lanes: 1,
                        enabled: false,
                        reason: "CPU oracle fallback".into(),
                    },
                    vector_main: 0,
                    vector_tail: 0,
                },
            )),
            Err(e) => Err(e),
        }
    }
}

fn prepared_execution(prepared: &PreparedScheduleItem, output_elements: usize) -> JitExecution {
    JitExecution {
        cache_key: prepared.native_cache_key.clone(),
        native: true,
        vector_main: if prepared.vector.enabled {
            output_elements / prepared.vector.lanes * prepared.vector.lanes
        } else {
            0
        },
        vector_tail: if prepared.vector.enabled {
            output_elements % prepared.vector.lanes
        } else {
            output_elements
        },
        vector: prepared.vector.clone(),
    }
}

fn jit_error(e: JitError) -> JitBackendError {
    match e {
        JitError::Unsupported(s) | JitError::Symbolic(s) => JitBackendError::Unsupported(s),
        other => JitBackendError::Native(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Scalar, Shape};
    #[test]
    fn native_graph_boundary_matches_oracle_and_caches() {
        let mut g = Graph::new();
        let x = g.input_dtype("x", Shape::from([5]), DType::F32);
        let y = g.square(x).unwrap();
        let input =
            TensorData::from_scalars([5], DType::F32, (0..5).map(|v| Scalar::F(v as f64 - 2.)))
                .unwrap();
        let inputs = HashMap::from([("x".into(), input)]);
        let b = CpuJitBackend::new(JitFallback::Error).vectorized(true);
        let (a, first) = b.execute(&g, y, &inputs).unwrap();
        let (c, second) = b.execute(&g, y, &inputs).unwrap();
        let expected = CpuBackend.execute(&g, y, &inputs).unwrap();
        assert!(first.native && second.native);
        assert_eq!(a.storage(), expected.storage());
        assert_eq!(c.storage(), expected.storage());
        assert_eq!(b.cache_len(), 1);
    }

    #[test]
    fn native_llama_unaries_match_oracle() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let exp = graph.exp(input).unwrap();
        let reciprocal = graph.reciprocal(exp).unwrap();
        let output = graph.rsqrt(reciprocal).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::new([3], vec![-1.0, 0.0, 1.0]).unwrap(),
        )]);
        let actual = CpuJitBackend::new(JitFallback::Error)
            .execute_native(&graph, output, &inputs)
            .unwrap()
            .0;
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        for (actual, expected) in actual.values().iter().zip(expected.values()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn native_float_extrema_reductions_match_oracle() {
        for (dtype, maximum) in [DType::F32, DType::F64]
            .into_iter()
            .flat_map(|dtype| [false, true].map(move |maximum| (dtype, maximum)))
        {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", Shape::from([3, 3]), dtype);
            let output = graph
                .reduce(
                    input,
                    if maximum {
                        crate::ReduceKind::Max
                    } else {
                        crate::ReduceKind::Min
                    },
                    Some(vec![1]),
                    true,
                )
                .unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [3, 3],
                    dtype,
                    [
                        f64::NAN,
                        2.0,
                        -1.0,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        -0.0,
                        0.0,
                        0.0,
                    ]
                    .map(Scalar::F),
                )
                .unwrap(),
            )]);
            let actual = CpuJitBackend::new(JitFallback::Error)
                .execute_native(&graph, output, &inputs)
                .unwrap()
                .0;
            let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
            assert_eq!(
                actual.to_le_bytes().unwrap(),
                expected.to_le_bytes().unwrap(),
                "{dtype:?} maximum={maximum}"
            );
        }
    }

    #[test]
    fn native_materializing_movements_match_oracle_and_check_indices() {
        let backend = CpuJitBackend::new(JitFallback::Error);

        let mut concat_graph = Graph::new();
        let left = concat_graph.input_dtype("left", [2, 1], DType::F32);
        let right = concat_graph.input_dtype("right", [2, 2], DType::F32);
        let concat = concat_graph.concat(vec![left, right], 1).unwrap();
        let concat_inputs = HashMap::from([
            (
                "left".into(),
                TensorData::new([2, 1], vec![1.0, 4.0]).unwrap(),
            ),
            (
                "right".into(),
                TensorData::new([2, 2], vec![2.0, 3.0, 5.0, 6.0]).unwrap(),
            ),
        ]);
        let native = backend
            .execute_native(&concat_graph, concat, &concat_inputs)
            .unwrap()
            .0;
        let oracle = CpuBackend
            .execute(&concat_graph, concat, &concat_inputs)
            .unwrap();
        assert_eq!(native.storage(), oracle.storage());

        let mut indexed = Graph::new();
        let base = indexed.input_dtype("base", [2, 3], DType::F32);
        let index = indexed.input_dtype("index", [2, 2], DType::I64);
        let updates = indexed.input_dtype("updates", [2, 2], DType::F32);
        let gather = indexed.gather(base, index, 1).unwrap();
        let scatter = indexed.scatter_add(base, index, updates, 1).unwrap();
        let indexed_inputs = HashMap::from([
            (
                "base".into(),
                TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_scalars(
                    [2, 2],
                    DType::I64,
                    [2_i64, 0, 1, 1].map(crate::Scalar::I),
                )
                .unwrap(),
            ),
            (
                "updates".into(),
                TensorData::new([2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
            ),
        ]);
        for output in [gather, scatter] {
            let native = backend
                .execute_native(&indexed, output, &indexed_inputs)
                .unwrap()
                .0;
            let oracle = CpuBackend
                .execute(&indexed, output, &indexed_inputs)
                .unwrap();
            assert_eq!(native.storage(), oracle.storage());
        }

        let mut invalid = indexed_inputs;
        invalid.insert(
            "index".into(),
            TensorData::from_scalars([2, 2], DType::I64, [2_i64, 0, 3, 1].map(crate::Scalar::I))
                .unwrap(),
        );
        assert!(matches!(
            backend.execute_native(&indexed, gather, &invalid),
            Err(JitBackendError::Native(message)) if message.contains("out of bounds at 2")
        ));

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input_dtype("empty", [0, 2], DType::F32);
        let empty_index = empty_graph.input_dtype("empty_index", [0, 1], DType::I64);
        let empty_concat = empty_graph.concat([empty, empty], 0).unwrap();
        let empty_gather = empty_graph.gather(empty, empty_index, 1).unwrap();
        let empty_inputs = HashMap::from([
            (
                "empty".into(),
                TensorData::new([0, 2], Vec::<f32>::new()).unwrap(),
            ),
            (
                "empty_index".into(),
                TensorData::from_scalars([0, 1], DType::I64, []).unwrap(),
            ),
        ]);
        for output in [empty_concat, empty_gather] {
            let actual = backend
                .execute_native(&empty_graph, output, &empty_inputs)
                .unwrap()
                .0;
            assert!(actual.is_empty());
        }
    }

    #[test]
    fn vector_trace_covers_main_tail_and_scalar_fallbacks() {
        for len in [0usize, 1, 3, 4, 5, 8, 17] {
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", Shape::from([len]), DType::F32);
            let y = graph.square(x).unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [len],
                    DType::F32,
                    (0..len).map(|value| Scalar::F(value as f64 - 3.0)),
                )
                .unwrap(),
            )]);
            let backend = CpuJitBackend::new(JitFallback::Error).vectorized(true);
            let (actual, trace) = backend.execute_native(&graph, y, &inputs).unwrap();
            assert_eq!(
                actual.storage(),
                CpuBackend.execute(&graph, y, &inputs).unwrap().storage()
            );
            assert!(trace.vector.enabled);
            assert_eq!(trace.vector.lanes, 4);
            assert_eq!(trace.vector_main, len / 4 * 4);
            assert_eq!(trace.vector_tail, len % 4);
        }

        let mut scalar = Graph::new();
        let x = scalar.input_dtype("x", Shape::from([5]), DType::F64);
        let y = scalar.square(x).unwrap();
        let (_, trace) = CpuJitBackend::new(JitFallback::Error)
            .vectorized(true)
            .execute_native(
                &scalar,
                y,
                &HashMap::from([(
                    "x".into(),
                    TensorData::from_scalars([5], DType::F64, (0..5).map(|_| Scalar::F(1.0)))
                        .unwrap(),
                )]),
            )
            .unwrap();
        assert!(trace.vector.enabled);
        assert_eq!(trace.vector.lanes, 2);

        let mut broadcast = Graph::new();
        let lhs = broadcast.input_dtype("lhs", Shape::from([2, 3]), DType::F32);
        let rhs = broadcast.input_dtype("rhs", Shape::from([1, 3]), DType::F32);
        let out = broadcast.add(lhs, rhs).unwrap();
        let (_, trace) = CpuJitBackend::new(JitFallback::Error)
            .vectorized(true)
            .execute_native(
                &broadcast,
                out,
                &HashMap::from([
                    ("lhs".into(), TensorData::new([2, 3], vec![1.; 6]).unwrap()),
                    ("rhs".into(), TensorData::new([1, 3], vec![2.; 3]).unwrap()),
                ]),
            )
            .unwrap();
        assert!(!trace.vector.enabled);
        assert!(trace.vector.reason.contains("varying broadcast"));
    }
    #[test]
    fn newly_supported_compositions_do_not_trigger_fallback() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::from([2]));
        let y = g.sinh(x).unwrap();
        let inputs = HashMap::from([("x".into(), TensorData::new([2], vec![1., 2.]).unwrap())]);
        let (_, strict_trace) = CpuJitBackend::new(JitFallback::Error)
            .execute(&g, y, &inputs)
            .unwrap();
        assert!(strict_trace.native);
        let (result, trace) = CpuJitBackend::new(JitFallback::CpuOracle)
            .execute(&g, y, &inputs)
            .unwrap();
        assert!(trace.native);
        assert_eq!(result.to_vec_f64().len(), 2);
    }

    #[test]
    fn native_reduction_and_narrow_select_match_oracle() {
        let mut reduce = Graph::new();
        let x = reduce.input_dtype("x", Shape::from([2, 2]), DType::BF16);
        let out = reduce
            .reduce(x, crate::ReduceKind::Mean, Some(vec![1]), true)
            .unwrap();
        let input = TensorData::from_scalars(
            [2, 2],
            DType::BF16,
            [Scalar::F(1.), Scalar::F(2.), Scalar::F(-3.), Scalar::F(4.)],
        )
        .unwrap();
        let inputs = HashMap::from([("x".into(), input)]);
        let backend = CpuJitBackend::new(JitFallback::Error);
        let (actual, trace) = backend.execute(&reduce, out, &inputs).unwrap();
        assert!(trace.native);
        assert_eq!(
            actual.storage(),
            CpuBackend.execute(&reduce, out, &inputs).unwrap().storage()
        );

        let mut select = Graph::new();
        let a = select.input_dtype("a", Shape::from([2]), DType::I32);
        let b = select.input_dtype("b", Shape::from([2]), DType::I32);
        let condition = select.gt(a, b).unwrap();
        let out = select.select(condition, a, b).unwrap();
        let inputs = HashMap::from([
            (
                "a".into(),
                TensorData::from_scalars([2], DType::I32, [Scalar::I(1), Scalar::I(-4)]).unwrap(),
            ),
            (
                "b".into(),
                TensorData::from_scalars([2], DType::I32, [Scalar::I(2), Scalar::I(-5)]).unwrap(),
            ),
        ]);
        let (actual, _) = CpuJitBackend::new(JitFallback::Error)
            .execute(&select, out, &inputs)
            .unwrap();
        assert_eq!(
            actual.storage(),
            CpuBackend.execute(&select, out, &inputs).unwrap().storage()
        );
    }
}
