//! Deterministic WGSL lowering for a static exact-storage scalar/reduction subset.
use super::{
    WebGpuCapabilities, WebGpuError,
    guard::emit_transactional,
    narrow::{self, WEBGPU_NARROW_ABI_VERSION},
    transaction::WebGpuTransactionAbi,
};
use crate::{
    AffineView, DType, IndexValue, LiteralValue, MovementValue, Operation, ScheduleInputBinding,
    Shape, UOp,
    runtime::scalar_lane::{
        ScalarLaneDialect, dialect_seal, emit_scalar_lane, project_scalar_lane,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Deterministic renderer/source identity.
pub const WGSL_RENDERER_VERSION: &str = "rustgrad-wgsl-static-v9";
pub const WGSL_RAW_COPY_RENDERER_VERSION: &str = "rustgrad-wgsl-raw-copy-v1";
pub const WGSL_PORTABLE_BITCAST_RENDERER_VERSION: &str = "rustgrad-wgsl-portable-bitcast-v1";
pub const WGSL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION: &str =
    "rustgrad-wgsl-portable-dense-materialization-v1";
pub const WGSL_STATIC_POSITION_RENDERER_VERSION: &str = "rustgrad-wgsl-static-position-v1";
pub const WGSL_PORTABLE_F32_MATMUL_RENDERER_VERSION: &str = "rustgrad-wgsl-portable-f32-matmul-v1";
pub const WGSL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION: &str =
    "rustgrad-wgsl-portable-prefix-scan-v1";
pub const WGSL_PORTABLE_SORT_RENDERER_VERSION: &str = "rustgrad-wgsl-portable-sort-v1";
pub const WGSL_PORTABLE_THREEFRY_RENDERER_VERSION: &str = "rustgrad-wgsl-portable-threefry-v1";
/// Ordered storage-plus-extent bind-group ABI version.
pub const WEBGPU_ABI_VERSION: u32 = 3;
/// Guarded candidate/status ABI version included in source and cache identity.
pub const WEBGPU_STATUS_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// One ordered storage-buffer entry in the WGSL bind-group ABI.
pub struct WgslBufferAbi {
    /// Stable scheduled buffer identity.
    pub id: u64,
    /// Exact logical storage dtype.
    pub dtype: DType,
    /// Physical source-storage shape.
    pub source_shape: Shape,
    /// Logical source-storage element count.
    pub elements: usize,
    /// Whether this is an output binding. Mutable entries are ordered by
    /// their scheduled output ordinal.
    pub mutable: bool,
    /// Optional source-backed affine logical mapping.
    pub view: Option<AffineView>,
}

impl WgslBufferAbi {
    /// Logical RustGrad bytes. Native allocation rounds this value to four.
    pub fn logical_bytes(&self) -> Result<usize, WebGpuError> {
        self.elements
            .checked_mul(self.dtype.itemsize())
            .ok_or(WebGpuError::Overflow)
    }

    /// Public ABI storage size after WebGPU's required four-byte rounding.
    /// Prepared prefixes may privately back a logical zero with one word when
    /// a nonempty launch still requires a native binding.
    pub fn physical_bytes(&self) -> Result<usize, WebGpuError> {
        let logical = self.logical_bytes()?;
        Ok(logical.checked_add(3).ok_or(WebGpuError::Overflow)? / 4 * 4)
    }
}

#[derive(Clone, Debug)]
/// Immutable WGSL source plus its complete checked launch contract.
pub struct RenderedWgsl {
    /// Deterministically emitted WGSL source.
    pub source: String,
    /// Expression IDs to one-based source lines.
    pub source_map: BTreeMap<usize, usize>,
    /// Ordered inputs followed by the ordered output bindings.
    pub buffers: Vec<WgslBufferAbi>,
    /// Exact launch work-item count supplied through the final uniform.
    /// This equals the logical output extent for ordinary kernels; PrefixScan
    /// and coupled Sort launch per independent lane, while Bitcast launches
    /// per raw byte.
    pub extent: usize,
    /// Generated compute entry point.
    pub entry: String,
    /// Content-addressed renderer/capability/ABI identity.
    pub cache_key: String,
    /// Exact adapter capabilities used for rendering.
    pub capabilities: WebGpuCapabilities,
    /// Checked workgroup width encoded in source and launch metadata.
    pub local_size: u32,
    /// Guard/status metadata when output must commit transactionally.
    pub transaction: Option<WebGpuTransactionAbi>,
    pub(super) schedule_inputs: Vec<WgslBufferAbi>,
    pub(super) semantic_program: Arc<super::dispatch::KernelSemanticProgram>,
}

impl RenderedWgsl {
    /// Validates schedule-owned first-load ordering against the bind-group ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), WebGpuError> {
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Matmul(value) = root.operation()
        {
            crate::matmul::PortableF32Matmul::new(value)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::PrefixScan(value) = root.operation()
        {
            let portable = crate::prefix_scan_native::PortablePrefixScan::new(value)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
            crate::runtime::static_schedule::validate_portable_prefix_scan_bindings(
                &portable, bindings,
            )
            .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Sort(value) = root.operation()
        {
            crate::portable_sort::PortableSortPair::new(value)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Threefry(value) = root.operation()
        {
            crate::portable_threefry::PortableThreefry::new(value)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
                .validate_schedule_bindings(bindings)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Movement(MovementValue::Plan(plan)) = root.operation()
            && matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. })
        {
            crate::movement_plan::PortableBitcast::new(plan)
                .and_then(|portable| portable.validate_schedule_bindings(bindings))
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        }
        if let super::dispatch::KernelSemanticProgram::UOp(root) = self.semantic_program.as_ref()
            && let Operation::Movement(MovementValue::Plan(plan)) = root.operation()
            && matches!(
                &plan.kind,
                crate::MovementKernelKind::Pad { .. } | crate::MovementKernelKind::Concat { .. }
            )
        {
            crate::movement_plan::PortableDenseMaterialization::new(plan)
                .and_then(|portable| portable.validate_schedule_bindings(bindings))
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        }
        if bindings.len() != self.schedule_inputs.len() {
            return Err(WebGpuError::InvalidBinding(
                "schedule/WebGPU input count mismatch".into(),
            ));
        }
        for (position, (binding, expected)) in
            bindings.iter().zip(&self.schedule_inputs).enumerate()
        {
            if binding.abi_index != position
                || binding.desc.id != expected.id
                || binding.desc.dtype != expected.dtype
                || binding.desc.shape != expected.source_shape
                || binding.desc.view != expected.view
                || binding.desc.bytes != expected.logical_bytes()?
            {
                return Err(WebGpuError::InvalidBinding(format!(
                    "schedule binding {position} mismatches WebGPU ABI"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn validate_artifact(&self) -> Result<(), WebGpuError> {
        let portable_matmul = match self.semantic_program.as_ref() {
            super::dispatch::KernelSemanticProgram::UOp(root)
                if matches!(root.operation(), Operation::Matmul(_)) =>
            {
                let Operation::Matmul(value) = root.operation() else {
                    unreachable!("guarded above")
                };
                Some(
                    crate::matmul::PortableF32Matmul::new(value).map_err(|error| match error {
                        crate::matmul::PortableF32MatmulError::Unsupported(reason) => {
                            WebGpuError::Unsupported(reason.into())
                        }
                        crate::matmul::PortableF32MatmulError::Overflow => WebGpuError::Overflow,
                        other => WebGpuError::InvalidBinding(other.to_string()),
                    })?,
                )
            }
            _ => None,
        };
        let portable_scan = match self.semantic_program.as_ref() {
            super::dispatch::KernelSemanticProgram::UOp(root)
                if matches!(root.operation(), Operation::PrefixScan(_)) =>
            {
                let Operation::PrefixScan(value) = root.operation() else {
                    unreachable!("guarded above")
                };
                Some(
                    crate::prefix_scan_native::PortablePrefixScan::new(value).map_err(|error| {
                        match error {
                            crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                                reason,
                            ) => WebGpuError::Unsupported(reason.into()),
                            crate::prefix_scan_native::PortablePrefixScanError::Overflow => {
                                WebGpuError::Overflow
                            }
                            other => WebGpuError::InvalidBinding(other.to_string()),
                        }
                    })?,
                )
            }
            _ => None,
        };
        let portable_sort = match self.semantic_program.as_ref() {
            super::dispatch::KernelSemanticProgram::UOp(root)
                if matches!(root.operation(), Operation::Sort(_)) =>
            {
                let Operation::Sort(value) = root.operation() else {
                    unreachable!("guarded above")
                };
                Some(crate::portable_sort::PortableSortPair::new(value).map_err(
                    |error| match error {
                        crate::portable_sort::PortableSortError::Unsupported(reason) => {
                            WebGpuError::Unsupported(reason.into())
                        }
                        crate::portable_sort::PortableSortError::Overflow => WebGpuError::Overflow,
                        other => WebGpuError::InvalidBinding(other.to_string()),
                    },
                )?)
            }
            _ => None,
        };
        let portable_threefry = match self.semantic_program.as_ref() {
            super::dispatch::KernelSemanticProgram::UOp(root)
                if matches!(root.operation(), Operation::Threefry(_)) =>
            {
                let Operation::Threefry(value) = root.operation() else {
                    unreachable!("guarded above")
                };
                Some(
                    crate::portable_threefry::PortableThreefry::new(value).map_err(|error| {
                        match error {
                            crate::portable_threefry::PortableThreefryError::Unsupported(
                                reason,
                            ) => WebGpuError::Unsupported(reason.into()),
                            crate::portable_threefry::PortableThreefryError::Overflow => {
                                WebGpuError::Overflow
                            }
                            other => WebGpuError::InvalidBinding(other.to_string()),
                        }
                    })?,
                )
            }
            _ => None,
        };
        if let Some(portable) = portable_scan.as_ref() {
            validate_portable_serial_launch(
                portable.launch_extent(),
                self.local_size,
                &self.capabilities,
            )?;
        }
        if let Some(portable) = portable_sort.as_ref() {
            validate_portable_serial_launch(
                portable.launch_extent(),
                self.local_size,
                &self.capabilities,
            )?;
        }
        if let Some(portable) = portable_threefry.as_ref() {
            validate_portable_serial_launch(
                portable.elements(),
                self.local_size,
                &self.capabilities,
            )?;
        }
        let raw_movement = match self.semantic_program.as_ref() {
            super::dispatch::KernelSemanticProgram::UOp(root)
                if matches!(root.operation(), Operation::Movement(_)) =>
            {
                let Operation::Movement(MovementValue::Plan(plan)) = root.operation() else {
                    return Err(WebGpuError::Unsupported(
                        "quantized movement is outside WGSL contiguous-copy lowering".into(),
                    ));
                };
                let raw_copy = plan
                    .raw_copy()
                    .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
                    .is_some();
                let static_position = plan
                    .static_position_write()
                    .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
                    .is_some();
                let bitcast = if matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. }) {
                    crate::movement_plan::PortableBitcast::new(plan)
                        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
                    true
                } else {
                    false
                };
                let dense_materialization = if matches!(
                    &plan.kind,
                    crate::MovementKernelKind::Pad { .. }
                        | crate::MovementKernelKind::Concat { .. }
                ) {
                    crate::movement_plan::PortableDenseMaterialization::new(plan)
                        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
                    true
                } else {
                    false
                };
                if !raw_copy && !static_position && !bitcast && !dense_materialization {
                    return Err(WebGpuError::Unsupported(
                        "movement plan is outside raw WGSL lowering".into(),
                    ));
                }
                Some(plan.as_ref())
            }
            _ => None,
        };
        let portable_bitcast = raw_movement
            .filter(|plan| matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. }))
            .map(crate::movement_plan::PortableBitcast::new)
            .transpose()
            .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        let portable_dense = raw_movement
            .filter(|plan| {
                matches!(
                    &plan.kind,
                    crate::MovementKernelKind::Pad { .. }
                        | crate::MovementKernelKind::Concat { .. }
                )
            })
            .map(crate::movement_plan::PortableDenseMaterialization::new)
            .transpose()
            .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        if let Some(portable) = portable_bitcast.as_ref() {
            validate_portable_serial_launch(portable.bytes(), self.local_size, &self.capabilities)?;
        }
        if let Some(portable) = portable_dense.as_ref() {
            validate_portable_serial_launch(
                portable.elements(),
                self.local_size,
                &self.capabilities,
            )?;
        }
        let expected_mutable = if portable_sort.is_some() { 2 } else { 1 };
        if self.buffers.is_empty()
            || self.buffers.iter().filter(|buffer| buffer.mutable).count() != expected_mutable
            || self.buffers[self.buffers.len() - expected_mutable..]
                .iter()
                .any(|buffer| !buffer.mutable)
            || self.buffers[..self.buffers.len() - expected_mutable]
                .iter()
                .any(|buffer| buffer.mutable)
        {
            return Err(WebGpuError::InvalidBinding(
                "artifact mutable output ABI mismatch".into(),
            ));
        }
        if self.buffers.len() > self.capabilities.max_storage_buffers_per_shader_stage as usize
            || self.local_size == 0
            || self.local_size > self.capabilities.max_compute_workgroup_size_x
            || self.extent > u32::MAX as usize
        {
            return Err(WebGpuError::InvalidBinding(
                "artifact capability or indexing metadata mismatch".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for buffer in &self.buffers {
            if portable_threefry.is_some() {
                if buffer.dtype != DType::U64
                    || buffer.view.is_some()
                    || buffer.elements > u32::MAX as usize / 2 + 1
                {
                    return Err(WebGpuError::InvalidBinding(
                        "portable Threefry requires addressable dense U64 storage".into(),
                    ));
                }
            } else if raw_movement.is_none() {
                supported_storage(buffer.dtype)?;
            } else if !matches!(buffer.dtype.itemsize(), 1 | 2 | 4 | 8) {
                return Err(WebGpuError::Unsupported(
                    "WGSL raw copy requires a concrete storage width".into(),
                ));
            }
            let source_elements = buffer
                .source_shape
                .numel()
                .map_err(|_| WebGpuError::Overflow)?;
            let physical_bytes = if (portable_matmul.is_some()
                || portable_scan.is_some()
                || portable_sort.is_some())
                && self.extent != 0
                && buffer.elements == 0
            {
                DType::F32.itemsize()
            } else {
                buffer.physical_bytes()?
            };
            if source_elements != buffer.elements
                || !ids.insert(buffer.id)
                || physical_bytes > self.capabilities.max_buffer_size
            {
                return Err(WebGpuError::InvalidBinding(
                    "artifact buffer storage metadata mismatch".into(),
                ));
            }
            if let Some(view) = &buffer.view {
                let access = WgslViewAccess::new(view)?;
                if access.source_shape != buffer.source_shape {
                    return Err(WebGpuError::InvalidBinding(
                        "artifact affine source shape mismatch".into(),
                    ));
                }
            }
        }
        if let Some(portable) = portable_threefry.as_ref() {
            if self.buffers.len() != portable.inputs().len() + 1 {
                return Err(WebGpuError::InvalidBinding(
                    "portable Threefry artifact buffer count mismatch".into(),
                ));
            }
            for (buffer, input) in self.buffers.iter().zip(portable.inputs()) {
                if buffer.id != input.node.index() as u64
                    || buffer.source_shape != input.shape
                    || buffer.elements != input.elements
                    || buffer.mutable
                {
                    return Err(WebGpuError::InvalidBinding(
                        "portable Threefry artifact input ABI mismatch".into(),
                    ));
                }
            }
            let output = self.buffers.last().expect("nonempty checked above");
            if output.id != portable.value().output.index() as u64
                || output.source_shape != portable.value().output_shape
                || output.elements != portable.elements()
                || !output.mutable
            {
                return Err(WebGpuError::InvalidBinding(
                    "portable Threefry artifact output ABI mismatch".into(),
                ));
            }
        }
        let expected_extent = portable_scan
            .as_ref()
            .map(|scan| scan.launch_extent())
            .or_else(|| portable_sort.as_ref().map(|sort| sort.launch_extent()))
            .or_else(|| portable_threefry.as_ref().map(|plan| plan.elements()))
            .or_else(|| portable_bitcast.as_ref().map(|plan| plan.bytes()))
            .unwrap_or_else(|| {
                self.buffers
                    .last()
                    .expect("nonempty checked above")
                    .elements
            });
        if self.extent != expected_extent {
            return Err(WebGpuError::InvalidBinding(
                "artifact output extent mismatch".into(),
            ));
        }
        if let Some(portable) = portable_scan {
            let plan = portable.plan();
            let [input, output] = self.buffers.as_slice() else {
                return Err(WebGpuError::InvalidBinding(
                    "WGSL prefix scan requires two buffers".into(),
                ));
            };
            if self.transaction.is_some()
                || self.schedule_inputs.len() != 1
                || self.schedule_inputs.first() != Some(input)
                || input.id != plan.input
                || input.dtype != plan.input_dtype
                || input.source_shape != portable.value().input_shape
                || input.elements != plan.elements
                || input.mutable
                || input.view.is_some()
                || output.id != plan.output
                || output.dtype != plan.output_dtype
                || output.source_shape != portable.value().output_shape
                || output.elements != plan.elements
                || !output.mutable
                || output.view.is_some()
            {
                return Err(WebGpuError::InvalidBinding(
                    "WGSL prefix-scan artifact disagrees with its plan".into(),
                ));
            }
            return Ok(());
        }
        if let Some(portable) = portable_sort {
            let value = portable.value();
            let [input, values, indices] = self.buffers.as_slice() else {
                return Err(WebGpuError::InvalidBinding(
                    "WGSL portable sort requires three buffers".into(),
                ));
            };
            if self.transaction.is_some()
                || self.schedule_inputs.len() != 1
                || self.schedule_inputs.first() != Some(input)
                || input.id != value.input.index() as u64
                || input.dtype != value.dtype
                || input.source_shape != value.input_shape
                || input.elements != portable.elements()
                || input.mutable
                || input.view.is_some()
                || values.id != value.values.index() as u64
                || values.dtype != value.dtype
                || values.source_shape != value.input_shape
                || values.elements != portable.elements()
                || !values.mutable
                || values.view.is_some()
                || indices.id != value.indices.index() as u64
                || indices.dtype != DType::I32
                || indices.source_shape != value.input_shape
                || indices.elements != portable.elements()
                || !indices.mutable
                || indices.view.is_some()
            {
                return Err(WebGpuError::InvalidBinding(
                    "WGSL portable-sort artifact disagrees with its plan".into(),
                ));
            }
            return Ok(());
        }
        if let Some(portable) = portable_matmul {
            let plan = portable.plan();
            let inputs = portable.inputs();
            if self.transaction.is_some()
                || self.schedule_inputs.len() != inputs.len()
                || self.buffers.len() != inputs.len() + 1
            {
                return Err(WebGpuError::InvalidBinding(
                    "WGSL matmul pointer count mismatch".into(),
                ));
            }
            for (position, input) in inputs.iter().enumerate() {
                let elements = input.shape.numel().map_err(|_| WebGpuError::Overflow)?;
                let abi = &self.buffers[position];
                if abi.id != input.node.index() as u64
                    || abi.dtype != DType::F32
                    || abi.source_shape != *input.shape
                    || abi.elements != elements
                    || abi.mutable
                    || abi.view.is_some()
                    || &self.schedule_inputs[position] != abi
                {
                    return Err(WebGpuError::InvalidBinding(
                        "WGSL matmul input ABI mismatch".into(),
                    ));
                }
            }
            let output = self.buffers.last().expect("checked nonempty above");
            if output.id != plan.output.index() as u64
                || output.dtype != DType::F32
                || output.source_shape != plan.output_shape
                || output.elements != portable.extent()
                || !output.mutable
                || output.view.is_some()
            {
                return Err(WebGpuError::InvalidBinding(
                    "WGSL matmul output ABI mismatch".into(),
                ));
            }
            return Ok(());
        }
        if let Some(plan) = raw_movement {
            if let Some(portable) = portable_dense {
                if self.transaction.is_some()
                    || self.extent != portable.elements()
                    || self.schedule_inputs.len() != portable.inputs().len()
                    || self.buffers.len() != portable.inputs().len() + 1
                {
                    return Err(WebGpuError::InvalidBinding(
                        "portable dense materialization ABI mismatch".into(),
                    ));
                }
                for (position, input) in portable.inputs().iter().enumerate() {
                    let abi = &self.buffers[position];
                    let elements = input.shape.numel().map_err(|_| WebGpuError::Overflow)?;
                    if abi.id != input.node.index() as u64
                        || abi.dtype != input.dtype
                        || abi.source_shape != input.shape
                        || abi.elements != elements
                        || abi.mutable
                        || abi.view.is_some()
                        || &self.schedule_inputs[position] != abi
                    {
                        return Err(WebGpuError::InvalidBinding(
                            "portable dense materialization input ABI mismatch".into(),
                        ));
                    }
                }
                let output = self.buffers.last().expect("checked nonempty above");
                if output.id != plan.output.index() as u64
                    || output.dtype != plan.dtype
                    || output.source_shape != plan.output_shape
                    || output.elements != portable.elements()
                    || !output.mutable
                    || output.view.is_some()
                {
                    return Err(WebGpuError::InvalidBinding(
                        "portable dense materialization output ABI mismatch".into(),
                    ));
                }
                return Ok(());
            }
            let inputs = plan.input_operands();
            let [input] = inputs.as_slice() else {
                return Err(WebGpuError::InvalidBinding(
                    "raw WGSL movement requires one input".into(),
                ));
            };
            let input_elements = input.shape.numel().map_err(|_| WebGpuError::Overflow)?;
            let output_elements = plan
                .output_shape
                .numel()
                .map_err(|_| WebGpuError::Overflow)?;
            let expected_extent = if matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. })
            {
                crate::movement_plan::PortableBitcast::new(plan)
                    .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
                    .bytes()
            } else {
                output_elements
            };
            if self.buffers.len() != 2
                || self.transaction.is_some()
                || self.extent != expected_extent
                || self.buffers[0].id != input.node.index() as u64
                || self.buffers[0].dtype != input.dtype
                || self.buffers[0].source_shape != input.shape
                || self.buffers[0].elements != input_elements
                || self.buffers[0].mutable
                || self.buffers[0].view.is_some()
                || self.buffers[1].id != plan.output.index() as u64
                || self.buffers[1].dtype != plan.dtype
                || self.buffers[1].source_shape != plan.output_shape
                || self.buffers[1].elements != output_elements
                || !self.buffers[1].mutable
                || self.buffers[1].view.is_some()
            {
                return Err(WebGpuError::InvalidBinding(
                    "WGSL raw-movement artifact disagrees with its plan".into(),
                ));
            }
            return Ok(());
        }
        let Some(transaction) = &self.transaction else {
            return Ok(());
        };
        if transaction.output_abi_index >= self.buffers.len()
            || !self.buffers[transaction.output_abi_index].mutable
            || self.buffers.len() + 1
                > self.capabilities.max_storage_buffers_per_shader_stage as usize
        {
            return Err(WebGpuError::InvalidBinding(
                "transaction artifact binding mismatch".into(),
            ));
        }
        transaction.validate_launch(self.extent, transaction.output_abi_index)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Pure renderer bound to one immutable adapter capability identity.
pub struct WgslRenderer {
    /// Checked X workgroup width.
    pub local_size: u32,
    /// Adapter capability identity included in output identity.
    pub capabilities: WebGpuCapabilities,
}

impl WgslRenderer {
    /// Creates a renderer after validating the static workgroup width.
    pub fn new(local_size: u32, capabilities: WebGpuCapabilities) -> Result<Self, WebGpuError> {
        if local_size == 0 {
            return Err(WebGpuError::InvalidArgument("zero local size"));
        }
        if local_size > capabilities.max_compute_workgroup_size_x {
            return Err(WebGpuError::InvalidArgument(
                "local size exceeds adapter workgroup limit",
            ));
        }
        Ok(Self {
            local_size,
            capabilities,
        })
    }

    /// Lowers a validated scheduled UOp without executing or allocating.
    pub fn render(&self, root: &UOp) -> Result<RenderedWgsl, WebGpuError> {
        if let Operation::Matmul(value) = root.operation() {
            return render_portable_f32_matmul(self, root, value);
        }
        if let Operation::PrefixScan(value) = root.operation() {
            return render_portable_prefix_scan(self, root, value);
        }
        if let Operation::Sort(value) = root.operation() {
            return render_portable_sort(self, root, value);
        }
        if let Operation::Threefry(value) = root.operation() {
            return render_portable_threefry(self, root, value);
        }
        if let Operation::Movement(value) = root.operation() {
            return match value {
                MovementValue::Plan(plan)
                    if matches!(
                        &plan.kind,
                        crate::MovementKernelKind::ScatterPositions { .. }
                    ) =>
                {
                    render_static_positions(self, root, plan)
                }
                MovementValue::Plan(plan)
                    if matches!(&plan.kind, crate::MovementKernelKind::Bitcast { .. }) =>
                {
                    render_portable_bitcast(self, root, plan)
                }
                MovementValue::Plan(plan)
                    if matches!(
                        &plan.kind,
                        crate::MovementKernelKind::Pad { .. }
                            | crate::MovementKernelKind::Concat { .. }
                    ) =>
                {
                    render_portable_dense_materialization(self, root, plan)
                }
                MovementValue::Plan(plan) => render_raw_copy(self, root, plan),
                MovementValue::QuantizedRowGather(_) => Err(WebGpuError::Unsupported(
                    "quantized movement is outside WGSL contiguous-copy lowering".into(),
                )),
            };
        }
        if let Operation::Random(plan) = root.operation() {
            return super::random::render(self, plan);
        }
        if matches!(root.operation(), Operation::TensorGuard(_)) {
            return Err(WebGpuError::Unsupported(
                "guards are outside WebGPU lowering".into(),
            ));
        }
        root.validate()
            .map_err(|error| WebGpuError::Unsupported(error.to_string()))?;
        let nodes = root
            .topological()
            .map_err(|error| WebGpuError::Unsupported(error.to_string()))?;
        if nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::Barrier | Operation::If | Operation::EndIf
            )
        }) {
            return Err(WebGpuError::Unsupported(
                "effects and control flow are outside the exact WGSL subset".into(),
            ));
        }
        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.operation(), Operation::Store))
            .ok_or_else(|| WebGpuError::Unsupported("sink has no store".into()))?;
        let output_index = store
            .sources()
            .first()
            .ok_or_else(|| WebGpuError::Unsupported("store has no index".into()))?;
        let Operation::Index(IndexValue::Buffer {
            buffer: output_id,
            elements: extent,
            input_shape: output_shape,
            output_shape: store_shape,
            addressing: crate::IndexAddressing::Broadcast,
        }) = output_index.operation()
        else {
            return Err(WebGpuError::Unsupported(
                "output requires a contiguous BufferIndex".into(),
            ));
        };
        if output_shape != store_shape {
            return Err(WebGpuError::Unsupported(
                "non-contiguous output addressing".into(),
            ));
        }
        if *extent > u32::MAX as usize {
            return Err(WebGpuError::Unsupported(
                "extent exceeds WGSL u32 indexing".into(),
            ));
        }
        let output_dtype = output_index
            .ty()
            .ok_or_else(|| WebGpuError::Unsupported("untyped output index".into()))?
            .scalar;
        supported_storage(output_dtype)?;

        let common_views = crate::schedule::common_buffer_views(&nodes);
        let mut inventory = BTreeMap::<u64, WgslBufferAbi>::new();
        for node in &nodes {
            let (buffer, source_shape, elements) = match node.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements,
                    input_shape,
                    ..
                }) => (*buffer, input_shape.clone(), *elements),
                Operation::Index(IndexValue::View { buffer, view, .. }) => {
                    let access = WgslViewAccess::new(view)?;
                    let elements = access
                        .source_shape
                        .numel()
                        .map_err(|_| WebGpuError::Overflow)?;
                    (*buffer, access.source_shape, elements)
                }
                _ => continue,
            };
            let dtype = node
                .ty()
                .ok_or_else(|| WebGpuError::Unsupported("untyped buffer index".into()))?
                .scalar;
            supported_storage(dtype)?;
            let abi = WgslBufferAbi {
                id: buffer,
                dtype,
                source_shape,
                elements,
                mutable: buffer == *output_id,
                view: common_views.get(&buffer).cloned().flatten(),
            };
            abi.logical_bytes()?;
            if let Some(previous) = inventory.insert(buffer, abi.clone())
                && previous != abi
            {
                return Err(WebGpuError::InvalidBinding(format!(
                    "buffer {buffer} has conflicting ABI metadata"
                )));
            }
        }

        let mut seen = BTreeSet::new();
        let mut schedule_inputs = Vec::new();
        for node in &nodes {
            if !matches!(node.operation(), Operation::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| WebGpuError::InvalidBinding("load lacks index".into()))?;
            let buffer = match index.operation() {
                Operation::Index(IndexValue::Buffer { buffer, .. })
                | Operation::Index(IndexValue::View { buffer, .. }) => *buffer,
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            if seen.insert(buffer) {
                schedule_inputs.push(
                    inventory
                        .get(&buffer)
                        .ok_or_else(|| WebGpuError::InvalidBinding("load ABI missing".into()))?
                        .clone(),
                );
            }
        }
        let mut buffers = schedule_inputs.clone();
        if seen.insert(*output_id) {
            buffers.push(
                inventory
                    .get(output_id)
                    .ok_or_else(|| WebGpuError::InvalidBinding("output ABI missing".into()))?
                    .clone(),
            );
        }
        if buffers.last().is_none_or(|buffer| buffer.id != *output_id) {
            return Err(WebGpuError::InvalidBinding(
                "output aliases an input buffer".into(),
            ));
        }
        if buffers.len() > self.capabilities.max_storage_buffers_per_shader_stage as usize {
            return Err(WebGpuError::Unsupported(
                "ordered bindings exceed adapter storage-buffer limit".into(),
            ));
        }
        for abi in &buffers {
            if abi.physical_bytes()? > self.capabilities.max_buffer_size {
                return Err(WebGpuError::Unsupported(
                    "binding exceeds adapter buffer limit".into(),
                ));
            }
        }

        let output_position = buffers.len() - 1;
        let ids = buffers
            .iter()
            .enumerate()
            .map(|(position, buffer)| (buffer.id, position))
            .collect::<BTreeMap<_, _>>();
        let value = store
            .sources()
            .get(1)
            .ok_or_else(|| WebGpuError::Unsupported("store has no value".into()))?;
        let reduction = crate::reduction_native::NativeReductionKernel::from_store(store)
            .map_err(|reason| WebGpuError::Unsupported(reason.into()))?;
        let reduction_roots = nodes
            .iter()
            .filter(|node| matches!(node.operation(), Operation::ReduceFinalize))
            .fold(Vec::<&UOp>::new(), |mut roots, node| {
                if !roots.iter().any(|root| node.shares_node_with(root)) {
                    roots.push(node);
                }
                roots
            })
            .len();
        if reduction_roots != usize::from(reduction.is_some()) {
            return Err(WebGpuError::Unsupported(
                "reduction must be the sole stored value".into(),
            ));
        }
        if let Some(reduction) = &reduction {
            let plan = &reduction.plan;
            for dtype in [plan.source_dtype, plan.accumulator_dtype, plan.output_dtype] {
                supported_storage(dtype)?;
            }
        }
        let transaction =
            WebGpuTransactionAbi::analyze(value, output_position, store_shape.clone())?;
        if transaction.is_some()
            && nodes
                .iter()
                .any(crate::projected_index::ProjectedIndexPlan::is_projected)
        {
            return Err(WebGpuError::Unsupported(
                "guarded projected indexing is outside the exact WGSL subset".into(),
            ));
        }
        if transaction.is_some()
            && buffers.len() + 1 > self.capabilities.max_storage_buffers_per_shader_stage as usize
        {
            return Err(WebGpuError::Unsupported(
                "transaction status exceeds adapter storage-buffer limit".into(),
            ));
        }
        let entry = format!("rg_webgpu_e{}_b{}", extent, buffers.len());
        let uses_narrow = nodes
            .iter()
            .any(|node| node.ty().is_some_and(|ty| narrow::is_narrow(ty.scalar)));
        let mut lines = vec![
            format!(
                "// {WGSL_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION} STATUS {WEBGPU_STATUS_VERSION} NARROW {WEBGPU_NARROW_ABI_VERSION}"
            ),
            "struct RustGradExtent { value: u32, };".into(),
            "fn rg_f32_to_i32(value: f32) -> i32 {".into(),
            "  if (isNan(value)) { return 0i; }".into(),
            "  if (value >= 2147483648.0) { return bitcast<i32>(0x7fffffffu); }".into(),
            "  if (value <= -2147483648.0) { return bitcast<i32>(0x80000000u); }".into(),
            "  return i32(value);".into(),
            "}".into(),
            "fn rg_f32_to_u32(value: f32) -> u32 {".into(),
            "  if (isNan(value) || value <= 0.0) { return 0u; }".into(),
            "  if (value >= 4294967296.0) { return 0xffffffffu; }".into(),
            "  return u32(value);".into(),
            "}".into(),
            "fn rg_i32_trunc_div(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return bitcast<i32>(0x80000000u); }".into(),
            "  return lhs / rhs;".into(),
            "}".into(),
            "fn rg_i32_fmod(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return 0i; }".into(),
            "  return lhs % rhs;".into(),
            "}".into(),
            "fn rg_i32_floor_div(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return bitcast<i32>(0x80000000u); }".into(),
            "  let quotient: i32 = lhs / rhs;".into(),
            "  let remainder: i32 = lhs % rhs;".into(),
            "  if (remainder < 0i) { return quotient - select(-1i, 1i, rhs > 0i); }".into(),
            "  return quotient;".into(),
            "}".into(),
            "fn rg_i32_mod(lhs: i32, rhs: i32) -> i32 {".into(),
            "  if (lhs == bitcast<i32>(0x80000000u) && rhs == -1i) { return 0i; }".into(),
            "  let remainder: i32 = lhs % rhs;".into(),
            "  if (remainder < 0i) {".into(),
            "    let magnitude: u32 = select(bitcast<u32>(rhs), 0u - bitcast<u32>(rhs), rhs < 0i);".into(),
            "    return bitcast<i32>(bitcast<u32>(remainder) + magnitude);".into(),
            "  }".into(),
            "  return remainder;".into(),
            "}".into(),
        ];
        if uses_narrow {
            lines.push(narrow::SOURCE.into());
        }
        for (position, buffer) in buffers.iter().enumerate() {
            let access = if buffer.mutable { "read_write" } else { "read" };
            let storage = wgsl_storage_decl(buffer.dtype, buffer.mutable);
            lines.push(format!(
                "@group(0) @binding({position}) var<storage, {access}> b{position}: array<{storage}>;"
            ));
        }
        lines.push(format!(
            "@group(0) @binding({}) var<uniform> rg_extent: RustGradExtent;",
            buffers.len()
        ));
        if transaction.is_some() {
            lines.push("struct RustGradStatus { value: atomic<u32>, };".into());
            lines.push(format!(
                "@group(0) @binding({}) var<storage, read_write> rg_status: RustGradStatus;",
                buffers.len() + 1
            ));
        }
        lines.push(format!(
            "@compute @workgroup_size({}, 1, 1)",
            self.local_size
        ));
        lines.push(format!(
            "fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"
        ));
        lines.push("  let gid: u32 = rg_global.x;".into());
        lines.push("  if (gid >= rg_extent.value) { return; }".into());
        let mut source_map = BTreeMap::new();
        let raw_predicated_narrow =
            if reduction.is_none() && transaction.is_none() && narrow::is_narrow(output_dtype) {
                emit_raw_predicated_narrow_load(value, &ids, &mut source_map, &mut lines, "gid")?
            } else {
                None
            };
        let preserves_raw_narrow = raw_predicated_narrow.is_some();
        let expression = if let Some(expression) = raw_predicated_narrow {
            expression
        } else if let Some(reduction) = &reduction {
            if transaction.is_some() {
                return Err(WebGpuError::Unsupported(
                    "guarded reduction producers are outside the exact WGSL subset".into(),
                ));
            }
            emit_wgsl_reduction(reduction, &ids, &mut source_map, &mut lines)?
        } else if let Some(transaction) = &transaction {
            emit_transactional(value, transaction, &ids, &mut source_map, &mut lines)?
        } else {
            emit_expr(value, &ids, &mut source_map, &mut lines, "gid")?
        };
        if output_dtype == DType::Bool {
            if transaction.is_some() {
                lines.push("  if (rg_ok) {".into());
            }
            let indent = if transaction.is_some() { "    " } else { "  " };
            lines.push(format!("{indent}let rg_shift: u32 = (gid & 3u) * 8u;"));
            lines.push(format!(
                "{indent}atomicAnd(&b{output_position}[gid >> 2u], ~(0xffu << rg_shift));"
            ));
            lines.push(format!(
                "{indent}atomicOr(&b{output_position}[gid >> 2u], select(0u, 1u, {expression}) << rg_shift);"
            ));
            if transaction.is_some() {
                lines.push("  }".into());
            }
        } else if narrow::is_narrow(output_dtype) {
            if transaction.is_some() {
                return Err(WebGpuError::Unsupported(
                    "guarded narrow-float output is outside the exact WGSL subset".into(),
                ));
            }
            let encoded = if preserves_raw_narrow {
                expression
            } else {
                narrow::encode(output_dtype, &expression).expect("validated narrow output dtype")
            };
            lines.push("  let rg_shift: u32 = (gid & 1u) * 16u;".into());
            lines.push(format!(
                "  atomicAnd(&b{output_position}[gid >> 1u], ~(0xffffu << rg_shift));"
            ));
            lines.push(format!(
                "  atomicOr(&b{output_position}[gid >> 1u], ({encoded} & 0xffffu) << rg_shift);"
            ));
        } else {
            lines.push(if transaction.is_some() {
                format!("  if (rg_ok) {{ b{output_position}[gid] = {expression}; }}")
            } else {
                format!("  b{output_position}[gid] = {expression};")
            });
        }
        lines.push("}".into());
        let source = lines.join("\n") + "\n";
        let cache_key = stable_key(&(
            WGSL_RENDERER_VERSION,
            WEBGPU_ABI_VERSION,
            WEBGPU_STATUS_VERSION,
            WEBGPU_NARROW_ABI_VERSION,
            self.local_size,
            &self.capabilities,
            &source,
            &buffers,
            &schedule_inputs,
            &transaction,
        ));
        Ok(RenderedWgsl {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            capabilities: self.capabilities.clone(),
            local_size: self.local_size,
            transaction,
            schedule_inputs,
            semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
                root.clone(),
            ))),
        })
    }
}

fn validate_portable_serial_launch(
    extent: usize,
    local_size: u32,
    capabilities: &WebGpuCapabilities,
) -> Result<(), WebGpuError> {
    if local_size == 0 {
        return Err(WebGpuError::InvalidBinding(
            "zero WGSL prefix-scan workgroup size".into(),
        ));
    }
    let extent = u32::try_from(extent).map_err(|_| WebGpuError::Overflow)?;
    if extent.div_ceil(local_size) > capabilities.max_compute_workgroups_per_dimension {
        return Err(WebGpuError::Unsupported(
            "portable launch exceeds adapter workgroup-count limit".into(),
        ));
    }
    Ok(())
}

fn render_portable_threefry(
    renderer: &WgslRenderer,
    root: &UOp,
    value: &crate::ThreefryValue,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::portable_threefry::PortableThreefry::new(value).map_err(|error| match error {
            crate::portable_threefry::PortableThreefryError::Unsupported(reason) => {
                WebGpuError::Unsupported(reason.into())
            }
            crate::portable_threefry::PortableThreefryError::Overflow => WebGpuError::Overflow,
            other => WebGpuError::InvalidBinding(other.to_string()),
        })?;
    validate_portable_serial_launch(
        portable.elements(),
        renderer.local_size,
        &renderer.capabilities,
    )?;
    let mut buffers = portable
        .inputs()
        .iter()
        .map(|input| WgslBufferAbi {
            id: input.node.index() as u64,
            dtype: DType::U64,
            source_shape: input.shape.clone(),
            elements: input.elements,
            mutable: false,
            view: None,
        })
        .collect::<Vec<_>>();
    let schedule_inputs = buffers.clone();
    buffers.push(WgslBufferAbi {
        id: value.output.index() as u64,
        dtype: DType::U64,
        source_shape: value.output_shape.clone(),
        elements: portable.elements(),
        mutable: true,
        view: None,
    });
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "portable Threefry bindings exceed adapter limit".into(),
        ));
    }
    for buffer in &buffers {
        if buffer.elements > u32::MAX as usize / 2 + 1 {
            return Err(WebGpuError::Unsupported(
                "portable Threefry exceeds packed WGSL word indexing".into(),
            ));
        }
        if buffer.physical_bytes()? > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "portable Threefry binding exceeds adapter buffer limit".into(),
            ));
        }
    }
    let entry = format!("rg_wgsl_threefry_e{}", portable.elements());
    let mut lines = vec![format!(
        "// {WGSL_PORTABLE_THREEFRY_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"
    )];
    for (index, buffer) in buffers.iter().enumerate() {
        let access = if buffer.mutable { "read_write" } else { "read" };
        lines.push(format!(
            "@group(0) @binding({index}) var<storage, {access}> b{index}: array<u32>;"
        ));
    }
    lines.extend([
        "struct RustGradExtent { value: u32, };".into(),
        format!(
            "@group(0) @binding({}) var<uniform> rg_extent: RustGradExtent;",
            buffers.len()
        ),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
    ]);
    lines.extend(crate::portable_threefry::emit_portable_threefry_body(
        &portable,
        &crate::portable_threefry::WgslPortableThreefryDialect,
    ));
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        WGSL_PORTABLE_THREEFRY_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.elements(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

fn render_portable_sort(
    renderer: &WgslRenderer,
    root: &UOp,
    value: &crate::SortValue,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::portable_sort::PortableSortPair::new(value).map_err(|error| match error {
            crate::portable_sort::PortableSortError::Unsupported(reason) => {
                WebGpuError::Unsupported(reason.into())
            }
            crate::portable_sort::PortableSortError::Overflow => WebGpuError::Overflow,
            other => WebGpuError::InvalidBinding(other.to_string()),
        })?;
    validate_portable_serial_launch(
        portable.launch_extent(),
        renderer.local_size,
        &renderer.capabilities,
    )?;
    let elements = portable.elements();
    let input = WgslBufferAbi {
        id: value.input.index() as u64,
        dtype: value.dtype,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: false,
        view: None,
    };
    let values = WgslBufferAbi {
        id: value.values.index() as u64,
        dtype: value.dtype,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: true,
        view: None,
    };
    let indices = WgslBufferAbi {
        id: value.indices.index() as u64,
        dtype: DType::I32,
        source_shape: value.input_shape.clone(),
        elements,
        mutable: true,
        view: None,
    };
    let buffers = vec![input.clone(), values, indices];
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "portable sort bindings exceed adapter limit".into(),
        ));
    }
    for buffer in &buffers {
        if buffer.physical_bytes()? > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "portable sort binding exceeds adapter buffer limit".into(),
            ));
        }
    }
    let entry = format!(
        "rg_wgsl_sort_{:?}_a{}_n{}",
        value.dtype,
        value.axis,
        portable.elements()
    )
    .to_ascii_lowercase();
    let input_storage = wgsl_storage_decl(value.dtype, false);
    let output_storage = wgsl_storage_decl(value.dtype, true);
    let mut lines = vec![
        format!("// {WGSL_PORTABLE_SORT_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"),
        format!("@group(0) @binding(0) var<storage, read> b0: array<{input_storage}>;"),
        format!("@group(0) @binding(1) var<storage, read_write> b1: array<{output_storage}>;"),
        "@group(0) @binding(2) var<storage, read_write> b2: array<i32>;".into(),
        "struct RustGradExtent { value: u32, };".into(),
        "@group(0) @binding(3) var<uniform> rg_extent: RustGradExtent;".into(),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
    ];
    lines.extend(
        crate::portable_sort::emit_portable_sort_body(
            &portable,
            &WgslPortableSortDialect { dtype: value.dtype },
        )
        .map_err(|error| WebGpuError::Unsupported(error.to_string()))?,
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        WGSL_PORTABLE_SORT_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.launch_extent(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs: vec![input],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

struct WgslPortableSortDialect {
    dtype: DType,
}

impl WgslPortableSortDialect {
    fn value_type(&self) -> &'static str {
        match self.dtype {
            DType::Bool => "bool",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::F32 => "f32",
            _ => unreachable!("portable sort validated storage"),
        }
    }

    fn padding(&self, descending: bool) -> &'static str {
        match (self.dtype, descending) {
            (DType::Bool, true) => "false",
            (DType::Bool, false) => "true",
            (DType::I32, true) => "bitcast<i32>(0x80000000u)",
            (DType::I32, false) => "bitcast<i32>(0x7fffffffu)",
            (DType::U32, true) => "0u",
            (DType::U32, false) => "0xffffffffu",
            (DType::F32, true) => "bitcast<f32>(0xff800000u)",
            (DType::F32, false) => "bitcast<f32>(0x7f800000u)",
            _ => unreachable!("portable sort validated storage"),
        }
    }
}

impl crate::portable_sort::PortableSortDialect for WgslPortableSortDialect {
    fn domain(&self, plan: &crate::portable_sort::PortableSortPair<'_>) -> Vec<String> {
        vec![
            format!("  let rg_row: u32 = gid / {}u;", plan.inner()),
            format!("  let rg_inner: u32 = gid % {}u;", plan.inner()),
        ]
    }

    fn storage(
        &self,
        plan: &crate::portable_sort::PortableSortPair<'_>,
    ) -> Result<Vec<String>, crate::portable_sort::PortableSortError> {
        let ty = self.value_type();
        Ok(vec![
            format!(
                "  var rg_original: array<{ty}, {}>;",
                plan.axis_len().max(1)
            ),
            format!("  var rg_work: array<{ty}, {}>;", plan.padded_len().max(1)),
            format!(
                "  var rg_original_count: array<i32, {}>;",
                plan.axis_len().max(1)
            ),
            format!(
                "  var rg_sorted_count: array<i32, {}>;",
                plan.axis_len().max(1)
            ),
        ])
    }

    fn load_original(
        &self,
        plan: &crate::portable_sort::PortableSortPair<'_>,
        lane: usize,
    ) -> Result<Vec<String>, crate::portable_sort::PortableSortError> {
        let offset = format!(
            "((rg_row * {}u + {lane}u) * {}u + rg_inner)",
            plan.axis_len(),
            plan.inner()
        );
        let load = if self.dtype == DType::Bool {
            format!("(((b0[{offset} >> 2u] >> (({offset} & 3u) * 8u)) & 0xffu) != 0u)")
        } else {
            format!("b0[{offset}]")
        };
        Ok(vec![
            format!("  rg_original[{lane}] = {load};"),
            format!("  rg_work[{lane}] = rg_original[{lane}];"),
        ])
    }

    fn pad_work(
        &self,
        plan: &crate::portable_sort::PortableSortPair<'_>,
        lane: usize,
    ) -> Result<String, crate::portable_sort::PortableSortError> {
        Ok(format!(
            "  rg_work[{lane}] = {};",
            self.padding(plan.value().descending)
        ))
    }

    fn swap(
        &self,
        _plan: &crate::portable_sort::PortableSortPair<'_>,
        left: usize,
        right: usize,
    ) -> Vec<String> {
        vec![
            "  {".into(),
            format!("    let rg_swap: {} = rg_work[{left}];", self.value_type()),
            format!("    rg_work[{left}] = rg_work[{right}];"),
            format!("    rg_work[{right}] = rg_swap;"),
            "  }".into(),
        ]
    }

    fn compare(
        &self,
        _plan: &crate::portable_sort::PortableSortPair<'_>,
        step: crate::portable_sort::PortableSortCompare,
    ) -> Vec<String> {
        let left = step.left;
        let right = step.right;
        let (first, second) = if step.left_takes_larger {
            ("rg_larger", "rg_smaller")
        } else {
            ("rg_smaller", "rg_larger")
        };
        let (larger, smaller) = if self.dtype == DType::Bool {
            (
                "(rg_left || rg_right)".to_owned(),
                "(rg_left && rg_right)".to_owned(),
            )
        } else {
            (
                "select(rg_left, rg_right, rg_right > rg_left)".to_owned(),
                "select(rg_left, rg_right, rg_right < rg_left)".to_owned(),
            )
        };
        vec![
            "  {".into(),
            format!("    let rg_left: {} = rg_work[{left}];", self.value_type()),
            format!(
                "    let rg_right: {} = rg_work[{right}];",
                self.value_type()
            ),
            format!("    let rg_larger: {} = {larger};", self.value_type()),
            format!("    let rg_smaller: {} = {smaller};", self.value_type()),
            format!("    rg_work[{left}] = {first};"),
            format!("    rg_work[{right}] = {second};"),
            "  }".into(),
        ]
    }

    fn count_original_open(
        &self,
        plan: &crate::portable_sort::PortableSortPair<'_>,
    ) -> Vec<String> {
        vec![
            format!(
                "  for (var rg_i: u32 = 0u; rg_i < {}u; rg_i = rg_i + 1u) {{",
                plan.axis_len()
            ),
            "    var rg_count: i32 = 0i;".into(),
            "    for (var rg_j: u32 = 0u; rg_j <= rg_i; rg_j = rg_j + 1u) {".into(),
        ]
    }

    fn count_original_step(&self) -> String {
        "      if (rg_original[rg_j] == rg_original[rg_i]) { rg_count = rg_count + 1i; }".into()
    }

    fn count_original_close(&self) -> Vec<String> {
        vec![
            "    }".into(),
            "    rg_original_count[rg_i] = rg_count;".into(),
            "  }".into(),
        ]
    }

    fn count_sorted_open(&self, plan: &crate::portable_sort::PortableSortPair<'_>) -> Vec<String> {
        vec![
            format!(
                "  for (var rg_i: u32 = 0u; rg_i < {}u; rg_i = rg_i + 1u) {{",
                plan.axis_len()
            ),
            "    var rg_count: i32 = 0i;".into(),
            "    for (var rg_j: u32 = 0u; rg_j <= rg_i; rg_j = rg_j + 1u) {".into(),
        ]
    }

    fn count_sorted_step(&self) -> String {
        "      if (rg_work[rg_j] == rg_work[rg_i]) { rg_count = rg_count + 1i; }".into()
    }

    fn count_sorted_close(&self) -> Vec<String> {
        vec![
            "    }".into(),
            "    rg_sorted_count[rg_i] = rg_count;".into(),
            "  }".into(),
        ]
    }

    fn reconstruct_open(&self, plan: &crate::portable_sort::PortableSortPair<'_>) -> Vec<String> {
        vec![
            format!(
                "  for (var rg_out: u32 = 0u; rg_out < {}u; rg_out = rg_out + 1u) {{",
                plan.axis_len()
            ),
            "    var rg_index: i32 = 0i;".into(),
            format!(
                "    for (var rg_in: u32 = 0u; rg_in < {}u; rg_in = rg_in + 1u) {{",
                plan.axis_len()
            ),
        ]
    }

    fn reconstruct_step(&self) -> String {
        "      if (rg_original[rg_in] == rg_work[rg_out] && rg_original_count[rg_in] == rg_sorted_count[rg_out]) { rg_index = rg_index + i32(rg_in); }".into()
    }

    fn reconstruct_store(
        &self,
        plan: &crate::portable_sort::PortableSortPair<'_>,
    ) -> Result<Vec<String>, crate::portable_sort::PortableSortError> {
        let mut lines = vec![
            "    }".into(),
            format!(
                "    let rg_offset: u32 = (rg_row * {}u + rg_out) * {}u + rg_inner;",
                plan.axis_len(),
                plan.inner()
            ),
        ];
        if self.dtype == DType::Bool {
            lines.extend([
                "    let rg_shift: u32 = (rg_offset & 3u) * 8u;".into(),
                "    atomicAnd(&b1[rg_offset >> 2u], ~(0xffu << rg_shift));".into(),
                "    atomicOr(&b1[rg_offset >> 2u], select(0u, 1u, rg_work[rg_out]) << rg_shift);"
                    .into(),
            ]);
        } else {
            lines.push("    b1[rg_offset] = rg_work[rg_out];".into());
        }
        lines.push("    b2[rg_offset] = rg_index;".into());
        Ok(lines)
    }

    fn reconstruct_close(&self) -> Vec<String> {
        vec!["  }".into()]
    }
}

fn render_portable_prefix_scan(
    renderer: &WgslRenderer,
    root: &UOp,
    value: &crate::PrefixScanValue,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let portable =
        crate::prefix_scan_native::PortablePrefixScan::new(value).map_err(|error| match error {
            crate::prefix_scan_native::PortablePrefixScanError::Unsupported(reason) => {
                WebGpuError::Unsupported(reason.into())
            }
            crate::prefix_scan_native::PortablePrefixScanError::Overflow => WebGpuError::Overflow,
            other => WebGpuError::InvalidBinding(other.to_string()),
        })?;
    validate_portable_serial_launch(
        portable.launch_extent(),
        renderer.local_size,
        &renderer.capabilities,
    )?;
    let plan = portable.plan();
    let input = WgslBufferAbi {
        id: plan.input,
        dtype: plan.input_dtype,
        source_shape: value.input_shape.clone(),
        elements: plan.elements,
        mutable: false,
        view: None,
    };
    let output = WgslBufferAbi {
        id: plan.output,
        dtype: plan.output_dtype,
        source_shape: value.output_shape.clone(),
        elements: plan.elements,
        mutable: true,
        view: None,
    };
    let buffers = vec![input.clone(), output];
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "portable scan bindings exceed adapter limit".into(),
        ));
    }
    for buffer in &buffers {
        if buffer.physical_bytes()? > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "portable scan binding exceeds adapter buffer limit".into(),
            ));
        }
    }
    let entry = format!(
        "rg_wgsl_scan_{:?}_{:?}_a{}_n{}",
        plan.kind, plan.result, plan.axis, plan.elements
    )
    .to_ascii_lowercase();
    let input_storage = wgsl_storage_decl(plan.input_dtype, false);
    let output_storage = wgsl_storage_decl(plan.output_dtype, true);
    let mut lines = vec![
        format!("// {WGSL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"),
        format!("@group(0) @binding(0) var<storage, read> b0: array<{input_storage}>;"),
        format!("@group(0) @binding(1) var<storage, read_write> b1: array<{output_storage}>;"),
        "struct RustGradExtent { value: u32, };".into(),
        "@group(0) @binding(2) var<uniform> rg_extent: RustGradExtent;".into(),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
    ];
    lines.extend(
        crate::prefix_scan_native::emit_portable_prefix_scan_body(
            &portable,
            &WgslPrefixScanDialect,
        )
        .map_err(|error| WebGpuError::Unsupported(error.to_string()))?,
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        WGSL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.launch_extent(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs: vec![input],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

struct WgslPrefixScanDialect;

impl WgslPrefixScanDialect {
    fn work_type(
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<&'static str, crate::prefix_scan_native::PortablePrefixScanError> {
        match plan.work_dtype {
            DType::Bool => Ok("bool"),
            DType::I32 => Ok("i32"),
            DType::U32 => Ok("u32"),
            DType::F32 => Ok("f32"),
            _ => Err(
                crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                    "WGSL portable scan work dtype",
                ),
            ),
        }
    }

    fn bool_store(offset: &str, value: &str, indent: &str) -> Vec<String> {
        vec![
            format!("{indent}let rg_shift: u32 = ({offset} & 3u) * 8u;"),
            format!("{indent}atomicAnd(&b1[{offset} >> 2u], ~(0xffu << rg_shift));"),
            format!("{indent}atomicOr(&b1[{offset} >> 2u], select(0u, 1u, {value}) << rg_shift);"),
        ]
    }
}

impl crate::prefix_scan_native::PortablePrefixScanDialect for WgslPrefixScanDialect {
    fn scalar_body(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<Vec<String>, crate::prefix_scan_native::PortablePrefixScanError> {
        Ok(match plan.result {
            crate::PrefixScanOutput::Indices => vec!["  b1[0] = 0i;".into()],
            crate::PrefixScanOutput::Values if plan.output_dtype == DType::Bool => {
                Self::bool_store("0u", "((b0[0] & 0xffu) != 0u)", "  ")
            }
            crate::PrefixScanOutput::Values if plan.input_dtype == plan.output_dtype => {
                vec![if plan.input_dtype == DType::F32 {
                    "  b1[0] = bitcast<f32>(bitcast<u32>(b0[0]));".into()
                } else {
                    "  b1[0] = b0[0];".into()
                }]
            }
            crate::PrefixScanOutput::Values => {
                vec!["  b1[0] = select(0i, 1i, (b0[0] & 0xffu) != 0u);".into()]
            }
        })
    }

    fn domain(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> Vec<String> {
        vec![
            format!("  let rg_row: u32 = gid / {}u;", plan.inner),
            format!("  let rg_inner: u32 = gid % {}u;", plan.inner),
        ]
    }

    fn identity(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<&'static str, crate::prefix_scan_native::PortablePrefixScanError> {
        Ok(match (plan.kind, plan.work_dtype) {
            (crate::PrefixScanKind::Sum, DType::F32) => "0.0",
            (crate::PrefixScanKind::Product, DType::F32) => "1.0",
            (crate::PrefixScanKind::Max, DType::F32) => "bitcast<f32>(0xff800000u)",
            (crate::PrefixScanKind::Min, DType::F32) => "bitcast<f32>(0x7f800000u)",
            (crate::PrefixScanKind::Product | crate::PrefixScanKind::Min, DType::Bool) => "true",
            (crate::PrefixScanKind::Product, DType::I32) => "1i",
            (crate::PrefixScanKind::Product, DType::U32) => "1u",
            (crate::PrefixScanKind::Max, DType::I32) => "bitcast<i32>(0x80000000u)",
            (crate::PrefixScanKind::Min, DType::I32) => "bitcast<i32>(0x7fffffffu)",
            (crate::PrefixScanKind::Min, DType::U32) => "0xffffffffu",
            (_, DType::Bool) => "false",
            (_, DType::I32) => "0i",
            (_, DType::U32) => "0u",
            _ => {
                return Err(
                    crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                        "WGSL portable scan identity dtype",
                    ),
                );
            }
        })
    }

    fn accumulator(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
        identity: &str,
    ) -> String {
        format!(
            "  var rg_acc: {} = {identity};",
            Self::work_type(plan).expect("portable projection validated work dtype")
        )
    }

    fn index(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!("  var rg_index: i32 = {}i;", plan.index_sentinel)
    }

    fn loop_open(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!(
            "  for (var rg_axis: u32 = 0u; rg_axis < {}u; rg_axis = rg_axis + 1u) {{",
            plan.axis_len
        )
    }

    fn offset(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!(
            "    let rg_offset: u32 = (rg_row * {}u + rg_axis) * {}u + rg_inner;",
            plan.axis_len, plan.inner
        )
    }

    fn load(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
    ) -> Result<String, crate::prefix_scan_native::PortablePrefixScanError> {
        let work = Self::work_type(plan)?;
        let expression = if plan.input_dtype == DType::Bool {
            let boolean = "(((b0[rg_offset >> 2u] >> ((rg_offset & 3u) * 8u)) & 0xffu) != 0u)";
            if plan.work_dtype == DType::I32 {
                format!("select(0i, 1i, {boolean})")
            } else {
                boolean.into()
            }
        } else {
            "b0[rg_offset]".into()
        };
        Ok(format!("    let rg_next: {work} = {expression};"))
    }

    fn strict(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
        operator: &str,
    ) -> String {
        if plan.work_dtype == DType::Bool {
            let expression = if plan.kind == crate::PrefixScanKind::Max {
                "(rg_next && !rg_acc)"
            } else {
                "(!rg_next && rg_acc)"
            };
            format!("    let rg_strict: bool = {expression};")
        } else {
            format!("    let rg_strict: bool = rg_next {operator} rg_acc;")
        }
    }

    fn equal_before(&self) -> String {
        "    let rg_equal_before: bool = rg_next == rg_acc;".into()
    }

    fn update_extrema(&self) -> String {
        "    if (rg_strict) { rg_acc = rg_next; }".into()
    }

    fn update_first_index(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> String {
        format!(
            "    if (rg_strict || (rg_index == {}i && rg_equal_before)) {{ rg_index = i32(rg_axis); }}",
            plan.index_sentinel
        )
    }

    fn arithmetic(
        &self,
        plan: &crate::prefix_scan_native::NativePrefixScanPlan,
        operator: &str,
    ) -> Result<String, crate::prefix_scan_native::PortablePrefixScanError> {
        let expression = match plan.work_dtype {
            DType::Bool => "(rg_acc && rg_next)".into(),
            DType::I32 => {
                format!("bitcast<i32>(bitcast<u32>(rg_acc) {operator} bitcast<u32>(rg_next))")
            }
            DType::U32 | DType::F32 => format!("rg_acc {operator} rg_next"),
            _ => {
                return Err(
                    crate::prefix_scan_native::PortablePrefixScanError::Unsupported(
                        "WGSL portable scan arithmetic dtype",
                    ),
                );
            }
        };
        Ok(format!("    rg_acc = {expression};"))
    }

    fn store(&self, plan: &crate::prefix_scan_native::NativePrefixScanPlan) -> Vec<String> {
        if plan.result == crate::PrefixScanOutput::Indices {
            vec!["    b1[rg_offset] = rg_index;".into()]
        } else if plan.output_dtype == DType::Bool {
            Self::bool_store("rg_offset", "rg_acc", "    ")
        } else {
            vec!["    b1[rg_offset] = rg_acc;".into()]
        }
    }

    fn loop_close(&self) -> String {
        "  }".into()
    }
}

fn render_portable_f32_matmul(
    renderer: &WgslRenderer,
    root: &UOp,
    value: &crate::MatmulValue,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let portable = crate::matmul::PortableF32Matmul::new(value).map_err(|error| match error {
        crate::matmul::PortableF32MatmulError::Unsupported(reason) => {
            WebGpuError::Unsupported(reason.into())
        }
        crate::matmul::PortableF32MatmulError::Overflow => WebGpuError::Overflow,
        other => WebGpuError::InvalidBinding(other.to_string()),
    })?;
    let plan = portable.plan();
    let extent = u32::try_from(portable.extent()).map_err(|_| {
        WebGpuError::Unsupported("portable matmul extent exceeds u32 indexing".into())
    })?;
    for value in [
        portable.lhs_elements(),
        portable.rhs_elements(),
        plan.m,
        plan.n,
        plan.k,
    ] {
        u32::try_from(value).map_err(|_| {
            WebGpuError::Unsupported("portable matmul address exceeds u32 indexing".into())
        })?;
    }
    for axis in portable
        .lhs_batch_axes()
        .iter()
        .chain(portable.rhs_batch_axes())
    {
        for value in [axis.divisor, axis.dimension, axis.input_stride] {
            u32::try_from(value).map_err(|_| {
                WebGpuError::Unsupported("portable matmul batch address exceeds u32".into())
            })?;
        }
    }
    let mut buffers = Vec::with_capacity(3);
    let mut schedule_inputs = Vec::with_capacity(2);
    for input in portable.inputs() {
        let elements = input.shape.numel().map_err(|_| WebGpuError::Overflow)?;
        let abi = WgslBufferAbi {
            id: input.node.index() as u64,
            dtype: DType::F32,
            source_shape: input.shape.clone(),
            elements,
            mutable: false,
            view: None,
        };
        schedule_inputs.push(abi.clone());
        buffers.push(abi);
    }
    buffers.push(WgslBufferAbi {
        id: plan.output.index() as u64,
        dtype: DType::F32,
        source_shape: plan.output_shape.clone(),
        elements: portable.extent(),
        mutable: true,
        view: None,
    });
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "portable matmul bindings exceed adapter limit".into(),
        ));
    }
    for buffer in &buffers {
        let physical_bytes = if portable.extent() != 0 && buffer.elements == 0 {
            DType::F32.itemsize()
        } else {
            buffer.physical_bytes()?
        };
        if physical_bytes > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "portable matmul binding exceeds adapter buffer limit".into(),
            ));
        }
    }
    let positions = buffers
        .iter()
        .enumerate()
        .map(|(position, abi)| (abi.id, position))
        .collect::<BTreeMap<_, _>>();
    let lhs_position = positions[&(plan.lhs.index() as u64)];
    let rhs_position = positions[&(plan.rhs.index() as u64)];
    let output_position = buffers.len() - 1;
    let entry = format!("rg_wgsl_matmul_f32_{}", plan.cache_key);
    let mut lines = vec![format!(
        "// {WGSL_PORTABLE_F32_MATMUL_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"
    )];
    for (position, _) in buffers[..output_position].iter().enumerate() {
        lines.push(format!(
            "@group(0) @binding({position}) var<storage, read> b{position}: array<f32>;"
        ));
    }
    lines.extend([
        format!(
            "@group(0) @binding({output_position}) var<storage, read_write> b{output_position}: array<f32>;"
        ),
        "struct RustGradExtent { value: u32, };".into(),
        format!(
            "@group(0) @binding({}) var<uniform> rg_extent: RustGradExtent;",
            buffers.len()
        ),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
        "  var rg_q: u32 = gid;".into(),
        "  var rg_col: u32 = 0u;".into(),
        "  var rg_row: u32 = 0u;".into(),
    ]);
    if !plan.rhs_vector && plan.n != 0 {
        lines.push(format!(
            "  rg_col = rg_q % {}u; rg_q = rg_q / {}u;",
            plan.n, plan.n
        ));
    }
    if !plan.lhs_vector && plan.m != 0 {
        lines.push(format!(
            "  rg_row = rg_q % {}u; rg_q = rg_q / {}u;",
            plan.m, plan.m
        ));
    }
    lines.push("  let rg_batch: u32 = rg_q;".into());
    for (name, axes) in [
        ("rg_lbatch", portable.lhs_batch_axes()),
        ("rg_rbatch", portable.rhs_batch_axes()),
    ] {
        lines.push(format!("  var {name}: u32 = 0u;"));
        if extent != 0 {
            for axis in axes {
                lines.push(format!(
                    "  {name} = {name} + ((rg_batch / {}u) % {}u) * {}u;",
                    axis.divisor, axis.dimension, axis.input_stride
                ));
            }
        }
    }
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
    lines.extend([
        "  var rg_acc: f32 = 0.0;".into(),
        format!("  for (var rg_k: u32 = 0u; rg_k < {}u; rg_k = rg_k + 1u) {{", plan.k),
        format!(
            "    let rg_product: f32 = b{lhs_position}[{lhs_offset}] * b{rhs_position}[{rhs_offset}];"
        ),
        "    rg_acc = rg_acc + rg_product;".into(),
        "  }".into(),
        format!("  b{output_position}[gid] = rg_acc;"),
        "}".into(),
    ]);
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        WGSL_PORTABLE_F32_MATMUL_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.value(),
        extent,
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.extent(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

fn render_portable_bitcast(
    renderer: &WgslRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let portable = crate::movement_plan::PortableBitcast::new(plan)
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let extent = u32::try_from(portable.bytes()).map_err(|_| {
        WebGpuError::Unsupported("portable Bitcast byte extent exceeds u32 indexing".into())
    })?;
    validate_portable_serial_launch(
        portable.bytes(),
        renderer.local_size,
        &renderer.capabilities,
    )?;
    let input = portable.input();
    let input_abi = WgslBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: portable.input_elements(),
        mutable: false,
        view: None,
    };
    let buffers = vec![
        input_abi.clone(),
        WgslBufferAbi {
            id: plan.output.index() as u64,
            dtype: plan.dtype,
            source_shape: plan.output_shape.clone(),
            elements: portable.output_elements(),
            mutable: true,
            view: None,
        },
    ];
    for buffer in &buffers {
        if buffer.physical_bytes()? > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "portable Bitcast binding exceeds adapter buffer limit".into(),
            ));
        }
    }
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "portable Bitcast bindings exceed adapter limit".into(),
        ));
    }
    let entry = "rg_wgsl_portable_bitcast".to_owned();
    let stored = if portable.normalizes_bool() {
        "select(0u, 1u, rg_bits != 0u)"
    } else {
        "rg_bits"
    };
    let source = [
        format!("// {WGSL_PORTABLE_BITCAST_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"),
        "struct RustGradExtent { value: u32, };".into(),
        "@group(0) @binding(0) var<storage, read> b0: array<u32>;".into(),
        "@group(0) @binding(1) var<storage, read_write> b1: array<atomic<u32>>;".into(),
        "@group(0) @binding(2) var<uniform> rg_extent: RustGradExtent;".into(),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
        "  let rg_word: u32 = gid >> 2u;".into(),
        "  let rg_shift: u32 = (gid & 3u) * 8u;".into(),
        "  let rg_bits: u32 = (b0[rg_word] >> rg_shift) & 0xffu;".into(),
        format!("  let rg_stored: u32 = {stored};"),
        "  atomicAnd(&b1[rg_word], ~(0xffu << rg_shift));".into(),
        "  atomicOr(&b1[rg_word], rg_stored << rg_shift);".into(),
        "}".into(),
    ]
    .join("\n")
        + "\n";
    let cache_key = stable_key(&(
        WGSL_PORTABLE_BITCAST_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.plan(),
        extent,
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.bytes(),
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

fn render_portable_dense_materialization(
    renderer: &WgslRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let portable = crate::movement_plan::PortableDenseMaterialization::new(plan)
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let extent = u32::try_from(portable.elements()).map_err(|_| {
        WebGpuError::Unsupported(
            "portable dense materialization extent exceeds u32 indexing".into(),
        )
    })?;
    validate_portable_serial_launch(
        portable.elements(),
        renderer.local_size,
        &renderer.capabilities,
    )?;
    if portable.elements() != 0 {
        for region in portable.regions() {
            for axis in &region.axes {
                for value in [
                    axis.output_dimension,
                    axis.output_divisor,
                    axis.output_start,
                    axis.length,
                    axis.source_stride,
                ] {
                    u32::try_from(value).map_err(|_| {
                        WebGpuError::Unsupported(
                            "portable dense materialization address exceeds u32 indexing".into(),
                        )
                    })?;
                }
                u32::try_from(
                    axis.output_start
                        .checked_add(axis.length)
                        .ok_or(WebGpuError::Overflow)?,
                )
                .map_err(|_| {
                    WebGpuError::Unsupported(
                        "portable dense materialization address exceeds u32 indexing".into(),
                    )
                })?;
            }
        }
    }
    let mut buffers = portable
        .inputs()
        .iter()
        .map(|input| {
            Ok(WgslBufferAbi {
                id: input.node.index() as u64,
                dtype: input.dtype,
                source_shape: input.shape.clone(),
                elements: input.shape.numel().map_err(|_| WebGpuError::Overflow)?,
                mutable: false,
                view: None,
            })
        })
        .collect::<Result<Vec<_>, WebGpuError>>()?;
    let schedule_inputs = buffers.clone();
    buffers.push(WgslBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: portable.elements(),
        mutable: true,
        view: None,
    });
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "portable dense materialization bindings exceed adapter limit".into(),
        ));
    }
    for buffer in &buffers {
        if buffer.physical_bytes()? > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "portable dense materialization binding exceeds adapter limit".into(),
            ));
        }
    }
    let entry = "rg_wgsl_portable_dense_materialization".to_owned();
    let mut lines = vec![format!(
        "// {WGSL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"
    )];
    for (index, _) in portable.inputs().iter().enumerate() {
        lines.push(format!(
            "@group(0) @binding({index}) var<storage, read> b{index}: array<u32>;"
        ));
    }
    let output = portable.inputs().len();
    lines.extend([
        format!(
            "@group(0) @binding({output}) var<storage, read_write> b{output}: array<atomic<u32>>;"
        ),
        "struct RustGradExtent { value: u32, };".into(),
        format!(
            "@group(0) @binding({}) var<uniform> rg_extent: RustGradExtent;",
            output + 1
        ),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
    ]);
    lines.extend(
        crate::portable_movement::emit_portable_dense_materialization_body(
            &portable,
            &crate::portable_movement::WgslPortableDenseDialect,
        ),
    );
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        WGSL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        portable.plan(),
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: extent as usize,
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs,
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

fn render_raw_copy(
    renderer: &WgslRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let copy = plan
        .raw_copy()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
        .ok_or_else(|| {
            WebGpuError::Unsupported(
                "only raw AffineCopy and Contiguous have WGSL movement lowering".into(),
            )
        })?;
    let input = copy.input();
    let extent = copy.elements();
    let extent_u32 = u32::try_from(extent).map_err(|_| {
        WebGpuError::Unsupported("raw-copy WGSL extent exceeds u32 indexing".into())
    })?;
    let words = copy.bytes().checked_add(3).ok_or(WebGpuError::Overflow)? / 4;
    u32::try_from(words)
        .map_err(|_| WebGpuError::Unsupported("raw-copy WGSL words exceed u32 indexing".into()))?;
    u32::try_from(copy.input_elements()).map_err(|_| {
        WebGpuError::Unsupported("raw-copy WGSL source extent exceeds u32 indexing".into())
    })?;
    let input_words = copy
        .input_bytes()
        .checked_add(3)
        .ok_or(WebGpuError::Overflow)?
        / 4;
    u32::try_from(input_words).map_err(|_| {
        WebGpuError::Unsupported("raw-copy WGSL source words exceed u32 indexing".into())
    })?;
    let width = copy.width();
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(WebGpuError::Unsupported(format!(
            "raw-copy WGSL storage width {width}"
        )));
    }
    let input_abi = WgslBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: copy.input_elements(),
        mutable: false,
        view: None,
    };
    let output_abi = WgslBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    let buffers = vec![input_abi.clone(), output_abi];
    for abi in &buffers {
        if abi.physical_bytes()? > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "raw-copy binding exceeds adapter buffer limit".into(),
            ));
        }
    }
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "raw-copy bindings exceed adapter limit".into(),
        ));
    }
    let entry = format!("rg_wgsl_raw_copy_w{width}");
    let mut address_lines = Vec::new();
    let source_index = if let Some(address) = copy
        .address()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
    {
        let offset = u32::try_from(address.offset).map_err(|_| {
            WebGpuError::Unsupported("raw-copy WGSL offset exceeds u32 indexing".into())
        })?;
        address_lines.push(format!("  var rg_source: u32 = {offset}u;"));
        for axis in address.axes {
            let output_axis = axis.output_axis;
            let dimension = u32::try_from(axis.dimension).map_err(|_| {
                WebGpuError::Unsupported("raw-copy WGSL dimension exceeds u32 indexing".into())
            })?;
            let stride = u32::try_from(axis.stride).map_err(|_| {
                WebGpuError::Unsupported("raw-copy WGSL stride exceeds u32 indexing".into())
            })?;
            let divisor = u32::try_from(axis.divisor).map_err(|_| {
                WebGpuError::Unsupported("raw-copy WGSL divisor exceeds u32 indexing".into())
            })?;
            address_lines.push(format!(
                "  var rg_axis_{output_axis}: u32 = (gid / {divisor}u) % {dimension}u;"
            ));
            if axis.reversed {
                address_lines.push(format!(
                    "  rg_axis_{output_axis} = {}u - rg_axis_{output_axis};",
                    dimension - 1
                ));
            }
            address_lines.push(format!(
                "  rg_source = rg_source + rg_axis_{output_axis} * {stride}u;"
            ));
        }
        "rg_source"
    } else {
        "gid"
    };
    let copy_lines = match width {
        1 => vec![
            format!("  let rg_source_word: u32 = {source_index} >> 2u;"),
            format!("  let rg_source_shift: u32 = ({source_index} & 3u) * 8u;"),
            "  let rg_output_word: u32 = gid >> 2u;".into(),
            "  let rg_output_shift: u32 = (gid & 3u) * 8u;".into(),
            "  let rg_bits: u32 = (b0[rg_source_word] >> rg_source_shift) & 0xffu;".into(),
            "  atomicAnd(&b1[rg_output_word], ~(0xffu << rg_output_shift));".into(),
            "  atomicOr(&b1[rg_output_word], rg_bits << rg_output_shift);".into(),
        ],
        2 => vec![
            format!("  let rg_source_word: u32 = {source_index} >> 1u;"),
            format!("  let rg_source_shift: u32 = ({source_index} & 1u) * 16u;"),
            "  let rg_output_word: u32 = gid >> 1u;".into(),
            "  let rg_output_shift: u32 = (gid & 1u) * 16u;".into(),
            "  let rg_bits: u32 = (b0[rg_source_word] >> rg_source_shift) & 0xffffu;".into(),
            "  atomicAnd(&b1[rg_output_word], ~(0xffffu << rg_output_shift));".into(),
            "  atomicOr(&b1[rg_output_word], rg_bits << rg_output_shift);".into(),
        ],
        4 => vec![format!("  atomicStore(&b1[gid], b0[{source_index}]);")],
        8 => vec![
            format!("  let rg_source_word: u32 = {source_index} * 2u;"),
            "  let rg_output_word: u32 = gid * 2u;".into(),
            "  atomicStore(&b1[rg_output_word], b0[rg_source_word]);".into(),
            "  atomicStore(&b1[rg_output_word + 1u], b0[rg_source_word + 1u]);".into(),
        ],
        _ => unreachable!("validated raw width"),
    };
    let mut lines = vec![
        format!("// {WGSL_RAW_COPY_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"),
        "struct RustGradExtent { value: u32, };".into(),
        "@group(0) @binding(0) var<storage, read> b0: array<u32>;".into(),
        "@group(0) @binding(1) var<storage, read_write> b1: array<atomic<u32>>;".into(),
        "@group(0) @binding(2) var<uniform> rg_extent: RustGradExtent;".into(),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
    ];
    lines.extend(address_lines);
    lines.extend(copy_lines);
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        WGSL_RAW_COPY_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        copy.plan(),
        extent_u32,
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent,
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

fn render_static_positions(
    renderer: &WgslRenderer,
    root: &UOp,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedWgsl, WebGpuError> {
    root.validate()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
    let placement = plan
        .static_position_write()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
        .ok_or_else(|| WebGpuError::InvalidBinding("missing static-position projection".into()))?;
    let input = placement.input();
    let extent = placement.elements();
    let extent_u32 = u32::try_from(extent).map_err(|_| {
        WebGpuError::Unsupported("static-position WGSL extent exceeds u32 indexing".into())
    })?;
    let words = placement
        .bytes()
        .checked_add(3)
        .ok_or(WebGpuError::Overflow)?
        / 4;
    u32::try_from(words).map_err(|_| {
        WebGpuError::Unsupported("static-position WGSL words exceed u32 indexing".into())
    })?;
    u32::try_from(placement.input_elements()).map_err(|_| {
        WebGpuError::Unsupported("static-position WGSL source extent exceeds u32 indexing".into())
    })?;
    let input_words = placement
        .input_bytes()
        .checked_add(3)
        .ok_or(WebGpuError::Overflow)?
        / 4;
    u32::try_from(input_words).map_err(|_| {
        WebGpuError::Unsupported("static-position WGSL source words exceed u32 indexing".into())
    })?;
    let width = placement.width();
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(WebGpuError::Unsupported(format!(
            "static-position WGSL storage width {width}"
        )));
    }
    let input_abi = WgslBufferAbi {
        id: input.node.index() as u64,
        dtype: input.dtype,
        source_shape: input.shape.clone(),
        elements: placement.input_elements(),
        mutable: false,
        view: None,
    };
    let output_abi = WgslBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.output_shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    let buffers = vec![input_abi.clone(), output_abi];
    for abi in &buffers {
        if abi.physical_bytes()? > renderer.capabilities.max_buffer_size {
            return Err(WebGpuError::Unsupported(
                "static-position binding exceeds adapter buffer limit".into(),
            ));
        }
    }
    if buffers.len() > renderer.capabilities.max_storage_buffers_per_shader_stage as usize {
        return Err(WebGpuError::Unsupported(
            "static-position bindings exceed adapter limit".into(),
        ));
    }
    let entry = format!("rg_wgsl_static_position_w{width}");
    let mut address_lines = vec![
        "  var rg_mapped: bool = false;".into(),
        "  var rg_source: u32 = 0u;".into(),
    ];
    if placement.has_source() {
        address_lines.push("  rg_mapped = true;".into());
        for axis in placement.axes() {
            let name = axis.output_axis;
            let output_dimension = u32::try_from(axis.output_dimension).map_err(|_| {
                WebGpuError::Unsupported(
                    "static-position WGSL dimension exceeds u32 indexing".into(),
                )
            })?;
            let output_divisor = u32::try_from(axis.output_divisor).map_err(|_| {
                WebGpuError::Unsupported("static-position WGSL divisor exceeds u32 indexing".into())
            })?;
            let source_dimension = u32::try_from(axis.source_dimension).map_err(|_| {
                WebGpuError::Unsupported(
                    "static-position WGSL source dimension exceeds u32 indexing".into(),
                )
            })?;
            let source_stride = u32::try_from(axis.source_stride).map_err(|_| {
                WebGpuError::Unsupported(
                    "static-position WGSL source stride exceeds u32 indexing".into(),
                )
            })?;
            let first = u32::try_from(axis.first).map_err(|_| {
                WebGpuError::Unsupported("static-position WGSL start exceeds u32 indexing".into())
            })?;
            let spacing = u32::try_from(axis.spacing).map_err(|_| {
                WebGpuError::Unsupported("static-position WGSL step exceeds u32 indexing".into())
            })?;
            address_lines.push(format!(
                "  let rg_coordinate_{name}: u32 = (gid / {output_divisor}u) % {output_dimension}u;"
            ));
            address_lines.push(format!(
                "  let rg_delta_{name}: u32 = select(0u, rg_coordinate_{name} - {first}u, rg_coordinate_{name} >= {first}u);"
            ));
            address_lines.push(format!(
                "  let rg_quotient_{name}: u32 = rg_delta_{name} / {spacing}u;"
            ));
            address_lines.push(format!(
                "  if (rg_coordinate_{name} < {first}u || rg_delta_{name} % {spacing}u != 0u || rg_quotient_{name} >= {source_dimension}u) {{ rg_mapped = false; }}"
            ));
            address_lines.push(format!("  var rg_source_axis_{name}: u32 = 0u;"));
            address_lines.push(format!(
                "  if (rg_quotient_{name} < {source_dimension}u) {{ rg_source_axis_{name} = {}; }}",
                if axis.reversed {
                    format!("{}u - rg_quotient_{name}", source_dimension - 1)
                } else {
                    format!("rg_quotient_{name}")
                }
            ));
            address_lines.push(format!(
                "  rg_source = rg_source + rg_source_axis_{name} * {source_stride}u;"
            ));
        }
    }
    let copy_lines = match width {
        1 => vec![
            "  var rg_bits: u32 = 0u;".into(),
            "  if (rg_mapped) {".into(),
            "    let rg_source_word: u32 = rg_source >> 2u;".into(),
            "    let rg_source_shift: u32 = (rg_source & 3u) * 8u;".into(),
            "    rg_bits = (b0[rg_source_word] >> rg_source_shift) & 0xffu;".into(),
            "  }".into(),
            "  let rg_output_word: u32 = gid >> 2u;".into(),
            "  let rg_output_shift: u32 = (gid & 3u) * 8u;".into(),
            "  atomicAnd(&b1[rg_output_word], ~(0xffu << rg_output_shift));".into(),
            "  atomicOr(&b1[rg_output_word], rg_bits << rg_output_shift);".into(),
        ],
        2 => vec![
            "  var rg_bits: u32 = 0u;".into(),
            "  if (rg_mapped) {".into(),
            "    let rg_source_word: u32 = rg_source >> 1u;".into(),
            "    let rg_source_shift: u32 = (rg_source & 1u) * 16u;".into(),
            "    rg_bits = (b0[rg_source_word] >> rg_source_shift) & 0xffffu;".into(),
            "  }".into(),
            "  let rg_output_word: u32 = gid >> 1u;".into(),
            "  let rg_output_shift: u32 = (gid & 1u) * 16u;".into(),
            "  atomicAnd(&b1[rg_output_word], ~(0xffffu << rg_output_shift));".into(),
            "  atomicOr(&b1[rg_output_word], rg_bits << rg_output_shift);".into(),
        ],
        4 => vec![
            "  var rg_bits: u32 = 0u;".into(),
            "  if (rg_mapped) { rg_bits = b0[rg_source]; }".into(),
            "  atomicStore(&b1[gid], rg_bits);".into(),
        ],
        8 => vec![
            "  var rg_low: u32 = 0u;".into(),
            "  var rg_high: u32 = 0u;".into(),
            "  if (rg_mapped) {".into(),
            "    let rg_source_word: u32 = rg_source * 2u;".into(),
            "    rg_low = b0[rg_source_word];".into(),
            "    rg_high = b0[rg_source_word + 1u];".into(),
            "  }".into(),
            "  let rg_output_word: u32 = gid * 2u;".into(),
            "  atomicStore(&b1[rg_output_word], rg_low);".into(),
            "  atomicStore(&b1[rg_output_word + 1u], rg_high);".into(),
        ],
        _ => unreachable!("validated raw width"),
    };
    let mut lines = vec![
        format!("// {WGSL_STATIC_POSITION_RENDERER_VERSION} ABI {WEBGPU_ABI_VERSION}"),
        "struct RustGradExtent { value: u32, };".into(),
        "@group(0) @binding(0) var<storage, read> b0: array<u32>;".into(),
        "@group(0) @binding(1) var<storage, read_write> b1: array<atomic<u32>>;".into(),
        "@group(0) @binding(2) var<uniform> rg_extent: RustGradExtent;".into(),
        format!("@compute @workgroup_size({}, 1, 1)", renderer.local_size),
        format!("fn {entry}(@builtin(global_invocation_id) rg_global: vec3<u32>) {{"),
        "  let gid: u32 = rg_global.x;".into(),
        "  if (gid >= rg_extent.value) { return; }".into(),
    ];
    lines.extend(address_lines);
    lines.extend(copy_lines);
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = stable_key(&(
        WGSL_STATIC_POSITION_RENDERER_VERSION,
        WEBGPU_ABI_VERSION,
        renderer.local_size,
        &renderer.capabilities,
        placement.plan(),
        extent_u32,
        &source,
        &buffers,
    ));
    let rendered = RenderedWgsl {
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent,
        entry,
        cache_key,
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs: vec![input_abi],
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
            root.clone(),
        ))),
    };
    rendered.validate_artifact()?;
    Ok(rendered)
}

fn emit_wgsl_reduction(
    reduction: &crate::reduction_native::NativeReductionKernel<'_>,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
) -> Result<String, WebGpuError> {
    let plan = &reduction.plan;
    let producer = reduction.producer;
    let reduction_len = u32::try_from(plan.reduction_len()).map_err(|_| {
        WebGpuError::Unsupported("reduction domain exceeds WGSL u32 indexing".into())
    })?;
    let accumulator_type = wgsl_reduction_type(plan.accumulator_dtype);
    let identity = wgsl_reduction_literal(plan.accumulator_dtype, plan.identity())?;
    lines.push(format!("  var rg_acc: {accumulator_type} = {identity};"));
    if reduction_len != 0 {
        lines.push(format!(
            "  for (var rg_r: u32 = 0u; rg_r < {reduction_len}u; rg_r = rg_r + 1u) {{"
        ));
        let source_index =
            crate::reduction_native::index_expression(&plan.geometry, "gid", "rg_r", "u");
        lines.push(format!("    let rg_src: u32 = {source_index};"));
        let candidate = emit_expr(producer, ids, source_map, lines, "rg_src")?;
        let candidate = WgslScalarDialect
            .cast(plan.source_dtype, plan.accumulator_dtype, &candidate)
            .map_err(WebGpuError::Unsupported)?;
        if plan.is_singleton_identity() {
            lines.push(format!("    rg_acc = {candidate};"));
        } else {
            match plan.kind {
                crate::ReduceKind::Sum | crate::ReduceKind::Mean => lines.push(format!(
                    "    rg_acc = {};",
                    if plan.accumulator_dtype == DType::Bool {
                        format!("(rg_acc || ({candidate}))")
                    } else {
                        wgsl_reduction_arithmetic(
                            plan.accumulator_dtype,
                            "rg_acc",
                            &candidate,
                            false,
                        )
                    }
                )),
                crate::ReduceKind::Product => lines.push(format!(
                    "    rg_acc = {};",
                    if plan.accumulator_dtype == DType::Bool {
                        format!("(rg_acc && ({candidate}))")
                    } else {
                        wgsl_reduction_arithmetic(
                            plan.accumulator_dtype,
                            "rg_acc",
                            &candidate,
                            true,
                        )
                    }
                )),
                crate::ReduceKind::Max | crate::ReduceKind::Min => {
                    if plan.accumulator_dtype == DType::Bool {
                        lines.push(format!(
                            "    rg_acc = rg_acc {} ({candidate});",
                            if plan.kind == crate::ReduceKind::Max {
                                "||"
                            } else {
                                "&&"
                            }
                        ));
                    } else {
                        let comparison = if plan.kind == crate::ReduceKind::Max {
                            ">"
                        } else {
                            "<"
                        };
                        lines.push(format!(
                            "    if (({candidate}) {comparison} rg_acc) {{ rg_acc = {candidate}; }}"
                        ));
                    }
                }
                crate::ReduceKind::Any => {
                    lines.push(format!("    rg_acc = rg_acc || ({candidate});"));
                }
                crate::ReduceKind::All => {
                    lines.push(format!("    rg_acc = rg_acc && ({candidate});"));
                }
            }
        }
        lines.push("  }".into());
    }
    if plan.kind == crate::ReduceKind::Mean {
        if reduction_len == 0 {
            lines.push("  rg_acc = bitcast<f32>(0x7fc00000u);".into());
        } else {
            let divisor = wgsl_reduction_literal(
                plan.accumulator_dtype,
                plan.mean_divisor()
                    .expect("nonempty validated Mean divisor"),
            )?;
            let divided = format!("(rg_acc / {divisor})");
            lines.push(format!(
                "  rg_acc = {};",
                wgsl_reduction_commit(plan.accumulator_dtype, &divided)
            ));
        }
    }
    let finalized = WgslScalarDialect
        .cast(plan.accumulator_dtype, plan.output_dtype, "rg_acc")
        .map_err(WebGpuError::Unsupported)?;
    let committed = wgsl_reduction_commit(plan.output_dtype, &finalized);
    if reduction.has_epilogue() {
        emit_expr_with_substitution(
            reduction.epilogue_root,
            ids,
            source_map,
            lines,
            "gid",
            Some((reduction.finalize, committed.as_str())),
        )
    } else {
        Ok(committed)
    }
}

fn wgsl_reduction_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::BF16 | DType::F32 => "f32",
        DType::Bool => "bool",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => unreachable!("validated WGSL reduction storage"),
    }
}

fn wgsl_reduction_literal(dtype: DType, value: crate::Scalar) -> Result<String, WebGpuError> {
    Ok(match value {
        crate::Scalar::Bool(value) => value.to_string(),
        crate::Scalar::I(value) if dtype == DType::I32 => {
            format!("bitcast<i32>(0x{:08x}u)", value as u32)
        }
        crate::Scalar::U(value) => format!("{value}u"),
        crate::Scalar::F(value) => {
            format!("bitcast<f32>(0x{:08x}u)", (value as f32).to_bits())
        }
        _ => {
            return Err(WebGpuError::Unsupported(
                "WGSL reduction identity is outside the exact storage subset".into(),
            ));
        }
    })
}

fn wgsl_reduction_arithmetic(dtype: DType, lhs: &str, rhs: &str, product: bool) -> String {
    let operator = if product { "*" } else { "+" };
    let value = if dtype == DType::I32 {
        format!("bitcast<i32>(bitcast<u32>({lhs}) {operator} bitcast<u32>({rhs}))")
    } else {
        format!("(({lhs}) {operator} ({rhs}))")
    };
    wgsl_reduction_commit(dtype, &value)
}

fn wgsl_reduction_commit(dtype: DType, value: &str) -> String {
    narrow::quantize(dtype, value).unwrap_or_else(|| match dtype {
        DType::F32 => format!("f32({value})"),
        DType::I32 => format!("i32({value})"),
        DType::U32 => format!("u32({value})"),
        DType::Bool => format!("bool({value})"),
        _ => unreachable!("validated WGSL reduction storage"),
    })
}

fn supported_storage(dtype: DType) -> Result<(), WebGpuError> {
    match dtype {
        DType::F16 | DType::BF16 | DType::F32 | DType::Bool | DType::I32 | DType::U32 => Ok(()),
        _ => Err(WebGpuError::Unsupported(format!(
            "dtype {dtype:?} is outside the exact WGSL static subset"
        ))),
    }
}

fn emit_raw_predicated_narrow_load(
    value: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
) -> Result<Option<String>, WebGpuError> {
    let Some(plan) = crate::projected_index::ProjectedIndexPlan::from_direct_predicated_load(value)
        .map_err(|_| WebGpuError::Unsupported("invalid predicated narrow load".into()))?
    else {
        return Ok(None);
    };
    if !plan.fits_i32()
        || plan.elements > i32::MAX as usize
        || plan.output_elements > i32::MAX as usize
    {
        return Err(WebGpuError::Unsupported(
            "projected index exceeds WGSL signed address range".into(),
        ));
    }
    let position = ids
        .get(&plan.buffer)
        .ok_or_else(|| WebGpuError::InvalidBinding("load buffer absent from ABI".into()))?;
    let access = crate::projected_index::render_infix_projected_access(
        &plan,
        format!("i32({linear})"),
        |literal| {
            i32::try_from(literal)
                .map(|literal| {
                    if literal == i32::MIN {
                        "((-2147483647i) - 1i)".into()
                    } else {
                        format!("{literal}i")
                    }
                })
                .map_err(|_| crate::UOpError::InvalidIndex)
        },
        |boolean| boolean.to_string(),
    )
    .map_err(|_| WebGpuError::Unsupported("invalid predicated narrow load".into()))?;
    let predicate = access
        .predicate
        .ok_or_else(|| WebGpuError::Unsupported("predicated narrow load has no guard".into()))?;
    let expression_id = source_map.len();
    source_map.insert(expression_id, lines.len() + 1);
    let temporary = format!("rg_predicated_raw_{expression_id}");
    let offset = format!("u32({})", access.offset);
    let raw = format!("((b{position}[({offset}) >> 1u] >> ((({offset}) & 1u) * 16u)) & 0xffffu)");
    lines.push(format!("var {temporary}: u32 = 0u;"));
    lines.push(format!("if ({predicate}) {{"));
    lines.push(format!("  {temporary} = {raw};"));
    lines.push("}".into());
    Ok(Some(temporary))
}

fn wgsl_storage_decl(dtype: DType, mutable: bool) -> &'static str {
    match (dtype, mutable) {
        (DType::F32, _) => "f32",
        (DType::I32, _) => "i32",
        (DType::U32, _) => "u32",
        (DType::Bool, true) => "atomic<u32>",
        (DType::Bool, false) => "u32",
        (DType::F16 | DType::BF16, true) => "atomic<u32>",
        (DType::F16 | DType::BF16, false) => "u32",
        _ => unreachable!("validated WGSL storage"),
    }
}

pub(super) struct WgslScalarDialect;

impl dialect_seal::Sealed for WgslScalarDialect {}

impl ScalarLaneDialect for WgslScalarDialect {
    fn name(&self) -> &'static str {
        "WGSL"
    }

    fn supports_value(&self, dtype: DType) -> bool {
        supported_storage(dtype).is_ok()
    }

    fn cast(&self, source: DType, target: DType, value: &str) -> Result<String, String> {
        emit_cast(source, target, value).map_err(|error| error.to_string())
    }

    fn finish_float(&self, dtype: DType, value: String) -> Result<String, String> {
        Ok(narrow::quantize(dtype, &value).unwrap_or(value))
    }

    fn signed_infix(
        &self,
        dtype: DType,
        operator: &'static str,
        lhs: &str,
        rhs: &str,
    ) -> Result<String, String> {
        if dtype == DType::I32 {
            Ok(format!(
                "bitcast<i32>(bitcast<u32>({lhs}) {operator} bitcast<u32>({rhs}))"
            ))
        } else {
            Err("WGSL signed wrapping requires I32".into())
        }
    }

    fn signed_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        if dtype == DType::I32 {
            Ok(format!("bitcast<i32>(0u - bitcast<u32>({value}))"))
        } else {
            Err("WGSL signed negation requires I32".into())
        }
    }

    fn unsigned_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        if dtype == DType::U32 {
            Ok(format!("(0u - ({value}))"))
        } else {
            Err("WGSL unsigned negation requires U32".into())
        }
    }

    fn signed_abs(&self, dtype: DType, value: &str) -> Result<String, String> {
        if dtype == DType::I32 {
            Ok(format!(
                "select(bitcast<i32>(0u - bitcast<u32>({value})), ({value}), ({value}) >= 0i)"
            ))
        } else {
            Err("WGSL signed absolute value requires I32".into())
        }
    }

    fn float_abs(&self, value: &str) -> String {
        format!("abs({value})")
    }

    fn bool_value(&self, expression: String) -> String {
        format!("({expression})")
    }

    fn select(&self, condition: &str, on_true: &str, on_false: &str) -> String {
        format!("select(({on_false}), ({on_true}), ({condition}))")
    }

    fn compare_operand(&self, dtype: DType, value: &str) -> String {
        ordered_compare_operand(dtype, value)
    }

    fn call_intrinsic(&self, canonical_name: &'static str, value: &str) -> String {
        format!("{canonical_name}({value})")
    }

    fn float_one(&self, dtype: DType) -> Result<&'static str, String> {
        if matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
            Ok("1.0")
        } else {
            Err("WGSL reciprocal requires floating dtype".into())
        }
    }
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
) -> Result<String, WebGpuError> {
    emit_expr_with_substitution(node, ids, source_map, lines, linear, None)
}

fn emit_expr_with_substitution(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
    substitution: Option<(&UOp, &str)>,
) -> Result<String, WebGpuError> {
    if let Some((target, value)) = substitution
        && node.shares_node_with(target)
    {
        return Ok(value.into());
    }
    let expression_id = source_map.len();
    source_map.insert(expression_id, lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| WebGpuError::Unsupported(format!("untyped {:?}", node.operation())))?
        .scalar;
    supported_storage(dtype)?;
    let child = |position: usize,
                 source_map: &mut BTreeMap<usize, usize>,
                 lines: &mut Vec<String>| {
        node.sources()
            .get(position)
            .ok_or_else(|| WebGpuError::Unsupported("missing expression operand".into()))
            .and_then(|source| {
                emit_expr_with_substitution(source, ids, source_map, lines, linear, substitution)
            })
    };
    match node.operation() {
        Operation::Const(value) => match value {
            LiteralValue::Scalar {
                dtype: DType::F32,
                bits,
            } => Ok(format!("bitcast<f32>(0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::Bool,
                bits,
            } if *bits <= 1 => Ok(if *bits == 0 {
                "false".into()
            } else {
                "true".into()
            }),
            LiteralValue::Scalar {
                dtype: DType::I32,
                bits,
            } => Ok(format!("bitcast<i32>(0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::U32,
                bits,
            } => Ok(format!("0x{:08x}u", *bits as u32)),
            LiteralValue::Scalar { dtype, bits } if narrow::is_narrow(*dtype) => {
                Ok(narrow::decode(*dtype, format!("0x{:04x}u", *bits as u16))
                    .expect("validated narrow scalar"))
            }
            _ => Err(WebGpuError::Unsupported(
                "invalid WGSL scalar literal".into(),
            )),
        },
        Operation::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| WebGpuError::Unsupported("load has no index".into()))?;
            let (buffer, input_shape, output_shape, view) = match index.operation() {
                Operation::Index(IndexValue::Buffer { buffer, .. })
                    if crate::projected_index::ProjectedIndexPlan::is_projected(index) =>
                {
                    let plan = crate::projected_index::ProjectedIndexPlan::from_index(index)
                        .map_err(|_| WebGpuError::Unsupported("invalid projected index".into()))?;
                    if !plan.fits_i32()
                        || plan.elements > i32::MAX as usize
                        || plan.output_elements > i32::MAX as usize
                    {
                        return Err(WebGpuError::Unsupported(
                            "projected index exceeds WGSL signed address range".into(),
                        ));
                    }
                    let access = crate::projected_index::render_infix_projected_access(
                        &plan,
                        format!("i32({linear})"),
                        |value| {
                            i32::try_from(value)
                                .map(|value| {
                                    if value == i32::MIN {
                                        "((-2147483647i) - 1i)".into()
                                    } else {
                                        format!("{value}i")
                                    }
                                })
                                .map_err(|_| crate::UOpError::InvalidIndex)
                        },
                        |value| value.to_string(),
                    )
                    .map_err(|_| WebGpuError::Unsupported("invalid projected index".into()))?;
                    let offset = format!("u32({})", access.offset);
                    let position = ids.get(buffer).ok_or_else(|| {
                        WebGpuError::InvalidBinding("load buffer absent from ABI".into())
                    })?;
                    let value = if dtype == DType::Bool {
                        format!(
                            "(((b{position}[({offset}) >> 2u] >> ((({offset}) & 3u) * 8u)) & 0xffu) != 0u)"
                        )
                    } else if narrow::is_narrow(dtype) {
                        let raw = format!(
                            "((b{position}[({offset}) >> 1u] >> ((({offset}) & 1u) * 16u)) & 0xffffu)"
                        );
                        narrow::decode(dtype, raw).expect("validated narrow load")
                    } else {
                        format!("b{position}[{offset}]")
                    };
                    let Some(predicate) = access.predicate else {
                        return Ok(value);
                    };
                    let zero = match dtype {
                        DType::Bool => "false",
                        DType::F16 | DType::BF16 | DType::F32 => "0.0f",
                        DType::I32 => "0i",
                        DType::U32 => "0u",
                        _ => unreachable!("validated WGSL storage"),
                    };
                    let temporary = format!("rg_predicated_{expression_id}");
                    lines.push(format!(
                        "var {temporary}: {} = {zero};",
                        wgsl_reduction_type(dtype)
                    ));
                    lines.push(format!("if ({predicate}) {{"));
                    lines.push(format!("  {temporary} = {value};"));
                    lines.push("}".into());
                    return Ok(temporary);
                }
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    input_shape,
                    output_shape,
                    ..
                }) => (*buffer, input_shape, output_shape, None),
                Operation::Index(IndexValue::View {
                    buffer,
                    input_shape,
                    output_shape,
                    view,
                    ..
                }) => (*buffer, input_shape, output_shape, Some(view)),
                _ => {
                    return Err(WebGpuError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            let position = ids
                .get(&buffer)
                .ok_or_else(|| WebGpuError::InvalidBinding("load buffer absent from ABI".into()))?;
            let logical = broadcast_offset(input_shape, output_shape, linear)?;
            let offset = match view {
                Some(view) => WgslViewAccess::new(view)?.expression(&logical),
                None => logical,
            };
            if dtype == DType::Bool {
                Ok(format!(
                    "(((b{position}[({offset}) >> 2u] >> ((({offset}) & 3u) * 8u)) & 0xffu) != 0u)"
                ))
            } else if narrow::is_narrow(dtype) {
                let raw = format!(
                    "((b{position}[({offset}) >> 1u] >> ((({offset}) & 1u) * 16u)) & 0xffffu)"
                );
                Ok(narrow::decode(dtype, raw).expect("validated narrow load"))
            } else {
                Ok(format!("b{position}[{offset}]"))
            }
        }
        other => {
            let mut sources = Vec::with_capacity(node.sources().len());
            for slot in 0..node.sources().len() {
                sources.push(child(slot, source_map, lines)?);
            }
            let instruction = project_scalar_lane(node, &sources)
                .map_err(WebGpuError::Unsupported)?
                .ok_or_else(|| WebGpuError::Unsupported(format!("{other:?}")))?;
            emit_scalar_lane(&WgslScalarDialect, &instruction).map_err(WebGpuError::Unsupported)
        }
    }
}

fn emit_cast(source: DType, target: DType, value: &str) -> Result<String, WebGpuError> {
    Ok(match (source, target) {
        (a, b) if a == b => value.into(),
        (DType::Bool, DType::F32) => format!("select(0.0, 1.0, {value})"),
        (DType::F32, DType::Bool) => format!("(({value}) != 0.0)"),
        (DType::Bool, DType::I32) => format!("select(0i, 1i, {value})"),
        (DType::Bool, DType::U32) => format!("select(0u, 1u, {value})"),
        (DType::I32, DType::Bool) => format!("(({value}) != 0i)"),
        (DType::U32, DType::Bool) => format!("(({value}) != 0u)"),
        (DType::I32, DType::U32) => format!("bitcast<u32>({value})"),
        (DType::U32, DType::I32) => format!("bitcast<i32>({value})"),
        (DType::I32, DType::F32) => format!("f32({value})"),
        (DType::U32, DType::F32) => format!("f32({value})"),
        (DType::F32, DType::I32) => format!("rg_f32_to_i32({value})"),
        (DType::F32, DType::U32) => format!("rg_f32_to_u32({value})"),
        (source, target)
            if narrow::is_narrow(target)
                && matches!(source, DType::F16 | DType::BF16 | DType::F32) =>
        {
            narrow::quantize(target, value).expect("validated narrow cast target")
        }
        (source, DType::F32) if narrow::is_narrow(source) => value.into(),
        _ => {
            return Err(WebGpuError::Unsupported(
                "cast is outside the exact WGSL subset".into(),
            ));
        }
    })
}

pub(super) fn ordered_compare_operand(dtype: DType, value: &str) -> String {
    if dtype == DType::Bool {
        format!("select(0u, 1u, {value})")
    } else {
        value.into()
    }
}

#[derive(Clone, Debug)]
pub(super) struct WgslViewAccess {
    source_shape: Shape,
    logical_shape: Shape,
    strides: Vec<i64>,
    offset: i64,
}

/// Ensures the emitted left-to-right WGSL `i32` affine expression cannot
/// overflow, including intermediate partial sums. WGSL has no portable i64.
fn signed_i32_safe(view: &AffineView) -> Result<(), WebGpuError> {
    if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&view.offset) {
        return Err(WebGpuError::Unsupported(
            "signed affine views exceed WGSL i32 indexing".into(),
        ));
    }
    let mut minimum = view.offset;
    let mut maximum = view.offset;
    for (&dim, &stride) in view.logical_shape.dims().iter().zip(&view.strides) {
        let coordinate_max =
            i64::try_from(dim.saturating_sub(1)).map_err(|_| WebGpuError::Overflow)?;
        let term = coordinate_max
            .checked_mul(stride)
            .ok_or(WebGpuError::Overflow)?;
        if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&term) {
            return Err(WebGpuError::Unsupported(
                "signed affine views exceed WGSL i32 indexing".into(),
            ));
        }
        if term < 0 {
            minimum = minimum.checked_add(term).ok_or(WebGpuError::Overflow)?;
        } else {
            maximum = maximum.checked_add(term).ok_or(WebGpuError::Overflow)?;
        }
        if minimum < 0 || maximum > i64::from(i32::MAX) {
            return Err(WebGpuError::Unsupported(
                "signed affine views exceed WGSL i32 indexing".into(),
            ));
        }
    }
    Ok(())
}

impl WgslViewAccess {
    pub(super) fn new(view: &AffineView) -> Result<Self, WebGpuError> {
        if view.logical_shape.rank() != view.strides.len() {
            return Err(WebGpuError::Unsupported("view rank/stride mismatch".into()));
        }
        view.validate_read()
            .map_err(|_| WebGpuError::Unsupported("invalid signed affine read map".into()))?;
        let source_elements = view
            .source_shape
            .numel()
            .map_err(|_| WebGpuError::Overflow)?;
        if source_elements > i32::MAX as usize {
            return Err(WebGpuError::Unsupported(
                "signed affine views exceed WGSL i32 indexing".into(),
            ));
        }
        signed_i32_safe(view)?;
        Ok(Self {
            source_shape: view.source_shape.clone(),
            logical_shape: view.logical_shape.clone(),
            strides: view.strides.clone(),
            offset: view.offset,
        })
    }

    pub(super) fn expression(&self, logical: &str) -> String {
        if self.offset >= 0 && self.strides.iter().all(|stride| *stride >= 0) {
            return self.unsigned_expression(logical);
        }
        self.signed_expression(logical)
    }

    fn unsigned_expression(&self, logical: &str) -> String {
        if self.logical_shape.numel().ok() == Some(1) {
            return format!("{}u", self.offset);
        }
        let logical_strides = self.logical_shape.contiguous_strides();
        let mut terms = Vec::new();
        if self.offset != 0 {
            terms.push(format!("{}u", self.offset));
        }
        for ((dim, stride), logical_stride) in self
            .logical_shape
            .dims()
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .zip(logical_strides)
        {
            if dim > 1 && stride != 0 {
                terms.push(format!(
                    "((({logical}) / {logical_stride}u) % {dim}u) * {stride}u"
                ));
            }
        }
        if terms.is_empty() {
            "0u".into()
        } else {
            format!("({})", terms.join(" + "))
        }
    }

    fn signed_expression(&self, logical: &str) -> String {
        let logical_strides = self.logical_shape.contiguous_strides();
        let mut terms = vec![format!("{}i", self.offset)];
        for ((dim, stride), logical_stride) in self
            .logical_shape
            .dims()
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .zip(logical_strides)
        {
            if dim > 1 && stride != 0 {
                terms.push(format!(
                    "(((i32({logical}) / {logical_stride}i) % {dim}i) * {stride}i)"
                ));
            }
        }
        // `signed_i32_safe` and `AffineView::validate_read` prove the final
        // expression is non-negative and does not overflow WGSL's i32 range.
        format!("u32({})", terms.join(" + "))
    }
}

pub(super) fn broadcast_offset(
    input: &Shape,
    output: &Shape,
    linear: &str,
) -> Result<String, WebGpuError> {
    if input.rank() > output.rank() {
        return Err(WebGpuError::Unsupported(
            "input rank exceeds output rank".into(),
        ));
    }
    if input.rank() == 0 {
        return Ok("0u".into());
    }
    let input_strides = input.contiguous_strides();
    let output_strides = output.contiguous_strides();
    if input_strides
        .iter()
        .chain(&output_strides)
        .any(|value| *value > u32::MAX as usize)
    {
        return Err(WebGpuError::Unsupported(
            "shape exceeds WGSL u32 indexing".into(),
        ));
    }
    let pad = output.rank() - input.rank();
    let mut terms = Vec::new();
    for axis in 0..input.rank() {
        let dim = input.dims()[axis];
        let output_dim = output.dims()[pad + axis];
        if dim != 1 && dim != output_dim {
            return Err(WebGpuError::Unsupported(
                "invalid broadcast metadata".into(),
            ));
        }
        if dim != 1 {
            terms.push(format!(
                "(({linear} / {}u) % {}u) * {}u",
                output_strides[pad + axis],
                dim,
                input_strides[axis]
            ));
        }
    }
    Ok(if terms.is_empty() {
        "0u".into()
    } else {
        terms.join(" + ")
    })
}

fn stable_key(value: &impl Hash) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod affine_view_tests {
    use super::*;
    #[test]
    fn signed_affine_view_lowers_without_unsigned_reinterpretation() {
        let view = AffineView {
            source_shape: Shape::from([4]),
            logical_shape: Shape::from([4]),
            strides: vec![-1],
            offset: 3,
        };
        let access = WgslViewAccess::new(&view).unwrap();
        assert!(access.expression("gid").contains("i32(gid)"));
    }

    #[test]
    fn signed_affine_view_rejects_unrepresentable_i32_intermediates() {
        let view = AffineView {
            source_shape: Shape::from([1]),
            logical_shape: Shape::from([0]),
            strides: vec![1],
            offset: i64::from(i32::MAX) + 1,
        };
        assert!(matches!(
            WgslViewAccess::new(&view),
            Err(WebGpuError::Unsupported(reason)) if reason.contains("i32 indexing")
        ));
    }
}
