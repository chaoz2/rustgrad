//! Thread-confined RAII resources, checked submission, and pipeline caching.
use super::{
    RenderedWgsl, WebGpuAdapterInfo, WebGpuBuffer, WebGpuError,
    buffer::{BufferSnapshot, PhysicalBuffer},
    dispatch::{
        CopyRegion, Dispatch, KernelSemantics, LaunchGeometry, RawAdapter, RawCommand, RawDevice,
        RawInstance, RawPipeline, RawQueue, RawShader,
    },
    ffi::NativeDispatch,
    transaction::{CLEAN_STATUS, detail_rhs_at, logical_offset},
};
use crate::DType;
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
/// Dynamically loaded or injected WebGPU entry point.
pub struct WebGpuRuntime {
    dispatch: Arc<dyn Dispatch>,
}

impl WebGpuRuntime {
    /// Loads a candidate C library. An unpinned callback ABI is rejected
    /// structurally before any native object or callback is created.
    pub fn load() -> Result<Self, WebGpuError> {
        Ok(Self {
            dispatch: Arc::new(NativeDispatch::load()?),
        })
    }

    #[cfg(test)]
    pub(super) fn from_dispatch(dispatch: Arc<dyn Dispatch>) -> Self {
        Self { dispatch }
    }

    /// Creates an owned instance. No native handle is exposed.
    pub fn create_instance(&self) -> Result<WebGpuInstance, WebGpuError> {
        let raw = self.dispatch.instance_create()?;
        Ok(WebGpuInstance {
            inner: Rc::new(InstanceInner {
                dispatch: self.dispatch.clone(),
                raw,
                closed: Cell::new(false),
                thread_confined: PhantomData,
            }),
        })
    }
}

struct InstanceInner {
    dispatch: Arc<dyn Dispatch>,
    raw: RawInstance,
    closed: Cell<bool>,
    thread_confined: PhantomData<Rc<()>>,
}

impl InstanceInner {
    fn live(&self) -> Result<(), WebGpuError> {
        if self.closed.get() {
            Err(WebGpuError::Closed("instance"))
        } else {
            Ok(())
        }
    }
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.dispatch.instance_release(self.raw);
        }
    }
}

#[derive(Clone)]
/// Owned thread-confined WebGPU instance.
pub struct WebGpuInstance {
    inner: Rc<InstanceInner>,
}

impl WebGpuInstance {
    /// Enumerates adapters in stable backend/vendor/device/name/driver order.
    pub fn adapters(&self) -> Result<Vec<WebGpuAdapter>, WebGpuError> {
        self.inner.live()?;
        let mut raws = self.inner.dispatch.adapters(self.inner.raw)?;
        if raws.is_empty() {
            return Err(WebGpuError::NoAdapters);
        }
        let mut adapters = Vec::with_capacity(raws.len());
        while let Some(raw) = raws.pop() {
            let info = match self.inner.dispatch.adapter_info(raw) {
                Ok(info) => info,
                Err(error) => {
                    self.inner.dispatch.adapter_release(raw);
                    for raw in raws {
                        self.inner.dispatch.adapter_release(raw);
                    }
                    return Err(error);
                }
            };
            adapters.push(WebGpuAdapter {
                inner: Rc::new(AdapterInner {
                    instance: self.inner.clone(),
                    raw,
                    info,
                    closed: Cell::new(false),
                }),
            });
        }
        adapters.sort_by(|left, right| {
            left.info()
                .backend
                .cmp(&right.info().backend)
                .then_with(|| left.info().vendor.cmp(&right.info().vendor))
                .then_with(|| left.info().device.cmp(&right.info().device))
                .then_with(|| left.info().name.cmp(&right.info().name))
                .then_with(|| left.info().driver.cmp(&right.info().driver))
        });
        Ok(adapters)
    }
}

struct AdapterInner {
    instance: Rc<InstanceInner>,
    raw: RawAdapter,
    info: WebGpuAdapterInfo,
    closed: Cell<bool>,
}

impl AdapterInner {
    fn live(&self) -> Result<(), WebGpuError> {
        self.instance.live()?;
        if self.closed.get() {
            Err(WebGpuError::Closed("adapter"))
        } else {
            Ok(())
        }
    }
}

impl Drop for AdapterInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.instance.dispatch.adapter_release(self.raw);
        }
    }
}

#[derive(Clone)]
/// Owned adapter retaining its instance.
pub struct WebGpuAdapter {
    inner: Rc<AdapterInner>,
}

impl WebGpuAdapter {
    /// Returns immutable handle-free adapter metadata.
    pub fn info(&self) -> &WebGpuAdapterInfo {
        &self.inner.info
    }

    /// Requests one logical device with exactly the advertised safe subset.
    pub fn request_device(&self) -> Result<WebGpuDevice, WebGpuError> {
        self.inner.live()?;
        let owner = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
        let raw = self
            .inner
            .instance
            .dispatch
            .device_create(self.inner.raw, owner)?;
        Ok(WebGpuDevice {
            inner: Rc::new(DeviceInner {
                dispatch: self.inner.instance.dispatch.clone(),
                adapter: self.inner.clone(),
                raw,
                info: self.inner.info.clone(),
                owner,
                closed: Cell::new(false),
                thread_confined: PhantomData,
            }),
            cache_entries: Rc::new(RefCell::new(BTreeMap::new())),
        })
    }
}

pub(super) struct DeviceInner {
    pub(super) dispatch: Arc<dyn Dispatch>,
    adapter: Rc<AdapterInner>,
    pub(super) raw: RawDevice,
    pub(super) info: WebGpuAdapterInfo,
    pub(super) owner: u64,
    closed: Cell<bool>,
    thread_confined: PhantomData<Rc<()>>,
}

impl DeviceInner {
    pub(super) fn live(&self) -> Result<(), WebGpuError> {
        self.adapter.live()?;
        if self.closed.get() {
            Err(WebGpuError::Closed("device"))
        } else {
            Ok(())
        }
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.dispatch.device_release(self.raw, self.owner);
        }
    }
}

#[derive(Clone)]
/// Owned thread-confined device retaining its adapter and instance.
pub struct WebGpuDevice {
    inner: Rc<DeviceInner>,
    // Pipelines retain `DeviceInner`, not this cache owner. Cloned logical
    // devices therefore share one thread-confined cache without a resource
    // cycle or cross-device leakage.
    cache_entries: Rc<RefCell<BTreeMap<String, Rc<WebGpuPipeline>>>>,
}

impl WebGpuDevice {
    /// Returns immutable handle-free adapter/device metadata.
    pub fn info(&self) -> &WebGpuAdapterInfo {
        &self.inner.info
    }
    /// Returns the stable Rust owner identity, never a native handle.
    pub fn owner_id(&self) -> u64 {
        self.inner.owner
    }

    /// Creates an owned queue retained by this device.
    pub fn create_queue(&self) -> Result<WebGpuQueue, WebGpuError> {
        self.inner.live()?;
        let raw = self
            .inner
            .dispatch
            .queue_create(self.inner.raw, self.inner.owner)?;
        Ok(WebGpuQueue {
            inner: Rc::new(QueueInner {
                device: self.inner.clone(),
                raw,
                closed: Cell::new(false),
            }),
        })
    }

    /// Allocates an untyped logical byte buffer.
    pub fn allocate(&self, bytes: usize) -> Result<WebGpuBuffer, WebGpuError> {
        WebGpuBuffer::allocate(self.inner.clone(), bytes, None)
    }

    /// Allocates an exact element-count and dtype buffer.
    pub fn allocate_typed(
        &self,
        elements: usize,
        dtype: DType,
    ) -> Result<WebGpuBuffer, WebGpuError> {
        let bytes = elements
            .checked_mul(dtype.itemsize())
            .ok_or(WebGpuError::Overflow)?;
        WebGpuBuffer::allocate(self.inner.clone(), bytes, Some(dtype))
    }

    /// Compiles one validated immutable WGSL artifact.
    pub fn compile(&self, rendered: &RenderedWgsl) -> Result<WebGpuShader, WebGpuError> {
        self.inner.live()?;
        rendered.validate_artifact()?;
        if rendered.capabilities != self.inner.info.capabilities {
            return Err(WebGpuError::InvalidBinding(
                "renderer/device capability identity mismatch".into(),
            ));
        }
        let raw = match self.inner.dispatch.shader_create(
            self.inner.raw,
            &rendered.source,
            self.inner.owner,
        ) {
            Err(WebGpuError::Build { diagnostic }) => {
                return Err(WebGpuError::Build {
                    diagnostic: bounded_diagnostic(diagnostic),
                });
            }
            result => result?,
        };
        Ok(WebGpuShader {
            inner: Rc::new(ShaderInner {
                device: self.inner.clone(),
                raw,
                rendered: rendered.clone(),
                closed: Cell::new(false),
            }),
        })
    }

    /// Creates an empty process-local content-addressed pipeline cache.
    pub fn cache(&self) -> WebGpuCache {
        WebGpuCache {
            device: self.clone(),
            entries: self.cache_entries.clone(),
        }
    }
}

struct QueueInner {
    device: Rc<DeviceInner>,
    raw: RawQueue,
    closed: Cell<bool>,
}

impl QueueInner {
    fn live(&self) -> Result<(), WebGpuError> {
        self.device.live()?;
        if self.closed.get() {
            Err(WebGpuError::Closed("queue"))
        } else {
            Ok(())
        }
    }
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.device
                .dispatch
                .queue_release(self.raw, self.device.owner);
        }
    }
}

#[derive(Clone)]
/// Owned queue used for checked transfers and compute submission.
pub struct WebGpuQueue {
    inner: Rc<QueueInner>,
}

impl WebGpuQueue {
    /// Copies host bytes into a checked logical buffer range.
    pub fn write(
        &self,
        buffer: &WebGpuBuffer,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), WebGpuError> {
        self.inner.live()?;
        let snapshot = buffer.snapshot(&self.inner.device, offset, bytes.len(), None)?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.inner.device.dispatch.buffer_write(
            self.inner.raw,
            snapshot.raw().ok_or(WebGpuError::Bounds)?,
            offset,
            bytes,
            self.inner.device.owner,
        )
    }

    /// Copies a checked logical buffer range into host bytes.
    pub fn read(
        &self,
        buffer: &WebGpuBuffer,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), WebGpuError> {
        self.inner.live()?;
        let snapshot = buffer.snapshot(&self.inner.device, offset, bytes.len(), None)?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.inner.device.dispatch.buffer_read(
            snapshot.raw().ok_or(WebGpuError::Bounds)?,
            offset,
            bytes,
            self.inner.device.owner,
        )
    }

    /// Encodes an aligned native D2D copy. WebGPU mandates four-byte regions.
    pub fn copy(
        &self,
        src: &WebGpuBuffer,
        dst: &WebGpuBuffer,
        src_offset: usize,
        dst_offset: usize,
        bytes: usize,
    ) -> Result<Option<WebGpuCommand>, WebGpuError> {
        self.inner.live()?;
        if let (Some(src_dtype), Some(dst_dtype)) = (src.dtype(), dst.dtype())
            && src_dtype != dst_dtype
        {
            return Err(WebGpuError::InvalidBinding(
                "D2D copy dtype mismatch".into(),
            ));
        }
        let src = src.snapshot(&self.inner.device, src_offset, bytes, None)?;
        let dst = dst.snapshot(&self.inner.device, dst_offset, bytes, None)?;
        if bytes == 0 {
            return Ok(None);
        }
        if src.same_logical(&dst) {
            return Err(WebGpuError::InvalidArgument(
                "D2D source and destination are the same buffer",
            ));
        }
        if !src_offset.is_multiple_of(4)
            || !dst_offset.is_multiple_of(4)
            || !bytes.is_multiple_of(4)
        {
            return Err(WebGpuError::InvalidArgument(
                "D2D copy region is not four-byte aligned",
            ));
        }
        let raw = self.inner.device.dispatch.buffer_copy(
            self.inner.raw,
            src.raw().ok_or(WebGpuError::Bounds)?,
            dst.raw().ok_or(WebGpuError::Bounds)?,
            CopyRegion {
                src_offset,
                dst_offset,
                bytes,
            },
            self.inner.device.owner,
        )?;
        Ok(Some(WebGpuCommand::new(
            self.inner.device.clone(),
            raw,
            vec![src, dst],
            0,
        )))
    }
}

struct ShaderInner {
    device: Rc<DeviceInner>,
    raw: RawShader,
    rendered: RenderedWgsl,
    closed: Cell<bool>,
}

impl Drop for ShaderInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.device
                .dispatch
                .shader_release(self.raw, self.device.owner);
        }
    }
}

/// Owned shader module retaining source and device.
pub struct WebGpuShader {
    inner: Rc<ShaderInner>,
}

impl WebGpuShader {
    /// Returns the immutable WGSL artifact used to create the module.
    pub fn rendered(&self) -> &RenderedWgsl {
        &self.inner.rendered
    }

    /// Creates a compute pipeline and retains semantic mock metadata.
    pub fn create_pipeline(&self) -> Result<WebGpuPipeline, WebGpuError> {
        self.inner.device.live()?;
        let raw = self.inner.device.dispatch.pipeline_create(
            self.inner.device.raw,
            self.inner.raw,
            &self.inner.rendered.entry,
            self.inner.rendered.buffers.len()
                + usize::from(self.inner.rendered.transaction.is_some()),
            self.inner.device.owner,
        )?;
        let inner = Rc::new(PipelineInner {
            shader: self.inner.clone(),
            raw,
            closed: Cell::new(false),
        });
        self.inner.device.dispatch.register_kernel_semantics(
            self.inner.device.owner,
            raw,
            Arc::new(KernelSemantics {
                buffers: self.inner.rendered.buffers.clone(),
                extent: self.inner.rendered.extent,
                program: self.inner.rendered.semantic_program.clone(),
                transaction: self.inner.rendered.transaction.clone(),
            }),
        )?;
        Ok(WebGpuPipeline { inner })
    }
}

struct PipelineInner {
    shader: Rc<ShaderInner>,
    raw: RawPipeline,
    closed: Cell<bool>,
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            let device = &self.shader.device;
            device
                .dispatch
                .unregister_kernel_semantics(device.owner, self.raw);
            device.dispatch.pipeline_release(self.raw, device.owner);
        }
    }
}

/// Owned compute pipeline retaining its shader module and launch contract.
pub struct WebGpuPipeline {
    inner: Rc<PipelineInner>,
}

impl WebGpuPipeline {
    /// Returns the immutable checked render artifact.
    pub fn rendered(&self) -> &RenderedWgsl {
        &self.inner.shader.rendered
    }

    /// Validates all ABI, owner, dtype, extent, and geometry inputs before submission.
    pub fn launch(
        &self,
        queue: &WebGpuQueue,
        bindings: &[&WebGpuBuffer],
    ) -> Result<Option<WebGpuCommand>, WebGpuError> {
        queue.inner.live()?;
        let device = &self.inner.shader.device;
        if !Rc::ptr_eq(device, &queue.inner.device) {
            return Err(WebGpuError::OwnerMismatch);
        }
        if self.inner.closed.get() {
            return Err(WebGpuError::Closed("pipeline"));
        }
        let rendered = &self.inner.shader.rendered;
        if rendered.transaction.is_some() {
            return Err(WebGpuError::InvalidArgument(
                "guarded kernel requires transactional launch",
            ));
        }
        if bindings.len() != rendered.buffers.len() {
            return Err(WebGpuError::InvalidBinding(
                "WebGPU launch binding count mismatch".into(),
            ));
        }
        let snapshots = bindings
            .iter()
            .zip(&rendered.buffers)
            .map(|(binding, abi)| {
                binding.snapshot(device, 0, abi.logical_bytes()?, Some(abi.dtype))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if rendered.extent == 0 {
            return Ok(None);
        }
        let local = rendered.local_size;
        let extent = u32::try_from(rendered.extent).map_err(|_| WebGpuError::Overflow)?;
        let workgroups = extent.checked_add(local - 1).ok_or(WebGpuError::Overflow)? / local;
        if workgroups
            > device
                .info
                .capabilities
                .max_compute_workgroups_per_dimension
        {
            return Err(WebGpuError::Unsupported(
                "launch exceeds adapter workgroup-count limit".into(),
            ));
        }
        let raws = snapshots
            .iter()
            .map(|snapshot| snapshot.raw().ok_or(WebGpuError::Bounds))
            .collect::<Result<Vec<_>, _>>()?;
        let raw = device.dispatch.launch(
            queue.inner.raw,
            self.inner.raw,
            &raws,
            LaunchGeometry {
                extent,
                workgroups,
                local,
                extent_binding: rendered.buffers.len(),
                status_binding: None,
            },
            device.owner,
        )?;
        Ok(Some(WebGpuCommand::new(
            device.clone(),
            raw,
            snapshots,
            rendered.extent,
        )))
    }

    /// Submits a guarded integer kernel into a provisional physical output.
    /// Only consuming a clean transaction may make that generation visible.
    pub fn launch_transactional<'a>(
        &'a self,
        queue: &'a WebGpuQueue,
        bindings: &'a [&'a WebGpuBuffer],
    ) -> Result<WebGpuTransaction<'a>, WebGpuError> {
        let rendered = &self.inner.shader.rendered;
        let transaction = rendered
            .transaction
            .as_ref()
            .ok_or(WebGpuError::InvalidArgument("kernel is not transactional"))?;
        queue.inner.live()?;
        let device = &self.inner.shader.device;
        if !Rc::ptr_eq(device, &queue.inner.device) {
            return Err(WebGpuError::OwnerMismatch);
        }
        if self.inner.closed.get() {
            return Err(WebGpuError::Closed("pipeline"));
        }
        if bindings.len() != rendered.buffers.len() {
            return Err(WebGpuError::InvalidBinding(
                "transaction binding count mismatch".into(),
            ));
        }
        if transaction.output_abi_index >= bindings.len()
            || !rendered.buffers[transaction.output_abi_index].mutable
        {
            return Err(WebGpuError::InvalidBinding(
                "transaction output binding mismatch".into(),
            ));
        }
        transaction.validate_launch(rendered.extent, transaction.output_abi_index)?;
        let local = rendered.local_size;
        let extent = u32::try_from(rendered.extent).map_err(|_| WebGpuError::Overflow)?;
        let workgroups = if extent == 0 {
            0
        } else {
            extent.checked_add(local - 1).ok_or(WebGpuError::Overflow)? / local
        };
        if workgroups
            > device
                .info
                .capabilities
                .max_compute_workgroups_per_dimension
        {
            return Err(WebGpuError::Unsupported(
                "launch exceeds adapter workgroup-count limit".into(),
            ));
        }
        let snapshots = bindings
            .iter()
            .zip(&rendered.buffers)
            .map(|(binding, abi)| {
                binding.snapshot(device, 0, abi.logical_bytes()?, Some(abi.dtype))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = bindings[transaction.output_abi_index];
        let base_generation = snapshots[transaction.output_abi_index].generation();
        if rendered.extent != 0 && snapshots.iter().any(|snapshot| snapshot.raw().is_none()) {
            return Err(WebGpuError::InvalidBinding(
                "nonempty transaction has an empty physical binding".into(),
            ));
        }

        // All public metadata and geometry have been validated before the
        // first provisional allocation or dispatch side effect.
        let candidate = output.candidate()?;
        if rendered.extent == 0 {
            return Ok(WebGpuTransaction {
                pipeline: self,
                queue,
                output,
                snapshots,
                base_generation,
                candidate,
                status: None,
                command: None,
            });
        }
        let status = WebGpuBuffer::allocate(device.clone(), 4, None)?;
        queue.write(&status, 0, &CLEAN_STATUS.to_le_bytes())?;
        let status_snapshot = status.snapshot(device, 0, 4, None)?;
        let mut raws = Vec::with_capacity(snapshots.len() + 1);
        for (index, snapshot) in snapshots.iter().enumerate() {
            let raw = if index == transaction.output_abi_index {
                candidate.raw
            } else {
                snapshot.raw()
            };
            raws.push(raw.ok_or(WebGpuError::Bounds)?);
        }
        raws.push(status_snapshot.raw().ok_or(WebGpuError::Bounds)?);
        let raw = device.dispatch.launch(
            queue.inner.raw,
            self.inner.raw,
            &raws,
            LaunchGeometry {
                extent,
                workgroups,
                local,
                extent_binding: rendered.buffers.len(),
                status_binding: Some(rendered.buffers.len() + 1),
            },
            device.owner,
        )?;
        let retained = snapshots
            .iter()
            .map(|snapshot| snapshot.physical.clone())
            .chain([candidate.clone(), status_snapshot.physical.clone()])
            .collect();
        Ok(WebGpuTransaction {
            pipeline: self,
            queue,
            output,
            snapshots,
            base_generation,
            candidate,
            status: Some(status),
            command: Some(WebGpuCommand {
                device: device.clone(),
                raw: Some(raw),
                snapshots: Vec::new(),
                retained,
                extent: rendered.extent,
            }),
        })
    }
}

/// Non-cloneable guarded launch retaining inputs, candidate, status, and command.
pub struct WebGpuTransaction<'a> {
    pipeline: &'a WebGpuPipeline,
    queue: &'a WebGpuQueue,
    output: &'a WebGpuBuffer,
    snapshots: Vec<BufferSnapshot>,
    base_generation: u64,
    candidate: Rc<PhysicalBuffer>,
    status: Option<WebGpuBuffer>,
    command: Option<WebGpuCommand>,
}

impl WebGpuTransaction<'_> {
    /// Reports command readiness without reading status or changing visibility.
    pub fn query(&self) -> Result<bool, WebGpuError> {
        match &self.command {
            Some(command) => command.query(),
            None => Ok(true),
        }
    }

    /// Waits, reconstructs any bounded fault, and commits only clean output.
    pub fn collect(mut self) -> Result<WebGpuCompletion, WebGpuError> {
        let extent = self.pipeline.rendered().extent;
        let retained_resources = self
            .command
            .as_ref()
            .map_or(self.snapshots.len() + 1, |command| command.retained.len());
        if let Some(command) = self.command.take() {
            command.collect()?;
            let mut bytes = [0u8; 4];
            self.queue.read(
                self.status
                    .as_ref()
                    .ok_or_else(|| WebGpuError::InvalidBinding("status absent".into()))?,
                0,
                &mut bytes,
            )?;
            let status = u32::from_le_bytes(bytes);
            if status != CLEAN_STATUS {
                let transaction = self.pipeline.rendered().transaction.as_ref().unwrap();
                let (index, guard) = transaction.decode(status)?;
                let count = if guard.operation.is_shift() {
                    Some(self.read_rhs(transaction, guard, index)?)
                } else {
                    None
                };
                return Err(WebGpuError::IntegerFault {
                    operation: guard.operation,
                    index,
                    count,
                    bits: usize::from(guard.dtype.bits()),
                });
            }
        }
        for snapshot in &self.snapshots {
            snapshot.validate_current()?;
        }
        self.output
            .commit_candidate(self.base_generation, self.candidate.clone())?;
        Ok(WebGpuCompletion {
            extent,
            retained_resources,
        })
    }

    /// Alias for consuming collection when completion metadata is unnecessary.
    pub fn wait(self) -> Result<(), WebGpuError> {
        self.collect().map(|_| ())
    }

    fn read_rhs(
        &self,
        transaction: &super::WebGpuTransactionAbi,
        guard: &super::WebGpuGuard,
        logical: usize,
    ) -> Result<i64, WebGpuError> {
        let value = detail_rhs_at(transaction, guard, logical, |arg, dtype, logical| {
            let buffer_id = match arg {
                crate::IndexValue::Buffer { buffer, .. }
                | crate::IndexValue::View { buffer, .. } => *buffer,
            };
            let position = self
                .pipeline
                .rendered()
                .buffers
                .iter()
                .position(|abi| abi.id == buffer_id)
                .ok_or_else(|| WebGpuError::InvalidBinding("detail buffer absent".into()))?;
            let abi = &self.pipeline.rendered().buffers[position];
            if abi.dtype != dtype {
                return Err(WebGpuError::InvalidBinding("detail dtype mismatch".into()));
            }
            let offset = logical_offset(arg, logical)?;
            let mut bytes = vec![0u8; dtype.itemsize()];
            let byte_offset = offset
                .checked_mul(bytes.len())
                .ok_or(WebGpuError::Overflow)?;
            let snapshot = &self.snapshots[position];
            self.pipeline.inner.shader.device.dispatch.buffer_read(
                snapshot.raw().ok_or(WebGpuError::Bounds)?,
                byte_offset,
                &mut bytes,
                self.pipeline.inner.shader.device.owner,
            )?;
            decode_detail_scalar(dtype, &bytes)
        })?;
        Ok(match guard.dtype {
            DType::I32 => value.as_i64(),
            DType::U32 => value.as_u64().min(i64::MAX as u64) as i64,
            _ => {
                return Err(WebGpuError::InvalidBinding("guard dtype mismatch".into()));
            }
        })
    }
}

fn decode_detail_scalar(dtype: DType, bytes: &[u8]) -> Result<crate::Scalar, WebGpuError> {
    Ok(match dtype {
        DType::Bool => crate::Scalar::Bool(bytes == [1]),
        DType::I32 => crate::Scalar::I(i32::from_le_bytes(
            bytes.try_into().map_err(|_| WebGpuError::Bounds)?,
        ) as i64),
        DType::U32 => crate::Scalar::U(u32::from_le_bytes(
            bytes.try_into().map_err(|_| WebGpuError::Bounds)?,
        ) as u64),
        _ => {
            return Err(WebGpuError::InvalidBinding(
                "detail storage dtype mismatch".into(),
            ));
        }
    })
}

/// Non-cloneable pending command retaining submitted physical generations.
pub struct WebGpuCommand {
    device: Rc<DeviceInner>,
    raw: Option<RawCommand>,
    snapshots: Vec<BufferSnapshot>,
    retained: Vec<Rc<PhysicalBuffer>>,
    extent: usize,
}

impl WebGpuCommand {
    fn new(
        device: Rc<DeviceInner>,
        raw: RawCommand,
        snapshots: Vec<BufferSnapshot>,
        extent: usize,
    ) -> Self {
        let retained = snapshots
            .iter()
            .map(|snapshot| snapshot.physical.clone())
            .collect();
        Self {
            device,
            raw: Some(raw),
            snapshots,
            retained,
            extent,
        }
    }

    /// Nonblocking readiness observation with no ownership transition.
    pub fn query(&self) -> Result<bool, WebGpuError> {
        self.device.live()?;
        self.device.dispatch.command_query(
            self.raw.ok_or(WebGpuError::Closed("command"))?,
            self.device.owner,
        )
    }

    /// Waits, revalidates every generation, and consumes the command.
    pub fn collect(mut self) -> Result<WebGpuCompletion, WebGpuError> {
        let raw = self.raw.take().ok_or(WebGpuError::Closed("command"))?;
        let result = self.device.dispatch.command_wait(raw, self.device.owner);
        self.device.dispatch.command_release(raw, self.device.owner);
        result?;
        for snapshot in &self.snapshots {
            snapshot.validate_current()?;
        }
        Ok(WebGpuCompletion {
            extent: self.extent,
            retained_resources: self.retained.len(),
        })
    }
}

impl Drop for WebGpuCommand {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            let _ = self.device.dispatch.command_wait(raw, self.device.owner);
            self.device.dispatch.command_release(raw, self.device.owner);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Handle-free completed command metadata.
pub struct WebGpuCompletion {
    /// Logical kernel extent, or zero for D2D.
    pub extent: usize,
    /// Number of physical resources retained through completion.
    pub retained_resources: usize,
}

/// Thread-confined content-addressed shader/pipeline cache.
pub struct WebGpuCache {
    device: WebGpuDevice,
    entries: Rc<RefCell<BTreeMap<String, Rc<WebGpuPipeline>>>>,
}

impl WebGpuCache {
    /// Returns the number of cached content identities.
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }
    /// Reports whether no shader/pipeline has been cached.
    pub fn is_empty(&self) -> bool {
        self.entries.borrow().is_empty()
    }

    /// Returns an existing pipeline or compiles and inserts one atomically for
    /// this thread-confined cache owner.
    pub fn load(&self, rendered: &RenderedWgsl) -> Result<Rc<WebGpuPipeline>, WebGpuError> {
        rendered.validate_artifact()?;
        if rendered.capabilities != self.device.info().capabilities {
            return Err(WebGpuError::InvalidBinding(
                "renderer/device capability identity mismatch".into(),
            ));
        }
        let key = stable_key(&(
            rendered.cache_key.as_str(),
            &self.device.info().capabilities,
            &self.device.info().backend,
            self.device.info().name.as_str(),
            self.device.info().vendor,
            self.device.info().device,
            self.device.info().driver.as_str(),
        ));
        if let Some(pipeline) = self.entries.borrow().get(&key) {
            return Ok(pipeline.clone());
        }
        let pipeline = Rc::new(self.device.compile(rendered)?.create_pipeline()?);
        self.entries.borrow_mut().insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

fn stable_key(value: &impl Hash) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn bounded_diagnostic(mut diagnostic: String) -> String {
    const MAX_BYTES: usize = 64 * 1024;
    if diagnostic.len() <= MAX_BYTES {
        return diagnostic;
    }
    let mut boundary = MAX_BYTES;
    while !diagnostic.is_char_boundary(boundary) {
        boundary -= 1;
    }
    diagnostic.truncate(boundary);
    diagnostic.push_str("…[truncated]");
    diagnostic
}
