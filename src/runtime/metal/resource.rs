//! Thread-confined safe Metal resources, caches, launch preflight, and events.
use super::{
    MetalBuffer, MetalDeviceInfo, MetalError, RenderedMetal,
    buffer::{BufferSnapshot, PhysicalBuffer},
    dispatch::{
        CopyRegion, Dispatch, KernelSemantics, LaunchGeometry, RawCommand, RawDevice, RawLibrary,
        RawPipeline, RawQueue,
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
/// Loaded Metal runtime entry point; creating it has no device side effects.
pub struct MetalRuntime {
    dispatch: Arc<dyn Dispatch>,
}

/// Typed outcome of non-submitting Metal device discovery.
///
/// Framework and symbol loading failures are returned as [`MetalError`] from
/// [`MetalRuntime::load`]. `NoDevices` means those steps succeeded but this
/// process could not enumerate a usable Metal device.
pub enum MetalDiscovery {
    Devices(Vec<MetalDevice>),
    NoDevices,
}

impl MetalRuntime {
    /// Dynamically loads Objective-C and Metal frameworks on macOS. No Apple
    /// SDK headers or link-time framework dependency is required.
    pub fn load() -> Result<Self, MetalError> {
        Ok(Self {
            dispatch: Arc::new(NativeDispatch::load()?),
        })
    }

    #[cfg(test)]
    pub(super) fn from_dispatch(dispatch: Arc<dyn Dispatch>) -> Self {
        Self { dispatch }
    }

    /// Enumerates devices in deterministic registry-ID/name order.
    pub fn devices(&self) -> Result<Vec<MetalDevice>, MetalError> {
        let mut raws = self.dispatch.devices()?;
        if raws.is_empty() {
            return Err(MetalError::NoDevices);
        }
        let mut devices = Vec::with_capacity(raws.len());
        while let Some(raw) = raws.pop() {
            let info = match self.dispatch.device_info(raw) {
                Ok(info) => info,
                Err(error) => {
                    self.dispatch.device_release(raw);
                    for raw in raws {
                        self.dispatch.device_release(raw);
                    }
                    return Err(error);
                }
            };
            let owner = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
            devices.push(MetalDevice {
                inner: Rc::new(DeviceInner {
                    dispatch: self.dispatch.clone(),
                    raw,
                    info,
                    owner,
                    closed: Cell::new(false),
                    thread_confined: PhantomData,
                }),
                cache_entries: Rc::new(RefCell::new(BTreeMap::new())),
            });
        }
        devices.sort_by(|left, right| {
            left.info()
                .registry_id
                .cmp(&right.info().registry_id)
                .then_with(|| left.info().name.cmp(&right.info().name))
        });
        Ok(devices)
    }

    /// Performs typed device discovery without conflating a loaded framework
    /// with a process-visible device. It creates no queue, pipeline, buffer,
    /// or command resource.
    pub fn discover(&self) -> Result<MetalDiscovery, MetalError> {
        match self.devices() {
            Ok(devices) => Ok(MetalDiscovery::Devices(devices)),
            Err(MetalError::NoDevices) => Ok(MetalDiscovery::NoDevices),
            Err(error) => Err(error),
        }
    }
}

pub(super) struct DeviceInner {
    pub(super) dispatch: Arc<dyn Dispatch>,
    pub(super) raw: RawDevice,
    pub(super) info: MetalDeviceInfo,
    pub(super) owner: u64,
    closed: Cell<bool>,
    thread_confined: PhantomData<Rc<()>>,
}

impl DeviceInner {
    pub(super) fn live(&self) -> Result<(), MetalError> {
        if self.closed.get() {
            Err(MetalError::Closed("device"))
        } else {
            Ok(())
        }
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.dispatch.device_release(self.raw);
        }
    }
}

#[derive(Clone)]
/// Thread-confined owned Metal device.
pub struct MetalDevice {
    inner: Rc<DeviceInner>,
    // Pipelines retain `DeviceInner`, not this outer cache owner. Keeping the
    // map alongside the cloneable device therefore shares one thread-confined
    // cache per discovered device without creating a resource cycle.
    cache_entries: Rc<RefCell<BTreeMap<String, Rc<MetalPipeline>>>>,
}

impl MetalDevice {
    /// Returns immutable discovered information without exposing a native handle.
    pub fn info(&self) -> &MetalDeviceInfo {
        &self.inner.info
    }

    /// Returns the stable Rust owner identity.
    pub fn owner_id(&self) -> u64 {
        self.inner.owner
    }

    /// Creates a retained command queue owned by this device.
    pub fn create_queue(&self) -> Result<MetalCommandQueue, MetalError> {
        self.inner.live()?;
        let raw = self
            .inner
            .dispatch
            .queue_create(self.inner.raw, self.inner.owner)?;
        Ok(MetalCommandQueue {
            device: self.inner.clone(),
            raw,
            closed: Cell::new(false),
        })
    }

    /// Allocates an untyped logical byte buffer.
    pub fn allocate(&self, bytes: usize) -> Result<MetalBuffer, MetalError> {
        MetalBuffer::allocate(self.inner.clone(), bytes, None)
    }

    /// Allocates a logical buffer with exact element-count and dtype metadata.
    pub fn allocate_typed(&self, elements: usize, dtype: DType) -> Result<MetalBuffer, MetalError> {
        let bytes = elements
            .checked_mul(dtype.itemsize())
            .ok_or(MetalError::Overflow)?;
        MetalBuffer::allocate(self.inner.clone(), bytes, Some(dtype))
    }

    pub(crate) fn allocate_static(
        &self,
        request: crate::runtime::static_schedule::StaticBufferAllocation,
    ) -> Result<MetalBuffer, MetalError> {
        if request.bytes
            != request
                .elements
                .checked_mul(request.dtype.itemsize())
                .ok_or(MetalError::Overflow)?
        {
            return Err(MetalError::InvalidBinding(
                "static allocation descriptor mismatch".into(),
            ));
        }
        MetalBuffer::allocate_with_handle(
            self.inner.clone(),
            request.bytes,
            Some(request.dtype),
            request.requires_native_handle,
        )
    }

    /// Compiles an already validated render artifact and preserves bounded
    /// native diagnostics as [`MetalError::Build`].
    pub fn compile(&self, rendered: &RenderedMetal) -> Result<MetalLibrary, MetalError> {
        self.inner.live()?;
        if rendered.capabilities != self.inner.info.capabilities {
            return Err(MetalError::InvalidBinding(
                "renderer/device capability identity mismatch".into(),
            ));
        }
        if rendered.source.as_bytes().contains(&0) {
            return Err(MetalError::InvalidArgument("interior NUL in source"));
        }
        let raw = self.inner.dispatch.library_compile(
            self.inner.raw,
            &rendered.source,
            self.inner.owner,
        )?;
        Ok(MetalLibrary {
            inner: Rc::new(LibraryInner {
                device: self.inner.clone(),
                raw,
                rendered: rendered.clone(),
                closed: Cell::new(false),
            }),
        })
    }

    /// Creates a process-local content-addressed pipeline cache.
    pub fn cache(&self) -> MetalCache {
        MetalCache {
            device: self.clone(),
            entries: self.cache_entries.clone(),
        }
    }
}

/// Thread-confined queue for checked copies and compute submission.
pub struct MetalCommandQueue {
    device: Rc<DeviceInner>,
    raw: RawQueue,
    closed: Cell<bool>,
}

impl MetalCommandQueue {
    fn live(&self) -> Result<(), MetalError> {
        self.device.live()?;
        if self.closed.get() {
            Err(MetalError::Closed("command queue"))
        } else {
            Ok(())
        }
    }

    /// Copies host bytes into a checked shared Metal buffer range.
    pub fn write(
        &self,
        buffer: &MetalBuffer,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), MetalError> {
        self.live()?;
        let snapshot = buffer.snapshot(&self.device, offset, bytes.len(), None)?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.device.dispatch.buffer_write(
            snapshot.raw().ok_or(MetalError::Bounds)?,
            offset,
            bytes,
            self.device.owner,
        )
    }

    /// Copies a checked shared Metal buffer range into host bytes.
    pub fn read(
        &self,
        buffer: &MetalBuffer,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), MetalError> {
        self.live()?;
        let snapshot = buffer.snapshot(&self.device, offset, bytes.len(), None)?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.device.dispatch.buffer_read(
            snapshot.raw().ok_or(MetalError::Bounds)?,
            offset,
            bytes,
            self.device.owner,
        )
    }

    /// Enqueues a checked device-to-device blit, returning no command for zero bytes.
    pub fn copy(
        &self,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        src_offset: usize,
        dst_offset: usize,
        bytes: usize,
    ) -> Result<Option<MetalCommand>, MetalError> {
        self.live()?;
        if let (Some(src_dtype), Some(dst_dtype)) = (src.dtype(), dst.dtype())
            && src_dtype != dst_dtype
        {
            return Err(MetalError::InvalidBinding("D2D copy dtype mismatch".into()));
        }
        let src = src.snapshot(&self.device, src_offset, bytes, None)?;
        let dst = dst.snapshot(&self.device, dst_offset, bytes, None)?;
        if bytes == 0 {
            return Ok(None);
        }
        let raw = self.device.dispatch.buffer_copy(
            self.raw,
            src.raw().ok_or(MetalError::Bounds)?,
            dst.raw().ok_or(MetalError::Bounds)?,
            CopyRegion {
                src_offset,
                dst_offset,
                bytes,
            },
            self.device.owner,
        )?;
        Ok(Some(MetalCommand::new(
            self.device.clone(),
            raw,
            vec![src, dst],
            0,
        )))
    }
}

impl Drop for MetalCommandQueue {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.device
                .dispatch
                .queue_release(self.raw, self.device.owner);
        }
    }
}

struct LibraryInner {
    device: Rc<DeviceInner>,
    raw: RawLibrary,
    rendered: RenderedMetal,
    closed: Cell<bool>,
}

impl Drop for LibraryInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.device
                .dispatch
                .library_release(self.raw, self.device.owner);
        }
    }
}

/// Owned native library compiled from one immutable render artifact.
pub struct MetalLibrary {
    inner: Rc<LibraryInner>,
}

impl MetalLibrary {
    /// Returns the render artifact used to build this library.
    pub fn rendered(&self) -> &RenderedMetal {
        &self.inner.rendered
    }

    /// Resolves the artifact entry point and creates a compute pipeline.
    pub fn create_pipeline(&self) -> Result<MetalPipeline, MetalError> {
        self.inner.device.live()?;
        let (raw, max_total_threads) = self.inner.device.dispatch.pipeline_create(
            self.inner.device.raw,
            self.inner.raw,
            &self.inner.rendered.entry,
            self.inner.device.owner,
        )?;
        let inner = Rc::new(PipelineInner {
            library: self.inner.clone(),
            raw,
            max_total_threads,
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
        Ok(MetalPipeline { inner })
    }
}

struct PipelineInner {
    library: Rc<LibraryInner>,
    raw: RawPipeline,
    max_total_threads: usize,
    closed: Cell<bool>,
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            let device = &self.library.device;
            device
                .dispatch
                .unregister_kernel_semantics(device.owner, self.raw);
            device.dispatch.pipeline_release(self.raw, device.owner);
        }
    }
}

/// Owned compute pipeline retaining its source library and render contract.
pub struct MetalPipeline {
    inner: Rc<PipelineInner>,
}

impl MetalPipeline {
    /// Returns the native per-threadgroup thread limit.
    pub fn max_total_threads(&self) -> usize {
        self.inner.max_total_threads
    }

    /// Returns the immutable checked render artifact.
    pub fn rendered(&self) -> &RenderedMetal {
        &self.inner.library.rendered
    }

    /// Validates and enqueues one static compute launch.
    pub fn launch<'a>(
        &'a self,
        queue: &'a MetalCommandQueue,
        bindings: &[&'a MetalBuffer],
        local_size: usize,
    ) -> Result<Option<MetalCommand>, MetalError> {
        queue.live()?;
        let device = &self.inner.library.device;
        if !Rc::ptr_eq(device, &queue.device) {
            return Err(MetalError::OwnerMismatch);
        }
        if self.inner.closed.get() {
            return Err(MetalError::Closed("compute pipeline"));
        }
        let rendered = &self.inner.library.rendered;
        if rendered.transaction.is_some() {
            return Err(MetalError::InvalidArgument(
                "guarded kernel requires transactional launch",
            ));
        }
        if bindings.len() != rendered.buffers.len() {
            return Err(MetalError::InvalidBinding(
                "Metal launch binding count mismatch".into(),
            ));
        }
        if local_size == 0 || local_size > self.inner.max_total_threads {
            return Err(MetalError::InvalidArgument(
                "local size exceeds pipeline thread limit",
            ));
        }
        if rendered.extent > u32::MAX as usize {
            return Err(MetalError::Unsupported(
                "extent exceeds the static uint thread-index ABI".into(),
            ));
        }
        let snapshots = bindings
            .iter()
            .zip(&rendered.buffers)
            .map(|(binding, abi)| {
                binding.snapshot(
                    device,
                    0,
                    abi.elements
                        .checked_mul(abi.dtype.itemsize())
                        .ok_or(MetalError::Overflow)?,
                    Some(abi.dtype),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if rendered.extent == 0 {
            return Ok(None);
        }
        let global = rendered
            .extent
            .checked_add(local_size - 1)
            .ok_or(MetalError::Overflow)?
            / local_size
            * local_size;
        let raws = snapshots
            .iter()
            .map(|snapshot| snapshot.raw().ok_or(MetalError::Bounds))
            .collect::<Result<Vec<_>, _>>()?;
        let raw = device.dispatch.launch(
            queue.raw,
            self.inner.raw,
            &raws,
            LaunchGeometry {
                extent: u64::try_from(rendered.extent).map_err(|_| MetalError::Overflow)?,
                extent_index: raws.len(),
                global,
                local: local_size,
            },
            device.owner,
        )?;
        Ok(Some(MetalCommand::new(
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
        queue: &'a MetalCommandQueue,
        bindings: &'a [&'a MetalBuffer],
        local_size: usize,
    ) -> Result<MetalTransaction<'a>, MetalError> {
        let rendered = &self.inner.library.rendered;
        let transaction = rendered
            .transaction
            .as_ref()
            .ok_or(MetalError::InvalidArgument("kernel is not transactional"))?;
        queue.live()?;
        let device = &self.inner.library.device;
        if !Rc::ptr_eq(device, &queue.device) {
            return Err(MetalError::OwnerMismatch);
        }
        if self.inner.closed.get() {
            return Err(MetalError::Closed("compute pipeline"));
        }
        if bindings.len() != rendered.buffers.len() {
            return Err(MetalError::InvalidBinding(
                "transaction binding count mismatch".into(),
            ));
        }
        if local_size == 0 || local_size > self.inner.max_total_threads {
            return Err(MetalError::InvalidArgument(
                "local size exceeds pipeline thread limit",
            ));
        }
        let snapshots = bindings
            .iter()
            .zip(&rendered.buffers)
            .map(|(binding, abi)| {
                binding.snapshot(
                    device,
                    0,
                    abi.elements
                        .checked_mul(abi.dtype.itemsize())
                        .ok_or(MetalError::Overflow)?,
                    Some(abi.dtype),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = bindings[transaction.output_abi_index];
        let base_generation = snapshots[transaction.output_abi_index].generation();
        let candidate = output.candidate()?;
        if rendered.extent == 0 {
            return Ok(MetalTransaction {
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
        if rendered.extent > u32::MAX as usize {
            return Err(MetalError::Unsupported(
                "extent exceeds the transactional uint status ABI".into(),
            ));
        }
        let status = MetalBuffer::allocate(device.clone(), 4, None)?;
        queue.write(&status, 0, &CLEAN_STATUS.to_le_bytes())?;
        let status_snapshot = status.snapshot(device, 0, 4, None)?;
        let mut raws = Vec::with_capacity(snapshots.len() + 1);
        for (index, snapshot) in snapshots.iter().enumerate() {
            let raw = if index == transaction.output_abi_index {
                candidate.raw
            } else {
                snapshot.raw()
            };
            raws.push(raw.ok_or(MetalError::Bounds)?);
        }
        raws.push(status_snapshot.raw().ok_or(MetalError::Bounds)?);
        let global = rendered
            .extent
            .checked_add(local_size - 1)
            .ok_or(MetalError::Overflow)?
            / local_size
            * local_size;
        let raw = device.dispatch.launch(
            queue.raw,
            self.inner.raw,
            &raws,
            LaunchGeometry {
                extent: rendered.extent as u64,
                extent_index: rendered.buffers.len(),
                global,
                local: local_size,
            },
            device.owner,
        )?;
        let retained = snapshots
            .iter()
            .map(|snapshot| snapshot.physical.clone())
            .chain([candidate.clone(), status_snapshot.physical.clone()])
            .collect();
        Ok(MetalTransaction {
            pipeline: self,
            queue,
            output,
            snapshots,
            base_generation,
            candidate,
            status: Some(status),
            command: Some(MetalCommand {
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
pub struct MetalTransaction<'a> {
    pipeline: &'a MetalPipeline,
    queue: &'a MetalCommandQueue,
    output: &'a MetalBuffer,
    snapshots: Vec<BufferSnapshot>,
    base_generation: u64,
    candidate: Rc<PhysicalBuffer>,
    status: Option<MetalBuffer>,
    command: Option<MetalCommand>,
}

impl MetalTransaction<'_> {
    /// Reports command readiness without reading status or changing visibility.
    pub fn query(&self) -> Result<bool, MetalError> {
        match &self.command {
            Some(command) => command.query(),
            None => Ok(true),
        }
    }

    /// Waits, reconstructs any bounded fault, and atomically commits only success.
    pub fn collect(mut self) -> Result<MetalCompletion, MetalError> {
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
                    .ok_or_else(|| MetalError::InvalidBinding("status absent".into()))?,
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
                return Err(MetalError::IntegerFault {
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
        Ok(MetalCompletion {
            extent,
            retained_resources,
        })
    }

    /// Alias for consuming collection when only success or failure is needed.
    pub fn wait(self) -> Result<(), MetalError> {
        self.collect().map(|_| ())
    }

    fn read_rhs(
        &self,
        transaction: &super::MetalTransactionAbi,
        guard: &super::MetalGuard,
        logical: usize,
    ) -> Result<i64, MetalError> {
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
                .ok_or_else(|| MetalError::InvalidBinding("detail buffer absent".into()))?;
            let abi = &self.pipeline.rendered().buffers[position];
            if abi.dtype != dtype {
                return Err(MetalError::InvalidBinding("detail dtype mismatch".into()));
            }
            let offset = logical_offset(arg, logical)?;
            let mut bytes = vec![0u8; dtype.itemsize()];
            let byte_offset = offset
                .checked_mul(bytes.len())
                .ok_or(MetalError::Overflow)?;
            let snapshot = &self.snapshots[position];
            self.pipeline.inner.library.device.dispatch.buffer_read(
                snapshot.raw().ok_or(MetalError::Bounds)?,
                byte_offset,
                &mut bytes,
                self.pipeline.inner.library.device.owner,
            )?;
            decode_detail_scalar(dtype, &bytes)
        })?;
        Ok(match guard.dtype {
            DType::I32 => value.as_i64(),
            DType::U32 => value.as_u64().min(i64::MAX as u64) as i64,
            _ => return Err(MetalError::InvalidBinding("guard dtype mismatch".into())),
        })
    }
}

fn decode_detail_scalar(dtype: DType, bytes: &[u8]) -> Result<crate::Scalar, MetalError> {
    Ok(match dtype {
        DType::Bool => crate::Scalar::Bool(bytes == [1]),
        DType::I32 => crate::Scalar::I(i32::from_le_bytes(
            bytes.try_into().map_err(|_| MetalError::Bounds)?,
        ) as i64),
        DType::U32 => crate::Scalar::U(u32::from_le_bytes(
            bytes.try_into().map_err(|_| MetalError::Bounds)?,
        ) as u64),
        _ => return Err(MetalError::InvalidBinding("detail storage dtype".into())),
    })
}

/// Non-cloneable pending command. It retains every submitted physical buffer
/// and consuming [`Self::collect`] yields a completed token.
pub struct MetalCommand {
    device: Rc<DeviceInner>,
    raw: Option<RawCommand>,
    snapshots: Vec<BufferSnapshot>,
    retained: Vec<Rc<PhysicalBuffer>>,
    extent: usize,
}

impl MetalCommand {
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

    /// Observes readiness without changing ownership or waiting.
    pub fn query(&self) -> Result<bool, MetalError> {
        self.device.live()?;
        self.device.dispatch.command_query(
            self.raw.ok_or(MetalError::Closed("command"))?,
            self.device.owner,
        )
    }

    /// Waits, validates submitted physical generations, and consumes the
    /// pending command regardless of success.
    pub fn collect(mut self) -> Result<MetalCompletion, MetalError> {
        let raw = self.raw.take().ok_or(MetalError::Closed("command"))?;
        let result = self.device.dispatch.command_wait(raw, self.device.owner);
        self.device.dispatch.command_release(raw, self.device.owner);
        result?;
        for snapshot in &self.snapshots {
            snapshot.validate_current()?;
        }
        Ok(MetalCompletion {
            extent: self.extent,
            retained_resources: self.retained.len(),
        })
    }
}

impl Drop for MetalCommand {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // A dropped pending token still owns the physical generations used
            // by the command. Best-effort completion keeps those allocations
            // alive until Metal has stopped accessing them.
            let _ = self.device.dispatch.command_wait(raw, self.device.owner);
            self.device.dispatch.command_release(raw, self.device.owner);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Completed command metadata returned by consuming collection.
pub struct MetalCompletion {
    /// Logical kernel extent, or zero for a D2D command.
    pub extent: usize,
    /// Number of physical resources retained through completion.
    pub retained_resources: usize,
}

/// Thread-confined process-local content-addressed pipeline cache.
pub struct MetalCache {
    device: MetalDevice,
    entries: Rc<RefCell<BTreeMap<String, Rc<MetalPipeline>>>>,
}

impl MetalCache {
    /// Returns the number of compiled pipeline identities.
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    /// Reports whether no pipeline has been compiled.
    pub fn is_empty(&self) -> bool {
        self.entries.borrow().is_empty()
    }

    /// Returns an existing pipeline or compiles and inserts it atomically for
    /// this single-threaded cache owner.
    pub fn load(&self, rendered: &RenderedMetal) -> Result<Rc<MetalPipeline>, MetalError> {
        if rendered.capabilities != self.device.info().capabilities {
            return Err(MetalError::InvalidBinding(
                "renderer/device capability identity mismatch".into(),
            ));
        }
        let key = stable_key(&(
            rendered.cache_key.as_str(),
            &self.device.info().capabilities,
            self.device.info().registry_id,
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
