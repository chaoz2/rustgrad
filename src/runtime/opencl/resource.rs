//! Safe thread-confined OpenCL resource ownership and launch validation.
use super::{
    BufferCopyRegion, BuildInfo, DeviceInfo, Dispatch, OpenClError, RawContext, RawDevice,
    RawEvent, RawKernel, RawPlatform, RawProgram, RawQueue, RenderedOpenCl,
    buffer::{BufferSnapshot, OpenClBuffer, PhysicalBuffer},
    dispatch::KernelSemantics,
    ffi::NativeDispatch,
    transaction::{CLEAN_STATUS, detail_rhs_at, logical_offset},
};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

const MAX_BUILD_LOG_BYTES: usize = 64 * 1024;
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct OpenClIcd {
    dispatch: Arc<dyn Dispatch>,
}

impl OpenClIcd {
    /// Dynamically loads the system OpenCL ICD without a compile-time SDK.
    pub fn load() -> Result<Self, OpenClError> {
        Ok(Self::from_dispatch(Arc::new(NativeDispatch::load()?)))
    }

    /// Installs a typed ICD implementation. This is the deterministic test and
    /// host-integration seam; production callers normally use [`Self::load`].
    pub fn from_dispatch(dispatch: Arc<dyn Dispatch>) -> Self {
        Self { dispatch }
    }

    pub fn platforms(&self) -> Result<Vec<OpenClPlatform>, OpenClError> {
        let raws = self.dispatch.platforms()?;
        if raws.is_empty() {
            return Err(OpenClError::NoPlatforms);
        }
        raws.into_iter()
            .map(|raw| {
                Ok(OpenClPlatform {
                    icd: self.clone(),
                    name: self.dispatch.platform_name(raw)?,
                    raw,
                })
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct OpenClPlatform {
    icd: OpenClIcd,
    raw: RawPlatform,
    name: String,
}

impl OpenClPlatform {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn devices(&self) -> Result<Vec<OpenClDevice>, OpenClError> {
        let raws = self.icd.dispatch.devices(self.raw)?;
        if raws.is_empty() {
            return Err(OpenClError::NoDevices);
        }
        raws.into_iter()
            .map(|raw| {
                Ok(OpenClDevice {
                    platform: self.clone(),
                    info: self.icd.dispatch.device_info(raw)?,
                    raw,
                })
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct OpenClDevice {
    platform: OpenClPlatform,
    raw: RawDevice,
    info: DeviceInfo,
}

impl OpenClDevice {
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn create_context(&self) -> Result<OpenClContext, OpenClError> {
        let owner = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
        let raw = self.platform.icd.dispatch.context_create(self.raw, owner)?;
        Ok(OpenClContext {
            inner: Rc::new(ContextInner {
                dispatch: self.platform.icd.dispatch.clone(),
                raw,
                owner,
                device: self.raw,
                device_info: self.info.clone(),
                closed: Cell::new(false),
            }),
        })
    }
}

pub(super) struct ContextInner {
    pub(super) dispatch: Arc<dyn Dispatch>,
    pub(super) raw: RawContext,
    pub(super) owner: u64,
    pub(super) device: RawDevice,
    pub(super) device_info: DeviceInfo,
    closed: Cell<bool>,
}

impl ContextInner {
    pub(super) fn live(&self) -> Result<(), OpenClError> {
        if self.closed.get() {
            Err(OpenClError::Closed("context"))
        } else {
            Ok(())
        }
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            let _ = self.dispatch.context_release(self.raw, self.owner);
        }
    }
}

/// An OpenCL context. It is intentionally thread-confined; its children retain
/// the context until their own cleanup has completed.
#[derive(Clone)]
pub struct OpenClContext {
    inner: Rc<ContextInner>,
}

impl OpenClContext {
    pub fn owner_id(&self) -> u64 {
        self.inner.owner
    }

    pub fn create_queue(&self) -> Result<OpenClQueue, OpenClError> {
        self.inner.live()?;
        let raw = self.inner.dispatch.queue_create(
            self.inner.raw,
            self.inner.device,
            self.inner.owner,
        )?;
        Ok(OpenClQueue {
            context: self.inner.clone(),
            raw,
            closed: Cell::new(false),
        })
    }

    /// Allocates device bytes. Zero bytes produce a logical sentinel and make
    /// no ICD call; such a buffer has no usable raw handle.
    pub fn allocate(&self, bytes: usize) -> Result<OpenClBuffer, OpenClError> {
        OpenClBuffer::allocate(self.inner.clone(), bytes, None)
    }

    /// Allocates a logical buffer with a checked storage dtype contract.
    pub fn allocate_typed(
        &self,
        elements: usize,
        dtype: crate::DType,
    ) -> Result<OpenClBuffer, OpenClError> {
        let bytes = elements
            .checked_mul(dtype.itemsize())
            .ok_or(OpenClError::Overflow)?;
        OpenClBuffer::allocate(self.inner.clone(), bytes, Some(dtype))
    }

    pub(crate) fn allocate_static(
        &self,
        request: crate::runtime::static_schedule::StaticBufferAllocation,
    ) -> Result<OpenClBuffer, OpenClError> {
        if request.bytes
            != request
                .elements
                .checked_mul(request.dtype.itemsize())
                .ok_or(OpenClError::Overflow)?
        {
            return Err(OpenClError::InvalidBinding(
                "static allocation descriptor mismatch".into(),
            ));
        }
        OpenClBuffer::allocate_with_handle(
            self.inner.clone(),
            request.bytes,
            Some(request.dtype),
            request.requires_native_handle,
        )
    }

    pub fn cache(&self) -> OpenClCache {
        OpenClCache {
            context: self.clone(),
            entries: RefCell::new(BTreeMap::new()),
        }
    }
}

pub struct OpenClQueue {
    context: Rc<ContextInner>,
    raw: RawQueue,
    closed: Cell<bool>,
}

impl OpenClQueue {
    fn live(&self) -> Result<(), OpenClError> {
        self.context.live()?;
        if self.closed.get() {
            Err(OpenClError::Closed("queue"))
        } else {
            Ok(())
        }
    }

    pub fn finish(&self) -> Result<(), OpenClError> {
        self.live()?;
        self.context
            .dispatch
            .queue_finish(self.raw, self.context.owner)
    }

    pub fn write(
        &self,
        buffer: &OpenClBuffer,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), OpenClError> {
        self.live()?;
        let snapshot = buffer.snapshot(&self.context, offset, bytes.len(), None)?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.context.dispatch.buffer_write(
            self.raw,
            snapshot.raw().ok_or(OpenClError::Bounds)?,
            offset,
            bytes,
            self.context.owner,
        )
    }

    pub fn read(
        &self,
        buffer: &OpenClBuffer,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), OpenClError> {
        self.live()?;
        let snapshot = buffer.snapshot(&self.context, offset, bytes.len(), None)?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.context.dispatch.buffer_read(
            self.raw,
            snapshot.raw().ok_or(OpenClError::Bounds)?,
            offset,
            bytes,
            self.context.owner,
        )
    }

    fn read_snapshot(
        &self,
        snapshot: &BufferSnapshot,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), OpenClError> {
        self.live()?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.context.dispatch.buffer_read(
            self.raw,
            snapshot.raw().ok_or(OpenClError::Bounds)?,
            offset,
            bytes,
            self.context.owner,
        )
    }

    pub fn copy(
        &self,
        src: &OpenClBuffer,
        dst: &OpenClBuffer,
        src_offset: usize,
        dst_offset: usize,
        bytes: usize,
    ) -> Result<Option<OpenClEvent>, OpenClError> {
        self.live()?;
        let src_snapshot = src.snapshot(&self.context, src_offset, bytes, None)?;
        let dst_snapshot = dst.snapshot(&self.context, dst_offset, bytes, None)?;
        if bytes == 0 {
            return Ok(None);
        }
        let raw = self.context.dispatch.buffer_copy(
            self.raw,
            src_snapshot.raw().ok_or(OpenClError::Bounds)?,
            dst_snapshot.raw().ok_or(OpenClError::Bounds)?,
            BufferCopyRegion {
                src_offset,
                dst_offset,
                bytes,
            },
            self.context.owner,
        )?;
        Ok(Some(OpenClEvent::new(
            self.context.clone(),
            raw,
            vec![src_snapshot.physical(), dst_snapshot.physical()],
        )))
    }
}

impl Drop for OpenClQueue {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            let _ = self
                .context
                .dispatch
                .queue_release(self.raw, self.context.owner);
        }
    }
}

struct ProgramInner {
    context: Rc<ContextInner>,
    raw: RawProgram,
    build: BuildInfo,
    closed: Cell<bool>,
}

impl Drop for ProgramInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            let _ = self
                .context
                .dispatch
                .program_release(self.raw, self.context.owner);
        }
    }
}

pub struct OpenClKernel {
    program: Rc<ProgramInner>,
    raw: RawKernel,
    rendered: RenderedOpenCl,
    cache_key: String,
    local_size: usize,
    closed: Cell<bool>,
}

impl OpenClKernel {
    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }

    pub fn build_info(&self) -> &BuildInfo {
        &self.program.build
    }

    pub fn rendered(&self) -> &RenderedOpenCl {
        &self.rendered
    }

    pub fn launch(
        &self,
        queue: &OpenClQueue,
        bindings: &[&OpenClBuffer],
    ) -> Result<Option<OpenClEvent>, OpenClError> {
        if self.rendered.transaction.is_some() {
            return Err(OpenClError::InvalidArgument(
                "guarded integer kernel requires transactional launch",
            ));
        }
        self.launch_direct(queue, bindings)
    }

    fn launch_direct(
        &self,
        queue: &OpenClQueue,
        bindings: &[&OpenClBuffer],
    ) -> Result<Option<OpenClEvent>, OpenClError> {
        queue.live()?;
        if self.closed.get() {
            return Err(OpenClError::Closed("kernel"));
        }
        let context = &self.program.context;
        if !Rc::ptr_eq(context, &queue.context) {
            return Err(OpenClError::OwnerMismatch);
        }
        if bindings.len() != self.rendered.buffers.len() {
            return Err(OpenClError::InvalidBinding(format!(
                "expected {} buffers, got {}",
                self.rendered.buffers.len(),
                bindings.len()
            )));
        }
        let snapshots = bindings
            .iter()
            .zip(&self.rendered.buffers)
            .enumerate()
            .map(|(index, (binding, abi))| {
                let bytes = abi
                    .elements
                    .checked_mul(abi.dtype.itemsize())
                    .ok_or(OpenClError::Overflow)?;
                binding
                    .snapshot(context, 0, bytes, Some(abi.dtype))
                    .map_err(|error| match error {
                        OpenClError::OwnerMismatch => OpenClError::OwnerMismatch,
                        other => OpenClError::InvalidBinding(format!(
                            "buffer {index} failed validation: {other}"
                        )),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if self.rendered.extent == 0 {
            return Ok(None);
        }
        for (index, snapshot) in snapshots.iter().enumerate() {
            context.dispatch.kernel_arg_buffer(
                self.raw,
                u32::try_from(index).map_err(|_| OpenClError::Overflow)?,
                snapshot.raw().ok_or_else(|| {
                    OpenClError::InvalidBinding("nonzero launch uses zero buffer".into())
                })?,
                context.owner,
            )?;
        }
        let extent_index = u32::try_from(bindings.len()).map_err(|_| OpenClError::Overflow)?;
        context.dispatch.kernel_arg_u64(
            self.raw,
            extent_index,
            u64::try_from(self.rendered.extent).map_err(|_| OpenClError::Overflow)?,
            context.owner,
        )?;
        let global = self
            .rendered
            .extent
            .checked_add(self.local_size - 1)
            .ok_or(OpenClError::Overflow)?
            / self.local_size
            * self.local_size;
        let raw = context.dispatch.kernel_launch(
            queue.raw,
            self.raw,
            global,
            self.local_size,
            context.owner,
        )?;
        Ok(Some(OpenClEvent::new(
            context.clone(),
            raw,
            snapshots.iter().map(BufferSnapshot::physical).collect(),
        )))
    }

    /// Submits a guarded integer kernel into provisional storage. The returned
    /// token is the sole authority that may commit those bytes to `output`.
    pub fn launch_transactional<'a>(
        &'a self,
        queue: &'a OpenClQueue,
        bindings: &'a [&'a OpenClBuffer],
    ) -> Result<OpenClTransaction<'a>, OpenClError> {
        let transaction = self
            .rendered
            .transaction
            .as_ref()
            .ok_or(OpenClError::InvalidArgument("kernel is not transactional"))?;
        queue.live()?;
        if self.closed.get() {
            return Err(OpenClError::Closed("kernel"));
        }
        if !Rc::ptr_eq(&self.program.context, &queue.context) {
            return Err(OpenClError::OwnerMismatch);
        }
        if bindings.len() != self.rendered.buffers.len() {
            return Err(OpenClError::InvalidBinding(
                "transaction binding count mismatch".into(),
            ));
        }
        let output = bindings[transaction.output_abi_index];
        let snapshots = bindings
            .iter()
            .zip(&self.rendered.buffers)
            .map(|(binding, abi)| {
                binding.snapshot(
                    &self.program.context,
                    0,
                    abi.elements
                        .checked_mul(abi.dtype.itemsize())
                        .ok_or(OpenClError::Overflow)?,
                    Some(abi.dtype),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let base_generation = snapshots[transaction.output_abi_index].generation();
        if self.rendered.extent == 0 {
            return Ok(OpenClTransaction {
                kernel: self,
                queue,
                output,
                snapshots,
                base_generation,
                candidate: output.candidate()?,
                status: OpenClContext {
                    inner: self.program.context.clone(),
                }
                .allocate(0)?,
                compute: None,
            });
        }
        let context = OpenClContext {
            inner: self.program.context.clone(),
        };
        let candidate = output.candidate()?;
        let status = context.allocate(4)?;
        queue.write(&status, 0, &CLEAN_STATUS.to_le_bytes())?;
        let status_snapshot = status.snapshot(&self.program.context, 0, 4, None)?;
        for (index, snapshot) in snapshots.iter().enumerate() {
            let raw = if index == transaction.output_abi_index {
                candidate.raw()
            } else {
                snapshot.raw()
            }
            .ok_or(OpenClError::Bounds)?;
            self.program.context.dispatch.kernel_arg_buffer(
                self.raw,
                u32::try_from(index).map_err(|_| OpenClError::Overflow)?,
                raw,
                self.program.context.owner,
            )?;
        }
        let extent_index = u32::try_from(bindings.len()).map_err(|_| OpenClError::Overflow)?;
        self.program.context.dispatch.kernel_arg_u64(
            self.raw,
            extent_index,
            self.rendered.extent as u64,
            self.program.context.owner,
        )?;
        self.program.context.dispatch.kernel_arg_buffer(
            self.raw,
            extent_index + 1,
            status_snapshot.raw().ok_or(OpenClError::Bounds)?,
            self.program.context.owner,
        )?;
        let global = self.rendered.extent.div_ceil(self.local_size) * self.local_size;
        let raw = self.program.context.dispatch.kernel_launch(
            queue.raw,
            self.raw,
            global,
            self.local_size,
            self.program.context.owner,
        )?;
        let retained = snapshots
            .iter()
            .map(BufferSnapshot::physical)
            .chain([candidate.clone(), status_snapshot.physical()])
            .collect();
        Ok(OpenClTransaction {
            kernel: self,
            queue,
            output,
            snapshots,
            base_generation,
            candidate: candidate.clone(),
            status,
            compute: Some(OpenClEvent::new(
                self.program.context.clone(),
                raw,
                retained,
            )),
        })
    }
}

/// Non-cloneable staged launch retaining every resource through status
/// collection and success-only output commit.
pub struct OpenClTransaction<'a> {
    kernel: &'a OpenClKernel,
    queue: &'a OpenClQueue,
    output: &'a OpenClBuffer,
    snapshots: Vec<BufferSnapshot>,
    base_generation: u64,
    candidate: Rc<PhysicalBuffer>,
    status: OpenClBuffer,
    compute: Option<OpenClEvent>,
}

impl OpenClTransaction<'_> {
    /// Reports compute-event readiness without reading status or committing.
    pub fn query(&self) -> Result<bool, OpenClError> {
        match &self.compute {
            Some(event) => event.query(),
            None => Ok(true),
        }
    }

    /// Consumes the token, checks the earliest fault, and commits only a clean
    /// provisional result. Failures before commit submission preserve output.
    pub fn wait(self) -> Result<(), OpenClError> {
        if let Some(event) = &self.compute {
            event.wait()?;
        } else {
            return Ok(());
        }
        let mut bytes = [0u8; 4];
        self.queue.read(&self.status, 0, &mut bytes)?;
        let status = u32::from_le_bytes(bytes);
        if status != CLEAN_STATUS {
            let transaction = self.kernel.rendered.transaction.as_ref().unwrap();
            let (index, guard) = transaction.decode(status)?;
            let count = if guard.operation.is_shift() {
                Some(self.read_rhs(transaction, guard, index)?)
            } else {
                None
            };
            return Err(OpenClError::IntegerFault {
                operation: guard.operation,
                index,
                count,
                bits: guard.dtype.bits(),
            });
        }
        for snapshot in &self.snapshots {
            snapshot.validate_current()?;
        }
        self.output
            .commit_candidate(self.base_generation, self.candidate.clone())?;
        Ok(())
    }

    fn read_rhs(
        &self,
        transaction: &super::OpenClTransactionAbi,
        guard: &super::OpenClGuard,
        logical: usize,
    ) -> Result<i64, OpenClError> {
        let value = detail_rhs_at(transaction, guard, logical, |arg, dtype, logical| {
            let buffer_id = match arg {
                crate::IndexValue::Buffer { buffer, .. }
                | crate::IndexValue::View { buffer, .. } => *buffer,
            };
            let position = self
                .kernel
                .rendered
                .buffers
                .iter()
                .position(|abi| abi.id == buffer_id)
                .ok_or_else(|| OpenClError::InvalidBinding("detail buffer absent".into()))?;
            let abi = &self.kernel.rendered.buffers[position];
            if abi.dtype != dtype {
                return Err(OpenClError::InvalidBinding("detail dtype mismatch".into()));
            }
            let offset = logical_offset(arg, logical)?;
            let mut bytes = vec![0u8; dtype.itemsize()];
            self.queue.read_snapshot(
                &self.snapshots[position],
                offset
                    .checked_mul(bytes.len())
                    .ok_or(OpenClError::Overflow)?,
                &mut bytes,
            )?;
            decode_detail_scalar(dtype, &bytes)
        })?;
        Ok(match guard.dtype {
            crate::DType::I32 | crate::DType::I64 => value.as_i64(),
            crate::DType::U32 | crate::DType::U64 => value.as_u64().min(i64::MAX as u64) as i64,
            _ => return Err(OpenClError::InvalidBinding("guard dtype mismatch".into())),
        })
    }
}

fn decode_detail_scalar(dtype: crate::DType, bytes: &[u8]) -> Result<crate::Scalar, OpenClError> {
    Ok(match dtype {
        crate::DType::Bool => crate::Scalar::Bool(bytes == [1]),
        crate::DType::I32 => crate::Scalar::I(i32::from_le_bytes(
            bytes.try_into().map_err(|_| OpenClError::Bounds)?,
        ) as i64),
        crate::DType::U32 => crate::Scalar::U(u32::from_le_bytes(
            bytes.try_into().map_err(|_| OpenClError::Bounds)?,
        ) as u64),
        crate::DType::I64 => crate::Scalar::I(i64::from_le_bytes(
            bytes.try_into().map_err(|_| OpenClError::Bounds)?,
        )),
        crate::DType::U64 => crate::Scalar::U(u64::from_le_bytes(
            bytes.try_into().map_err(|_| OpenClError::Bounds)?,
        )),
        _ => return Err(OpenClError::InvalidBinding("detail storage dtype".into())),
    })
}

impl Drop for OpenClKernel {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            self.program
                .context
                .dispatch
                .unregister_kernel_semantics(self.program.context.owner, self.raw);
            let _ = self
                .program
                .context
                .dispatch
                .kernel_release(self.raw, self.program.context.owner);
        }
    }
}

pub struct OpenClCache {
    context: OpenClContext,
    entries: RefCell<BTreeMap<String, Rc<OpenClKernel>>>,
}

impl OpenClCache {
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.borrow().is_empty()
    }

    pub fn load(
        &self,
        rendered: &RenderedOpenCl,
        options: &str,
        local_size: usize,
    ) -> Result<Rc<OpenClKernel>, OpenClError> {
        self.context.inner.live()?;
        if local_size == 0 || local_size > self.context.inner.device_info.max_work_group_size {
            return Err(OpenClError::InvalidArgument(
                "local size exceeds device work-group limit",
            ));
        }
        if options.as_bytes().contains(&0) || rendered.source.as_bytes().contains(&0) {
            return Err(OpenClError::InvalidArgument("interior NUL in build input"));
        }
        if !self
            .context
            .inner
            .device_info
            .capabilities
            .supports(rendered.required_capabilities)
        {
            return Err(OpenClError::Unsupported(
                "rendered kernel capabilities exceed the selected device".into(),
            ));
        }
        let cache_key = stable_key(&(
            rendered.cache_key.as_str(),
            options,
            self.context.inner.owner,
            self.context.inner.device.0,
            local_size,
        ));
        if let Some(kernel) = self.entries.borrow().get(&cache_key) {
            return Ok(kernel.clone());
        }
        let dispatch = &self.context.inner.dispatch;
        let raw_program = dispatch.program_create(
            self.context.inner.raw,
            &rendered.source,
            self.context.inner.owner,
        )?;
        let program = Rc::new(ProgramInner {
            context: self.context.inner.clone(),
            raw: raw_program,
            build: BuildInfo { log: String::new() },
            closed: Cell::new(false),
        });
        if let Err(error) = dispatch.program_build(
            raw_program,
            self.context.inner.device,
            options,
            self.context.inner.owner,
        ) {
            let log = dispatch
                .program_build_info(
                    raw_program,
                    self.context.inner.device,
                    self.context.inner.owner,
                )
                .map(|info| bounded_log(info.log))
                .unwrap_or_default();
            let code = match error {
                OpenClError::Driver { code, .. } => code,
                other => return Err(other),
            };
            return Err(OpenClError::Build { code, log });
        }
        let build = dispatch.program_build_info(
            raw_program,
            self.context.inner.device,
            self.context.inner.owner,
        )?;
        // The program is still uniquely owned here.
        let mut program = program;
        Rc::get_mut(&mut program)
            .expect("new program has one owner")
            .build = BuildInfo {
            log: bounded_log(build.log),
        };
        let raw_kernel =
            dispatch.kernel_create(raw_program, &rendered.entry, self.context.inner.owner)?;
        let semantics = Arc::new(KernelSemantics {
            buffers: rendered.buffers.clone(),
            extent: rendered.extent,
            program: rendered.semantic_program.clone(),
            transaction: rendered.transaction.clone(),
        });
        if let Err(error) =
            dispatch.register_kernel_semantics(self.context.inner.owner, raw_kernel, semantics)
        {
            let _ = dispatch.kernel_release(raw_kernel, self.context.inner.owner);
            return Err(error);
        }
        let kernel = Rc::new(OpenClKernel {
            program,
            raw: raw_kernel,
            rendered: rendered.clone(),
            cache_key: cache_key.clone(),
            local_size,
            closed: Cell::new(false),
        });
        self.entries.borrow_mut().insert(cache_key, kernel.clone());
        Ok(kernel)
    }
}

pub struct OpenClEvent {
    context: Rc<ContextInner>,
    raw: RawEvent,
    _retained: Vec<Rc<PhysicalBuffer>>,
    closed: Cell<bool>,
}

impl OpenClEvent {
    fn new(context: Rc<ContextInner>, raw: RawEvent, retained: Vec<Rc<PhysicalBuffer>>) -> Self {
        Self {
            context,
            raw,
            _retained: retained,
            closed: Cell::new(false),
        }
    }

    fn live(&self) -> Result<(), OpenClError> {
        self.context.live()?;
        if self.closed.get() {
            Err(OpenClError::Closed("event"))
        } else {
            Ok(())
        }
    }

    pub fn query(&self) -> Result<bool, OpenClError> {
        self.live()?;
        self.context
            .dispatch
            .event_query(self.raw, self.context.owner)
    }

    pub fn wait(&self) -> Result<(), OpenClError> {
        self.live()?;
        self.context
            .dispatch
            .event_wait(self.raw, self.context.owner)
    }
}

impl Drop for OpenClEvent {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            let _ = self
                .context
                .dispatch
                .event_release(self.raw, self.context.owner);
        }
    }
}

fn bounded_log(mut log: String) -> String {
    if log.len() <= MAX_BUILD_LOG_BYTES {
        return log;
    }
    let mut end = MAX_BUILD_LOG_BYTES;
    while !log.is_char_boundary(end) {
        end -= 1;
    }
    log.truncate(end);
    log
}

fn stable_key(value: &impl Hash) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
