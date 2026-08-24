//! Safe thread-confined OpenCL resource ownership and launch validation.
use super::{
    BufferCopyRegion, BuildInfo, DeviceInfo, Dispatch, OpenClError, RawBuffer, RawContext,
    RawDevice, RawEvent, RawKernel, RawPlatform, RawProgram, RawQueue, RenderedOpenCl,
    dispatch::KernelSemantics, ffi::NativeDispatch, transaction::CLEAN_STATUS,
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

struct ContextInner {
    dispatch: Arc<dyn Dispatch>,
    raw: RawContext,
    owner: u64,
    device: RawDevice,
    device_info: DeviceInfo,
    closed: Cell<bool>,
}

impl ContextInner {
    fn live(&self) -> Result<(), OpenClError> {
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
        self.inner.live()?;
        let raw = if bytes == 0 {
            None
        } else {
            Some(
                self.inner
                    .dispatch
                    .buffer_create(self.inner.raw, bytes, self.inner.owner)?,
            )
        };
        Ok(OpenClBuffer {
            context: self.inner.clone(),
            raw,
            bytes,
            closed: Cell::new(false),
        })
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
        buffer.preflight(&self.context, offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.context.dispatch.buffer_write(
            self.raw,
            buffer.raw.ok_or(OpenClError::Bounds)?,
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
        buffer.preflight(&self.context, offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.context.dispatch.buffer_read(
            self.raw,
            buffer.raw.ok_or(OpenClError::Bounds)?,
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
        src.preflight(&self.context, src_offset, bytes)?;
        dst.preflight(&self.context, dst_offset, bytes)?;
        if bytes == 0 {
            return Ok(None);
        }
        let raw = self.context.dispatch.buffer_copy(
            self.raw,
            src.raw.ok_or(OpenClError::Bounds)?,
            dst.raw.ok_or(OpenClError::Bounds)?,
            BufferCopyRegion {
                src_offset,
                dst_offset,
                bytes,
            },
            self.context.owner,
        )?;
        Ok(Some(OpenClEvent::new(self.context.clone(), raw)))
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

pub struct OpenClBuffer {
    context: Rc<ContextInner>,
    raw: Option<RawBuffer>,
    bytes: usize,
    closed: Cell<bool>,
}

impl OpenClBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn owner_id(&self) -> u64 {
        self.context.owner
    }

    fn preflight(
        &self,
        context: &Rc<ContextInner>,
        offset: usize,
        bytes: usize,
    ) -> Result<(), OpenClError> {
        self.context.live()?;
        if self.closed.get() {
            return Err(OpenClError::Closed("buffer"));
        }
        if !Rc::ptr_eq(&self.context, context) {
            return Err(OpenClError::OwnerMismatch);
        }
        let end = offset.checked_add(bytes).ok_or(OpenClError::Overflow)?;
        if end > self.bytes {
            return Err(OpenClError::Bounds);
        }
        Ok(())
    }
}

impl Drop for OpenClBuffer {
    fn drop(&mut self) {
        if !self.closed.replace(true)
            && let Some(raw) = self.raw
        {
            let _ = self
                .context
                .dispatch
                .buffer_release(raw, self.context.owner);
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
        for (index, (binding, abi)) in bindings.iter().zip(&self.rendered.buffers).enumerate() {
            let bytes = abi
                .elements
                .checked_mul(abi.dtype.itemsize())
                .ok_or(OpenClError::Overflow)?;
            binding
                .preflight(context, 0, bytes)
                .map_err(|error| match error {
                    OpenClError::OwnerMismatch => OpenClError::OwnerMismatch,
                    other => OpenClError::InvalidBinding(format!(
                        "buffer {index} failed validation: {other}"
                    )),
                })?;
        }
        if self.rendered.extent == 0 {
            return Ok(None);
        }
        for (index, binding) in bindings.iter().enumerate() {
            context.dispatch.kernel_arg_buffer(
                self.raw,
                u32::try_from(index).map_err(|_| OpenClError::Overflow)?,
                binding.raw.ok_or_else(|| {
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
        Ok(Some(OpenClEvent::new(context.clone(), raw)))
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
        for (binding, abi) in bindings.iter().zip(&self.rendered.buffers) {
            binding.preflight(
                &self.program.context,
                0,
                abi.elements
                    .checked_mul(abi.dtype.itemsize())
                    .ok_or(OpenClError::Overflow)?,
            )?;
        }
        if self.rendered.extent == 0 {
            return Ok(OpenClTransaction {
                kernel: self,
                queue,
                bindings,
                output,
                scratch: OpenClContext {
                    inner: self.program.context.clone(),
                }
                .allocate(0)?,
                status: OpenClContext {
                    inner: self.program.context.clone(),
                }
                .allocate(0)?,
                compute: None,
            });
        }
        if self.rendered.extent > u32::MAX as usize {
            return Err(OpenClError::Unsupported(
                "transaction extent exceeds status index".into(),
            ));
        }
        let output_bytes = self.rendered.buffers[transaction.output_abi_index]
            .elements
            .checked_mul(transaction.dtype.itemsize())
            .ok_or(OpenClError::Overflow)?;
        let context = OpenClContext {
            inner: self.program.context.clone(),
        };
        let scratch = context.allocate(output_bytes)?;
        let status = context.allocate(4)?;
        queue.write(&status, 0, &CLEAN_STATUS.to_le_bytes())?;
        for (index, binding) in bindings.iter().enumerate() {
            let raw = if index == transaction.output_abi_index {
                scratch.raw
            } else {
                binding.raw
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
            status.raw.ok_or(OpenClError::Bounds)?,
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
        Ok(OpenClTransaction {
            kernel: self,
            queue,
            bindings,
            output,
            scratch,
            status,
            compute: Some(OpenClEvent::new(self.program.context.clone(), raw)),
        })
    }
}

/// Non-cloneable staged launch retaining every resource through status
/// collection and success-only output commit.
pub struct OpenClTransaction<'a> {
    kernel: &'a OpenClKernel,
    queue: &'a OpenClQueue,
    bindings: &'a [&'a OpenClBuffer],
    output: &'a OpenClBuffer,
    scratch: OpenClBuffer,
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
        let index = u32::from_le_bytes(bytes);
        if index != CLEAN_STATUS {
            let transaction = self.kernel.rendered.transaction.as_ref().unwrap();
            let count = if transaction.operation.is_shift() {
                Some(self.read_rhs(index as usize)?)
            } else {
                None
            };
            return Err(OpenClError::IntegerFault {
                operation: transaction.operation,
                index: index as usize,
                count,
                bits: transaction.dtype.bits(),
            });
        }
        let commit = self
            .queue
            .copy(&self.scratch, self.output, 0, 0, self.scratch.len())?;
        if let Some(event) = commit {
            event.wait()?;
        }
        Ok(())
    }

    fn read_rhs(&self, logical: usize) -> Result<i64, OpenClError> {
        let transaction = self.kernel.rendered.transaction.as_ref().unwrap();
        let abi = &self.kernel.rendered.buffers[transaction.rhs_abi_index];
        let offset = transaction_rhs_offset(&transaction.rhs_index, logical)?;
        let mut bytes = vec![0u8; abi.dtype.itemsize()];
        self.queue.read(
            self.bindings[transaction.rhs_abi_index],
            offset
                .checked_mul(bytes.len())
                .ok_or(OpenClError::Overflow)?,
            &mut bytes,
        )?;
        Ok(match abi.dtype {
            crate::DType::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as i64,
            crate::DType::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as i64,
            crate::DType::I64 => i64::from_le_bytes(bytes.try_into().unwrap()),
            crate::DType::U64 => {
                u64::from_le_bytes(bytes.try_into().unwrap()).min(i64::MAX as u64) as i64
            }
            _ => {
                return Err(OpenClError::InvalidBinding(
                    "guarded RHS dtype mismatch".into(),
                ));
            }
        })
    }
}

pub(super) fn transaction_rhs_offset(
    arg: &crate::UArg,
    logical: usize,
) -> Result<usize, OpenClError> {
    let (input, output, view) = match arg {
        crate::UArg::BufferIndex {
            input_shape,
            output_shape,
            ..
        } => (input_shape, output_shape, None),
        crate::UArg::ViewBufferIndex {
            input_shape,
            output_shape,
            view,
            ..
        } => (input_shape, output_shape, Some(view)),
        _ => {
            return Err(OpenClError::InvalidBinding(
                "guarded RHS index mismatch".into(),
            ));
        }
    };
    let mut input_offset = 0usize;
    let output_strides = output.contiguous_strides();
    let input_strides = input.contiguous_strides();
    let rank_delta = output
        .rank()
        .checked_sub(input.rank())
        .ok_or(OpenClError::Bounds)?;
    for axis in 0..input.rank() {
        let dim = input.dims()[axis];
        let coordinate =
            (logical / output_strides[axis + rank_delta]) % output.dims()[axis + rank_delta];
        input_offset += if dim == 1 {
            0
        } else {
            coordinate * input_strides[axis]
        };
    }
    match view {
        Some(view) => view
            .element_offset(input_offset)
            .map_err(|_| OpenClError::Bounds),
        None => Ok(input_offset),
    }
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
    closed: Cell<bool>,
}

impl OpenClEvent {
    fn new(context: Rc<ContextInner>, raw: RawEvent) -> Self {
        Self {
            context,
            raw,
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
