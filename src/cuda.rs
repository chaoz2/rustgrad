//! A small, toolkit-free CUDA Driver API runtime.
//!
//! This module deliberately uses only the Driver API.  `Driver::load` opens
//! `libcuda` at runtime, so a CPU-only installation can compile and run all
//! default tests.  The ABI below follows CUDA's Driver API headers:
//! <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html> and
//! <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html>.
//! Every raw handle is private; the owning RAII type also carries its context.

use crate::cuda_profile::{Metadata, OperationKind, ProfilingSession, TimedSample, TimingError};

use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    fmt,
    marker::PhantomData,
    num::NonZeroUsize,
    ptr,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

type CuResult = i32;
type CuDevice = c_int;
type CuDevicePtr = u64;
type CuContext = *mut c_void;
type CuStream = *mut c_void;
type CuEvent = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;
const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NOT_READY: CuResult = 600;
const CU_CTX_SCHED_AUTO: c_uint = 0;
const CU_EVENT_DEFAULT: c_uint = 0;
const CU_STREAM_DEFAULT: c_uint = 0;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;

/// A CUDA ordinal, distinct from arbitrary signed integers at the public API.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DeviceId(pub u32);

/// Device properties used by the initial PTX policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capability {
    pub device: DeviceId,
    pub name: String,
    pub major: u32,
    pub minor: u32,
    pub total_memory: usize,
    pub max_threads_per_block: u32,
}
impl Capability {
    pub fn sm(&self) -> u32 {
        self.major * 10 + self.minor
    }
}

/// Failure returned by the loader, an API call, or a checked wrapper boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaError {
    JitCompile {
        code: i32,
        name: String,
        message: String,
        info_log: String,
        error_log: String,
    },
    LibraryNotFound {
        tried: Vec<&'static str>,
        detail: String,
    },
    MissingSymbol(&'static str),
    Version {
        found: i32,
        required: i32,
    },
    Driver {
        code: i32,
        name: String,
        message: String,
    },
    InvalidArgument(&'static str),
    Overflow,
    WrongDevice {
        expected: DeviceId,
        actual: DeviceId,
    },
    Closed(&'static str),
    /// A pooled allocation was used after its lease was returned.
    StaleLease,
    ContextMismatch,
    NotReady,
}
impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JitCompile {
                code,
                name,
                message,
                info_log,
                error_log,
            } => write!(
                f,
                "CUDA JIT {name} ({code}): {message}; info={info_log:?}; error={error_log:?}"
            ),
            Self::LibraryNotFound { tried, detail } => write!(
                f,
                "CUDA Driver library not found (tried {tried:?}): {detail}"
            ),
            Self::MissingSymbol(s) => write!(f, "CUDA Driver symbol is unavailable: {s}"),
            Self::Version { found, required } => write!(
                f,
                "CUDA Driver API version {found} is older than required {required}"
            ),
            Self::Driver {
                code,
                name,
                message,
            } => write!(f, "CUDA Driver {name} ({code}): {message}"),
            Self::InvalidArgument(s) => write!(f, "invalid CUDA argument: {s}"),
            Self::Overflow => write!(f, "CUDA size calculation overflow"),
            Self::WrongDevice { expected, actual } => write!(
                f,
                "CUDA resource belongs to device {actual:?}, not {expected:?}"
            ),
            Self::Closed(s) => write!(f, "CUDA {s} has already been closed"),
            Self::StaleLease => write!(f, "CUDA allocation lease is stale"),
            Self::ContextMismatch => {
                write!(f, "CUDA resource used with a different current context")
            }
            Self::NotReady => write!(f, "CUDA operation is not ready"),
        }
    }
}
/// CUDA JIT option identifiers from `cuda.h` (`CUjit_option`).
pub const CU_JIT_OPTIMIZATION_LEVEL: u32 = 7;
pub const CU_JIT_TARGET_FROM_CUCONTEXT: u32 = 8;
pub const CU_JIT_INFO_LOG_BUFFER: u32 = 4;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: u32 = 5;
pub const CU_JIT_ERROR_LOG_BUFFER: u32 = 6;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: u32 = 7;
/// Bounded writable log buffers used by `cuModuleLoadDataEx`.
#[derive(Clone, Debug)]
pub struct ModuleLoadOptions {
    pub optimization_level: u32,
    pub log_bytes: usize,
    pub capture_logs: bool,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleLoadMetadata {
    pub used_load_data_ex: bool,
    pub info_log: String,
    pub error_log: String,
}
impl Default for ModuleLoadOptions {
    fn default() -> Self {
        Self {
            optimization_level: 4,
            log_bytes: 4096,
            capture_logs: false,
        }
    }
}
impl ModuleLoadOptions {
    fn validate(&self) -> Result<(), CudaError> {
        if self.optimization_level > 4 || self.log_bytes == 0 || self.log_bytes > 65536 {
            Err(CudaError::InvalidArgument("JIT options"))
        } else {
            Ok(())
        }
    }
}
impl std::error::Error for CudaError {}

/// Injectable Driver calls.  It is intentionally a typed trait: mock tests do
/// not manufacture function pointers or rely on ABI casts.
pub trait Dispatch: Send + Sync + 'static {
    fn driver_version(&self, out: &mut c_int) -> CuResult;
    fn init(&self, flags: c_uint) -> CuResult;
    fn device_count(&self, out: &mut c_int) -> CuResult;
    fn device_get(&self, out: &mut CuDevice, ordinal: c_int) -> CuResult;
    fn device_name(&self, out: &mut [c_char], device: CuDevice) -> CuResult;
    fn device_cc(&self, major: &mut c_int, minor: &mut c_int, device: CuDevice) -> CuResult;
    fn device_memory(&self, out: &mut usize, device: CuDevice) -> CuResult;
    fn device_attribute(&self, out: &mut c_int, attr: c_int, device: CuDevice) -> CuResult;
    fn ctx_create(&self, out: &mut CuContext, flags: c_uint, device: CuDevice) -> CuResult;
    fn ctx_destroy(&self, context: CuContext) -> CuResult;
    fn ctx_get_current(&self, out: &mut CuContext) -> CuResult;
    fn ctx_set_current(&self, context: CuContext) -> CuResult;
    fn primary_ctx_retain(&self, _out: &mut CuContext, _device: CuDevice) -> CuResult {
        801
    }
    fn primary_ctx_release(&self, _device: CuDevice) -> CuResult {
        801
    }
    fn primary_ctx_get_state(
        &self,
        _device: CuDevice,
        _flags: &mut c_uint,
        _active: &mut c_int,
    ) -> CuResult {
        801
    }
    fn primary_ctx_set_flags(&self, _device: CuDevice, _flags: c_uint) -> CuResult {
        801
    }
    fn ctx_push_current(&self, _context: CuContext) -> CuResult {
        801
    }
    fn ctx_pop_current(&self, _out: &mut CuContext) -> CuResult {
        801
    }
    fn mem_alloc(&self, out: &mut CuDevicePtr, bytes: usize) -> CuResult;
    fn mem_free(&self, ptr: CuDevicePtr) -> CuResult;
    fn memcpy_htod(&self, dst: CuDevicePtr, src: *const c_void, bytes: usize) -> CuResult;
    fn memcpy_dtoh(&self, dst: *mut c_void, src: CuDevicePtr, bytes: usize) -> CuResult;
    fn memcpy_dtod(&self, dst: CuDevicePtr, src: CuDevicePtr, bytes: usize) -> CuResult;
    fn device_can_access_peer(&self, _: &mut c_int, _: CuDevice, _: CuDevice) -> CuResult {
        801
    }
    fn ctx_enable_peer_access(&self, _: CuContext, _: c_uint) -> CuResult {
        801
    }
    fn ctx_disable_peer_access(&self, _: CuContext) -> CuResult {
        801
    }
    fn memcpy_peer_async(
        &self,
        _: CuDevicePtr,
        _: CuContext,
        _: CuDevicePtr,
        _: CuContext,
        _: usize,
        _: CuStream,
    ) -> CuResult {
        801
    }
    fn memcpy_htod_async(
        &self,
        _dst: CuDevicePtr,
        _src: *const c_void,
        _bytes: usize,
        _stream: CuStream,
    ) -> CuResult {
        801
    }
    fn memcpy_dtoh_async(
        &self,
        _dst: *mut c_void,
        _src: CuDevicePtr,
        _bytes: usize,
        _stream: CuStream,
    ) -> CuResult {
        801
    }
    fn memcpy_dtod_async(
        &self,
        _dst: CuDevicePtr,
        _src: CuDevicePtr,
        _bytes: usize,
        _stream: CuStream,
    ) -> CuResult {
        801
    }
    fn mem_host_alloc(&self, _out: &mut *mut c_void, _bytes: usize, _flags: c_uint) -> CuResult {
        801
    }
    fn mem_free_host(&self, _ptr: *mut c_void) -> CuResult {
        801
    }
    fn supports_async_transfers(&self) -> bool {
        false
    }
    fn supports_pinned_host_memory(&self) -> bool {
        false
    }
    fn stream_begin_capture(&self, _stream: CuStream, _mode: c_uint) -> CuResult {
        801
    }
    fn stream_end_capture(&self, _stream: CuStream, _graph: &mut CuGraph) -> CuResult {
        801
    }
    fn graph_instantiate(&self, _exec: &mut CuGraphExec, _graph: CuGraph) -> CuResult {
        801
    }
    fn graph_launch(&self, _exec: CuGraphExec, _stream: CuStream) -> CuResult {
        801
    }
    fn graph_destroy(&self, _graph: CuGraph) -> CuResult {
        801
    }
    fn graph_exec_destroy(&self, _exec: CuGraphExec) -> CuResult {
        801
    }
    fn supports_graphs(&self) -> bool {
        false
    }
    fn stream_create(&self, out: &mut CuStream, flags: c_uint) -> CuResult;
    fn stream_destroy(&self, stream: CuStream) -> CuResult;
    fn stream_sync(&self, stream: CuStream) -> CuResult;
    fn event_create(&self, out: &mut CuEvent, flags: c_uint) -> CuResult;
    fn event_destroy(&self, event: CuEvent) -> CuResult;
    fn event_record(&self, event: CuEvent, stream: CuStream) -> CuResult;
    fn event_query(&self, event: CuEvent) -> CuResult;
    fn event_sync(&self, event: CuEvent) -> CuResult;
    fn stream_wait_event(&self, stream: CuStream, event: CuEvent, flags: c_uint) -> CuResult;
    /// `cuEventElapsedTime` is optional so older Driver libraries still expose
    /// the ordinary event lifecycle. Implementations return `MissingSymbol`
    /// when timing is unavailable rather than fabricating a Driver status.
    fn event_elapsed(
        &self,
        _out: &mut f32,
        _start: CuEvent,
        _end: CuEvent,
    ) -> Result<CuResult, CudaError> {
        Err(CudaError::MissingSymbol("cuEventElapsedTime"))
    }
    fn module_load_data(&self, out: &mut CuModule, image: *const c_void) -> CuResult;
    /// Exact `cuModuleLoadDataEx(CUmodule*, const void*, unsigned, CUjit_option*, void**)` ABI.
    /// The default is the documented no-option compatibility fallback.
    fn module_load_data_ex(
        &self,
        out: &mut CuModule,
        image: *const c_void,
        _options: &[u32],
        _values: &mut [*mut c_void],
    ) -> CuResult {
        self.module_load_data(out, image)
    }
    fn supports_module_load_data_ex(&self) -> bool {
        false
    }
    fn module_unload(&self, module: CuModule) -> CuResult;
    fn module_function(&self, out: &mut CuFunction, module: CuModule, name: &CStr) -> CuResult;
    fn launch(
        &self,
        function: CuFunction,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: CuStream,
        args: *mut *mut c_void,
    ) -> CuResult;
    fn error_name(&self, code: CuResult) -> Option<String>;
    fn error_string(&self, code: CuResult) -> Option<String>;
}

fn check(d: &dyn Dispatch, result: CuResult) -> Result<(), CudaError> {
    if result == CUDA_SUCCESS {
        return Ok(());
    }
    Err(CudaError::Driver {
        code: result,
        name: d
            .error_name(result)
            .unwrap_or_else(|| "CUDA_ERROR_UNKNOWN".into()),
        message: d.error_string(result).unwrap_or_default(),
    })
}

struct Inner {
    dispatch: Arc<dyn Dispatch>,
    initialized: AtomicBool,
}
/// The loaded Driver API. Cloning it shares initialization and the native
/// library lifetime (when present).
#[derive(Clone)]
pub struct Driver(Arc<Inner>);
impl Driver {
    pub const MIN_DRIVER_API_VERSION: i32 = 11000;
    /// Loads the platform Driver library. No CUDA library is linked into this crate.
    pub fn load() -> Result<Self, CudaError> {
        let dispatch = Arc::new(NativeDispatch::load()?);
        Self::from_dispatch(dispatch)
    }
    /// Constructs a driver around a typed mock or alternate host implementation.
    pub fn from_dispatch(dispatch: Arc<dyn Dispatch>) -> Result<Self, CudaError> {
        let mut version = 0;
        check(dispatch.as_ref(), dispatch.driver_version(&mut version))?;
        if version < Self::MIN_DRIVER_API_VERSION {
            return Err(CudaError::Version {
                found: version,
                required: Self::MIN_DRIVER_API_VERSION,
            });
        }
        Ok(Self(Arc::new(Inner {
            dispatch,
            initialized: AtomicBool::new(false),
        })))
    }
    fn init(&self) -> Result<(), CudaError> {
        if !self.0.initialized.swap(true, Ordering::AcqRel)
            && let Err(e) = check(self.0.dispatch.as_ref(), self.0.dispatch.init(0))
        {
            self.0.initialized.store(false, Ordering::Release);
            return Err(e);
        }
        Ok(())
    }
    pub fn device_count(&self) -> Result<u32, CudaError> {
        self.init()?;
        let mut n = 0;
        check(
            self.0.dispatch.as_ref(),
            self.0.dispatch.device_count(&mut n),
        )?;
        u32::try_from(n).map_err(|_| CudaError::InvalidArgument("negative device count"))
    }
    pub fn device(&self, id: DeviceId) -> Result<Device, CudaError> {
        self.init()?;
        let ordinal =
            c_int::try_from(id.0).map_err(|_| CudaError::InvalidArgument("device ordinal"))?;
        let mut raw = 0;
        check(
            self.0.dispatch.as_ref(),
            self.0.dispatch.device_get(&mut raw, ordinal),
        )?;
        Ok(Device {
            driver: self.clone(),
            id,
            raw,
        })
    }
}

#[derive(Clone)]
pub struct Device {
    driver: Driver,
    id: DeviceId,
    raw: CuDevice,
}
impl Device {
    pub fn id(&self) -> DeviceId {
        self.id
    }
    pub fn capability(&self) -> Result<Capability, CudaError> {
        let d = self.driver.0.dispatch.as_ref();
        let mut name = [0_i8; 256];
        let (mut major, mut minor, mut memory, mut threads) = (0, 0, 0_usize, 0);
        check(d, d.device_name(&mut name, self.raw))?;
        check(d, d.device_cc(&mut major, &mut minor, self.raw))?;
        check(d, d.device_memory(&mut memory, self.raw))?;
        check(
            d,
            d.device_attribute(
                &mut threads,
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                self.raw,
            ),
        )?;
        let bytes: Vec<u8> = name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        Ok(Capability {
            device: self.id,
            name: String::from_utf8_lossy(&bytes).into_owned(),
            major: u32::try_from(major)
                .map_err(|_| CudaError::InvalidArgument("compute capability"))?,
            minor: u32::try_from(minor)
                .map_err(|_| CudaError::InvalidArgument("compute capability"))?,
            total_memory: memory,
            max_threads_per_block: u32::try_from(threads)
                .map_err(|_| CudaError::InvalidArgument("max threads"))?,
        })
    }
    /// Creates an owned context, matching tinygrad's explicit context policy.
    pub fn create_context(&self) -> Result<Context, CudaError> {
        let mut raw = ptr::null_mut();
        check(
            self.driver.0.dispatch.as_ref(),
            self.driver
                .0
                .dispatch
                .ctx_create(&mut raw, CU_CTX_SCHED_AUTO, self.raw),
        )?;
        Ok(Context {
            inner: Rc::new(ContextInner {
                driver: self.driver.clone(),
                device: self.id,
                raw,
                closed: AtomicBool::new(false),
            }),
            _thread: PhantomData,
        })
    }
    /// Retains CUDA's primary context. This separate API leaves owned-context
    /// behaviour unchanged; clones share one retain and release once.
    pub fn retain_primary_context(&self) -> Result<PrimaryContext, CudaError> {
        let mut raw = ptr::null_mut();
        let d = self.driver.0.dispatch.as_ref();
        check(d, d.primary_ctx_retain(&mut raw, self.raw))?;
        Ok(PrimaryContext(Arc::new(PrimaryInner {
            driver: self.driver.clone(),
            device: self.id,
            raw,
            raw_device: self.raw,
        })))
    }
    /// Returns `(flags, active)` without retaining or activating the primary context.
    pub fn primary_context_state(&self) -> Result<(u32, bool), CudaError> {
        let mut flags = 0;
        let mut active = 0;
        let d = self.driver.0.dispatch.as_ref();
        check(
            d,
            d.primary_ctx_get_state(self.raw, &mut flags, &mut active),
        )?;
        Ok((flags, active != 0))
    }
    /// Sets primary-context flags only while it is inactive, as required by CUDA.
    pub fn set_primary_context_flags(&self, flags: u32) -> Result<(), CudaError> {
        let (_, active) = self.primary_context_state()?;
        if active {
            return Err(CudaError::InvalidArgument("primary context is active"));
        }
        let d = self.driver.0.dispatch.as_ref();
        check(d, d.primary_ctx_set_flags(self.raw, flags))
    }
}
struct PrimaryInner {
    driver: Driver,
    device: DeviceId,
    raw: CuContext,
    raw_device: CuDevice,
}
// CUDA primary contexts are shareable; all thread-local activation goes through
// push/pop guards and no raw handle is exposed.
unsafe impl Send for PrimaryInner {}
unsafe impl Sync for PrimaryInner {}
impl Drop for PrimaryInner {
    fn drop(&mut self) {
        let _ = self.driver.0.dispatch.primary_ctx_release(self.raw_device);
    }
}
/// Shareable retained primary context. Currentness is guarded per thread.
#[derive(Clone)]
pub struct PrimaryContext(Arc<PrimaryInner>);
impl PrimaryContext {
    pub fn peer_access_to(&self, destination: &PrimaryContext) -> Result<PeerAccess, CudaError> {
        if Arc::ptr_eq(&self.0, &destination.0) {
            return Err(CudaError::InvalidArgument(
                "peer access requires distinct primary owners",
            ));
        }
        if !Arc::ptr_eq(&self.0.driver.0, &destination.0.driver.0) {
            return Err(CudaError::ContextMismatch);
        }
        let d = self.0.driver.0.dispatch.as_ref();
        let mut can = 0;
        check(
            d,
            d.device_can_access_peer(&mut can, self.0.raw_device, destination.0.raw_device),
        )?;
        if can == 0 {
            return Err(CudaError::InvalidArgument("peer access unsupported"));
        };
        let _g = self.enter()?;
        check(d, d.ctx_enable_peer_access(destination.0.raw, 0))?;
        Ok(PeerAccess {
            source: self.clone(),
            destination: destination.clone(),
            closed: AtomicBool::new(false),
        })
    }
    pub fn allocator(&self) -> Arc<PrimaryCudaAllocator> {
        PrimaryCudaAllocator::new(self.clone())
    }
    pub fn device(&self) -> DeviceId {
        self.0.device
    }
    /// Stable, crate-private owner identity used to partition concurrent JIT caches.
    pub(crate) fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
    pub fn enter(&self) -> Result<PrimaryContextGuard, CudaError> {
        let d = self.0.driver.0.dispatch.as_ref();
        check(d, d.ctx_push_current(self.0.raw))?;
        Ok(PrimaryContextGuard {
            primary: self.clone(),
            active: true,
        })
    }
    pub fn allocate(&self, bytes: NonZeroUsize) -> Result<DeviceBuffer, CudaError> {
        Owner::Primary(self.clone()).allocate(bytes)
    }
    fn allocate_primary_block(&self, bytes: NonZeroUsize) -> Result<PrimaryBlock, CudaError> {
        let _guard = self.enter()?;
        let mut ptr = 0;
        let d = self.0.driver.0.dispatch.as_ref();
        check(d, d.mem_alloc(&mut ptr, bytes.get()))?;
        Ok(PrimaryBlock {
            primary: self.clone(),
            ptr,
            capacity: bytes.get(),
            generation: std::sync::atomic::AtomicU64::new(0),
            closed: AtomicBool::new(false),
        })
    }
    pub fn stream(&self) -> Result<Stream, CudaError> {
        Owner::Primary(self.clone()).stream()
    }
    pub fn event(&self) -> Result<Event, CudaError> {
        Owner::Primary(self.clone()).event()
    }
    /// A shareable primary-only completion fence for deferred resource work.
    pub fn event_fence(&self) -> Result<PrimaryEventFence, CudaError> {
        let _guard = self.enter()?;
        let mut raw = ptr::null_mut();
        let d = self.0.driver.0.dispatch.as_ref();
        check(d, d.event_create(&mut raw, CU_EVENT_DEFAULT))?;
        Ok(PrimaryEventFence {
            primary: self.clone(),
            raw: raw as usize,
            closed: AtomicBool::new(false),
        })
    }
    /// Creates an event with CUDA's default flags, which leave elapsed timing
    /// enabled. This crate-private entry point keeps profiling timing isolated
    /// from ordinary event users.
    pub(crate) fn timing_event(&self) -> Result<Event, CudaError> {
        Owner::Primary(self.clone()).event()
    }
    #[allow(dead_code)] // consumed by the crate-private profiled PTX launch surface.
    pub(crate) fn validate_launch(&self, config: LaunchConfig) -> Result<(), CudaError> {
        config.validate(
            self.0
                .driver
                .device(self.device())?
                .capability()?
                .max_threads_per_block,
        )
    }
    pub fn allocate_pinned(&self, bytes: NonZeroUsize) -> Result<PinnedHostBuffer, CudaError> {
        Owner::Primary(self.clone()).pinned(bytes)
    }
    pub fn module_from_ptx(&self, ptx: &CStr) -> Result<CudaModule, CudaError> {
        self.module_from_ptx_with_options(ptx, ModuleLoadOptions::default())
    }
    pub fn module_from_ptx_with_options(
        &self,
        ptx: &CStr,
        options: ModuleLoadOptions,
    ) -> Result<CudaModule, CudaError> {
        Owner::Primary(self.clone()).module_from_ptx_with_options(ptx, options)
    }
}
/// Directional source-to-destination primary-context peer mapping.
pub struct PeerAccess {
    source: PrimaryContext,
    destination: PrimaryContext,
    closed: AtomicBool,
}
impl PeerAccess {
    fn matches(&self, source: &PrimaryContext, destination: &PrimaryContext) -> bool {
        !self.closed.load(Ordering::Acquire)
            && Arc::ptr_eq(&self.source.0, &source.0)
            && Arc::ptr_eq(&self.destination.0, &destination.0)
    }
    pub fn close(&self) -> Result<(), CudaError> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        };
        let _g = self.source.enter()?;
        check(
            self.source.0.driver.0.dispatch.as_ref(),
            self.source
                .0
                .driver
                .0
                .dispatch
                .ctx_disable_peer_access(self.destination.0.raw),
        )
    }
}
impl Drop for PeerAccess {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
pub struct PrimaryContextGuard {
    primary: PrimaryContext,
    active: bool,
}
impl Drop for PrimaryContextGuard {
    fn drop(&mut self) {
        if self.active {
            let mut popped = ptr::null_mut();
            let _ = self
                .primary
                .0
                .driver
                .0
                .dispatch
                .ctx_pop_current(&mut popped);
            debug_assert_eq!(popped, self.primary.0.raw);
        }
    }
}
struct ContextInner {
    driver: Driver,
    device: DeviceId,
    raw: CuContext,
    closed: AtomicBool,
}
impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            let _ = self.driver.0.dispatch.ctx_destroy(self.raw);
        }
    }
}
/// Thread-affine owned CUDA context. CUDA current-context state is per thread,
/// so the `Rc` marker intentionally prevents Send and Sync.
#[derive(Clone)]
pub struct Context {
    inner: Rc<ContextInner>,
    _thread: PhantomData<Rc<()>>,
}
impl Context {
    pub fn allocator(&self) -> std::rc::Rc<CudaAllocator> {
        CudaAllocator::new(Owner::Owned(self.clone()))
    }
    pub fn device(&self) -> DeviceId {
        self.inner.device
    }
    pub fn close(&self) -> Result<(), CudaError> {
        if self.inner.closed.swap(true, Ordering::AcqRel) {
            return Err(CudaError::Closed("context"));
        }
        check(
            self.inner.driver.0.dispatch.as_ref(),
            self.inner.driver.0.dispatch.ctx_destroy(self.inner.raw),
        )
    }
    pub fn enter(&self) -> Result<ContextGuard, CudaError> {
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(CudaError::Closed("context"));
        }
        let d = self.inner.driver.0.dispatch.as_ref();
        let mut previous = ptr::null_mut();
        check(d, d.ctx_get_current(&mut previous))?;
        check(d, d.ctx_set_current(self.inner.raw))?;
        Ok(ContextGuard {
            context: self.clone(),
            previous,
            restore: true,
        })
    }
    fn current(&self) -> Result<ContextGuard, CudaError> {
        self.enter()
    }
    pub fn allocate(&self, bytes: NonZeroUsize) -> Result<DeviceBuffer, CudaError> {
        Owner::Owned(self.clone()).allocate(bytes)
    }
    pub fn stream(&self) -> Result<Stream, CudaError> {
        Owner::Owned(self.clone()).stream()
    }
    pub fn event(&self) -> Result<Event, CudaError> {
        Owner::Owned(self.clone()).event()
    }
    pub fn allocate_pinned(&self, bytes: NonZeroUsize) -> Result<PinnedHostBuffer, CudaError> {
        Owner::Owned(self.clone()).pinned(bytes)
    }
    pub fn module_from_ptx(&self, ptx: &CStr) -> Result<CudaModule, CudaError> {
        self.module_from_ptx_with_options(ptx, ModuleLoadOptions::default())
    }
    pub fn module_from_ptx_with_options(
        &self,
        ptx: &CStr,
        options: ModuleLoadOptions,
    ) -> Result<CudaModule, CudaError> {
        Owner::Owned(self.clone()).module_from_ptx_with_options(ptx, options)
    }
}

// Sealed resource owner.  This is deliberately an enum rather than a public
// trait object: callers cannot forge a raw context or mix driver instances.
// A resource retains this value, so destruction is ordered as resource cleanup
// followed by owned-context destruction or primary-context release.  The owned
// variant keeps every resource !Send/!Sync; primary contexts themselves are
// Send + Sync and use CUDA push/pop on the calling thread.
#[derive(Clone)]
enum Owner {
    Owned(Context),
    Primary(PrimaryContext),
}
#[allow(dead_code)] // guards deliberately live only to restore CUDA currentness on Drop.
enum OwnerGuard {
    Owned(ContextGuard),
    Primary(PrimaryContextGuard),
}
impl Owner {
    fn is_primary(&self) -> bool {
        matches!(self, Self::Primary(_))
    }
    fn device(&self) -> DeviceId {
        match self {
            Self::Owned(x) => x.device(),
            Self::Primary(x) => x.device(),
        }
    }
    fn dispatch(&self) -> &dyn Dispatch {
        match self {
            Self::Owned(x) => x.inner.driver.0.dispatch.as_ref(),
            Self::Primary(x) => x.0.driver.0.dispatch.as_ref(),
        }
    }
    fn current(&self) -> Result<OwnerGuard, CudaError> {
        match self {
            Self::Owned(x) => x.current().map(OwnerGuard::Owned),
            Self::Primary(x) => x.enter().map(OwnerGuard::Primary),
        }
    }
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Owned(a), Self::Owned(b)) => Rc::ptr_eq(&a.inner, &b.inner),
            (Self::Primary(a), Self::Primary(b)) => Arc::ptr_eq(&a.0, &b.0),
            _ => false,
        }
    }
    fn device_capability_threads(&self) -> Result<u32, CudaError> {
        match self {
            Self::Owned(x) => x.device_capability_threads(),
            Self::Primary(x) => Ok(x
                .0
                .driver
                .device(x.device())?
                .capability()?
                .max_threads_per_block),
        }
    }
    fn allocate(&self, bytes: NonZeroUsize) -> Result<DeviceBuffer, CudaError> {
        let _guard = self.current()?;
        let mut ptr = 0;
        check(
            self.dispatch(),
            self.dispatch().mem_alloc(&mut ptr, bytes.get()),
        )?;
        Ok(DeviceBuffer {
            owner: self.clone(),
            ptr,
            bytes: bytes.get(),
            closed: AtomicBool::new(false),
        })
    }
    fn stream(&self) -> Result<Stream, CudaError> {
        let _guard = self.current()?;
        let mut raw = ptr::null_mut();
        check(
            self.dispatch(),
            self.dispatch().stream_create(&mut raw, CU_STREAM_DEFAULT),
        )?;
        Ok(Stream {
            owner: self.clone(),
            raw,
            closed: AtomicBool::new(false),
        })
    }
    fn event(&self) -> Result<Event, CudaError> {
        let _guard = self.current()?;
        let mut raw = ptr::null_mut();
        check(
            self.dispatch(),
            self.dispatch().event_create(&mut raw, CU_EVENT_DEFAULT),
        )?;
        Ok(Event {
            owner: self.clone(),
            raw,
            closed: AtomicBool::new(false),
        })
    }
    fn pinned(&self, bytes: NonZeroUsize) -> Result<PinnedHostBuffer, CudaError> {
        if !self.dispatch().supports_pinned_host_memory() {
            return Err(CudaError::MissingSymbol("cuMemHostAlloc"));
        }
        let mut ptr = ptr::null_mut();
        check(
            self.dispatch(),
            self.dispatch().mem_host_alloc(&mut ptr, bytes.get(), 0),
        )?;
        Ok(PinnedHostBuffer {
            owner: self.clone(),
            ptr: ptr.cast(),
            bytes: bytes.get(),
            closed: AtomicBool::new(false),
        })
    }
    fn module_from_ptx_with_options(
        &self,
        ptx: &CStr,
        options: ModuleLoadOptions,
    ) -> Result<CudaModule, CudaError> {
        options.validate()?;
        let _guard = self.current()?;
        let mut raw = ptr::null_mut();
        let d = self.dispatch();
        if !d.supports_module_load_data_ex() {
            if options.capture_logs {
                return Err(CudaError::MissingSymbol("cuModuleLoadDataEx"));
            }
            check(d, d.module_load_data(&mut raw, ptx.as_ptr().cast()))?;
            return Ok(CudaModule {
                owner: self.clone(),
                raw,
                closed: AtomicBool::new(false),
                metadata: ModuleLoadMetadata {
                    used_load_data_ex: false,
                    info_log: String::new(),
                    error_log: String::new(),
                },
            });
        }
        let mut info = vec![0u8; options.log_bytes];
        let mut error = vec![0u8; options.log_bytes];
        let mut info_size = info.len();
        let mut error_size = error.len();
        let mut opt = options.optimization_level;
        let keys = [
            CU_JIT_OPTIMIZATION_LEVEL,
            CU_JIT_TARGET_FROM_CUCONTEXT,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut values = [
            (&mut opt as *mut u32).cast(),
            ptr::null_mut(),
            info.as_mut_ptr().cast(),
            (&mut info_size as *mut usize).cast(),
            error.as_mut_ptr().cast(),
            (&mut error_size as *mut usize).cast(),
        ];
        let result = d.module_load_data_ex(&mut raw, ptx.as_ptr().cast(), &keys, &mut values);
        if result != CUDA_SUCCESS {
            return Err(CudaError::JitCompile {
                code: result,
                name: d
                    .error_name(result)
                    .unwrap_or_else(|| "CUDA_ERROR_UNKNOWN".into()),
                message: d.error_string(result).unwrap_or_default(),
                info_log: jit_log(&info),
                error_log: jit_log(&error),
            });
        }
        Ok(CudaModule {
            owner: self.clone(),
            raw,
            closed: AtomicBool::new(false),
            metadata: ModuleLoadMetadata {
                used_load_data_ex: true,
                info_log: jit_log(&info),
                error_log: jit_log(&error),
            },
        })
    }
}
fn jit_log(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
/// Restores the prior context even while unwinding.
pub struct ContextGuard {
    context: Context,
    previous: CuContext,
    restore: bool,
}
impl Drop for ContextGuard {
    fn drop(&mut self) {
        if self.restore {
            let _ = self
                .context
                .inner
                .driver
                .0
                .dispatch
                .ctx_set_current(self.previous);
        }
    }
}

pub struct DeviceBuffer {
    owner: Owner,
    ptr: CuDevicePtr,
    bytes: usize,
    closed: AtomicBool,
}
/// A checked logical view of a CUDA allocation.  In particular, a view made
/// from a pool lease never exposes the allocation's size-class capacity.
#[derive(Clone, Copy)]
pub struct BufferView<'a> {
    descriptor: CheckedBufferDescriptor<'a>,
    pooled: bool,
    primary_lease: Option<&'a PrimaryBufferLease>,
}
/// Private common denominator for direct/owned and primary-only views.  This
/// deliberately does not expose ownership or a raw pointer outside this module.
#[derive(Clone, Copy)]
struct CheckedBufferDescriptor<'a> {
    owner: ViewOwner<'a>,
    ptr: CuDevicePtr,
    bytes: usize,
}
#[derive(Clone, Copy)]
enum ViewOwner<'a> {
    Mixed(&'a Owner),
    Primary(&'a PrimaryContext),
}
impl ViewOwner<'_> {
    fn device(self) -> DeviceId {
        match self {
            Self::Mixed(x) => x.device(),
            Self::Primary(x) => x.device(),
        }
    }
    fn dispatch(&self) -> &dyn Dispatch {
        match self {
            Self::Mixed(x) => x.dispatch(),
            Self::Primary(x) => x.0.driver.0.dispatch.as_ref(),
        }
    }
    fn current(self) -> Result<OwnerGuard, CudaError> {
        match self {
            Self::Mixed(x) => x.current(),
            Self::Primary(x) => x.enter().map(OwnerGuard::Primary),
        }
    }
    fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Mixed(a), Self::Mixed(b)) => a.same(b),
            (Self::Primary(a), Self::Primary(b)) => Arc::ptr_eq(&a.0, &b.0),
            (Self::Mixed(Owner::Primary(a)), Self::Primary(b))
            | (Self::Primary(b), Self::Mixed(Owner::Primary(a))) => Arc::ptr_eq(&a.0, &b.0),
            _ => false,
        }
    }
    fn belongs_to_primary(self, primary: &PrimaryContext) -> bool {
        match self {
            Self::Primary(x) => Arc::ptr_eq(&x.0, &primary.0),
            Self::Mixed(Owner::Primary(x)) => Arc::ptr_eq(&x.0, &primary.0),
            Self::Mixed(Owner::Owned(_)) => false,
        }
    }
}
impl CheckedBufferDescriptor<'_> {
    fn range(&self, offset: usize, bytes: usize) -> Result<CuDevicePtr, CudaError> {
        let end = offset.checked_add(bytes).ok_or(CudaError::Overflow)?;
        if end > self.bytes {
            return Err(CudaError::InvalidArgument("range exceeds leased buffer"));
        }
        self.ptr
            .checked_add(u64::try_from(offset).map_err(|_| CudaError::Overflow)?)
            .ok_or(CudaError::Overflow)
    }
}
impl BufferView<'_> {
    pub fn len(&self) -> usize {
        self.descriptor.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.descriptor.bytes == 0
    }
    pub fn device(&self) -> DeviceId {
        self.descriptor.owner.device()
    }
    pub fn copy_from(&self, offset: usize, src: &[u8]) -> Result<(), CudaError> {
        self.range(offset, src.len())?;
        let ptr = self.descriptor.range(offset, src.len())?;
        let _guard = self.descriptor.owner.current()?;
        let d = self.descriptor.owner.dispatch();
        check(d, d.memcpy_htod(ptr, src.as_ptr().cast(), src.len()))
    }
    pub fn copy_to(&self, offset: usize, dst: &mut [u8]) -> Result<(), CudaError> {
        self.range(offset, dst.len())?;
        let ptr = self.descriptor.range(offset, dst.len())?;
        let _guard = self.descriptor.owner.current()?;
        let d = self.descriptor.owner.dispatch();
        check(d, d.memcpy_dtoh(dst.as_mut_ptr().cast(), ptr, dst.len()))
    }
    pub fn copy_from_view(
        &self,
        offset: usize,
        src: &BufferView<'_>,
        src_offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        self.range(offset, bytes)?;
        src.range(src_offset, bytes)?;
        if !self.descriptor.owner.same(src.descriptor.owner) {
            return Err(CudaError::WrongDevice {
                expected: self.device(),
                actual: src.device(),
            });
        }
        let dst = self.descriptor.range(offset, bytes)?;
        let src = src.descriptor.range(src_offset, bytes)?;
        let _guard = self.descriptor.owner.current()?;
        let d = self.descriptor.owner.dispatch();
        check(d, d.memcpy_dtod(dst, src, bytes))
    }
    pub(crate) fn device_ptr(&self) -> Result<CuDevicePtr, CudaError> {
        self.range(0, 0)?;
        self.range(0, 0)?;
        Ok(self.descriptor.ptr)
    }
    pub(crate) fn belongs_to_primary(&self, primary: &PrimaryContext) -> bool {
        self.descriptor.owner.belongs_to_primary(primary)
    }
    pub(crate) fn is_pooled(&self) -> bool {
        self.pooled
    }
    pub(crate) fn primary_lease(&self) -> Option<&PrimaryBufferLease> {
        self.primary_lease
    }
    pub fn copy_from_pinned_async<'a>(
        &'a self,
        offset: usize,
        src: &'a PinnedHostBuffer,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<Transfer<'a>, CudaError> {
        self.range(offset, bytes)?;
        self.async_check(
            ViewOwner::Mixed(&src.owner).same(self.descriptor.owner)
                && stream.same_view_owner(self.descriptor.owner),
            bytes,
        )?;
        let dst = self.descriptor.range(offset, bytes)?;
        let src_ptr = src.range(src_offset, bytes)?;
        let event = stream.event_for_view_owner(self.descriptor.owner)?;
        let d = self.descriptor.owner.dispatch();
        check(
            d,
            d.memcpy_htod_async(dst, src_ptr.cast(), bytes, stream.raw),
        )?;
        event.record(stream)?;
        Ok(Transfer {
            event,
            _stream: stream,
            _device_a: *self,
            _device_b: None,
            _host: Some(src),
            complete: false,
        })
    }
    pub fn copy_to_pinned_async<'a>(
        &'a self,
        offset: usize,
        dst: &'a PinnedHostBuffer,
        dst_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<Transfer<'a>, CudaError> {
        self.range(offset, bytes)?;
        self.async_check(
            ViewOwner::Mixed(&dst.owner).same(self.descriptor.owner)
                && stream.same_view_owner(self.descriptor.owner),
            bytes,
        )?;
        let src_ptr = self.descriptor.range(offset, bytes)?;
        let dst_ptr = dst.range(dst_offset, bytes)?;
        let event = stream.event_for_view_owner(self.descriptor.owner)?;
        let d = self.descriptor.owner.dispatch();
        check(
            d,
            d.memcpy_dtoh_async(dst_ptr.cast(), src_ptr, bytes, stream.raw),
        )?;
        event.record(stream)?;
        Ok(Transfer {
            event,
            _stream: stream,
            _device_a: *self,
            _device_b: None,
            _host: Some(dst),
            complete: false,
        })
    }
    pub fn copy_from_view_async<'a>(
        &'a self,
        offset: usize,
        src: &'a BufferView<'a>,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<Transfer<'a>, CudaError> {
        self.range(offset, bytes)?;
        src.range(src_offset, bytes)?;
        self.async_check(
            self.descriptor.owner.same(src.descriptor.owner)
                && stream.same_view_owner(self.descriptor.owner),
            bytes,
        )?;
        let dst = self.descriptor.range(offset, bytes)?;
        let src_ptr = src.descriptor.range(src_offset, bytes)?;
        let event = stream.event_for_view_owner(self.descriptor.owner)?;
        let d = self.descriptor.owner.dispatch();
        check(d, d.memcpy_dtod_async(dst, src_ptr, bytes, stream.raw))?;
        event.record(stream)?;
        Ok(Transfer {
            event,
            _stream: stream,
            _device_a: *self,
            _device_b: Some(*src),
            _host: None,
            complete: false,
        })
    }
    fn range(&self, offset: usize, bytes: usize) -> Result<(), CudaError> {
        self.descriptor.range(offset, bytes).map(|_| ())
    }
    fn async_check(&self, same_owner: bool, bytes: usize) -> Result<(), CudaError> {
        if bytes == 0 {
            return Err(CudaError::InvalidArgument("zero-length async copy"));
        }
        if !same_owner {
            return Err(CudaError::ContextMismatch);
        }
        if !self.descriptor.owner.dispatch().supports_async_transfers() {
            return Err(CudaError::MissingSymbol("cuMemcpy*Async"));
        }
        Ok(())
    }
}
impl DeviceBuffer {
    /// Makes a full-capacity checked view for a directly owned allocation.
    pub fn view(&self) -> BufferView<'_> {
        BufferView {
            descriptor: CheckedBufferDescriptor {
                owner: ViewOwner::Mixed(&self.owner),
                ptr: self.ptr,
                bytes: self.bytes,
            },
            pooled: false,
            primary_lease: None,
        }
    }
}

/// Thread-affine owned-context device-memory cache.  Size classes are powers
/// of two (minimum 256 bytes), so a request wastes less than one class; best
/// fit is selected from the ordered cache.  `0` is rejected because CUDA has
/// no useful zero-byte allocation API.
pub struct CudaAllocator {
    owner: Owner,
    cached: std::cell::RefCell<std::collections::BTreeMap<usize, Vec<DeviceBuffer>>>,
    in_use: std::cell::Cell<usize>,
    peak: std::cell::Cell<usize>,
}
pub struct BufferLease {
    allocator: std::rc::Rc<CudaAllocator>,
    buffer: Option<DeviceBuffer>,
    bytes: usize,
}
fn size_class(bytes: usize) -> Result<usize, CudaError> {
    const MIN: usize = 256;
    if bytes == 0 {
        return Err(CudaError::InvalidArgument("zero-sized allocation"));
    }
    bytes
        .checked_next_power_of_two()
        .ok_or(CudaError::Overflow)
        .map(|n| n.max(MIN))
}
impl CudaAllocator {
    fn new(owner: Owner) -> std::rc::Rc<Self> {
        std::rc::Rc::new(Self {
            owner,
            cached: std::cell::RefCell::new(std::collections::BTreeMap::new()),
            in_use: std::cell::Cell::new(0),
            peak: std::cell::Cell::new(0),
        })
    }
    pub fn cached_bytes(&self) -> usize {
        self.cached.borrow().iter().map(|(n, v)| n * v.len()).sum()
    }
    pub fn in_use_bytes(&self) -> usize {
        self.in_use.get()
    }
    pub fn peak_bytes(&self) -> usize {
        self.peak.get()
    }
    pub fn allocate(
        self: &std::rc::Rc<Self>,
        bytes: NonZeroUsize,
    ) -> Result<BufferLease, CudaError> {
        let capacity = size_class(bytes.get())?;
        let buffer = self
            .cached
            .borrow_mut()
            .range_mut(capacity..)
            .next()
            .and_then(|(_, blocks)| blocks.pop())
            .map(Ok)
            .unwrap_or_else(|| {
                self.owner
                    .allocate(NonZeroUsize::new(capacity).expect("nonzero class"))
            });
        let buffer = match buffer {
            Ok(b) => b,
            Err(e) if is_oom(&e) => {
                self.trim()?;
                self.owner
                    .allocate(NonZeroUsize::new(capacity).expect("nonzero class"))?
            }
            Err(e) => return Err(e),
        };
        let now = self
            .in_use
            .get()
            .checked_add(bytes.get())
            .ok_or(CudaError::Overflow)?;
        self.in_use.set(now);
        self.peak.set(self.peak.get().max(now));
        Ok(BufferLease {
            allocator: self.clone(),
            buffer: Some(buffer),
            bytes: bytes.get(),
        })
    }
    pub fn trim(&self) -> Result<(), CudaError> {
        let detached = std::mem::take(&mut *self.cached.borrow_mut());
        drop(detached);
        Ok(())
    }
}
impl BufferLease {
    /// Returns a borrow-tied logical view. `release` cannot run while this
    /// view exists, making stale use unrepresentable in safe Rust.
    pub fn view(&self) -> Result<BufferView<'_>, CudaError> {
        Ok(BufferView {
            descriptor: {
                let buffer = self.buffer.as_ref().ok_or(CudaError::StaleLease)?;
                CheckedBufferDescriptor {
                    owner: ViewOwner::Mixed(&buffer.owner),
                    ptr: buffer.ptr,
                    bytes: self.bytes,
                }
            },
            pooled: true,
            primary_lease: None,
        })
    }
    pub fn release(mut self) {
        if let Some(buffer) = self.buffer.take() {
            let n = self.bytes;
            self.allocator.in_use.set(self.allocator.in_use.get() - n);
            self.allocator
                .cached
                .borrow_mut()
                .entry(buffer.len())
                .or_default()
                .push(buffer)
        }
    }
}
impl Drop for BufferLease {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let n = self.bytes;
            self.allocator.in_use.set(self.allocator.in_use.get() - n);
            self.allocator
                .cached
                .borrow_mut()
                .entry(buffer.len())
                .or_default()
                .push(buffer)
        }
    }
}

/// Send + Sync cache state used only by a retained primary context.  It stores
/// plain pointer/capacity records plus the shareable primary owner, never an
/// `Owner`/`DeviceBuffer` sum which could carry an owned context.
pub struct PrimaryCudaAllocator {
    primary: PrimaryContext,
    state: Mutex<PrimaryPoolState>,
}
struct PrimaryPoolState {
    cached: std::collections::BTreeMap<usize, Vec<Arc<PrimaryBlock>>>,
    cached_bytes: usize,
    deferred: Vec<DeferredPrimaryBlock>,
    deferred_bytes: usize,
    quarantined: Vec<Arc<PrimaryBlock>>,
    in_use: usize,
    reserved: usize,
    peak: usize,
    closed: bool,
}
struct DeferredPrimaryBlock {
    block: Arc<PrimaryBlock>,
    generation: u64,
    fences: Vec<Arc<PrimaryEventFence>>,
}
/// A primary-context-only physical allocation.  Unlike `DeviceBuffer`, this
/// can never contain the mixed, thread-affine `Owner` enum.  It is retained by
/// `Arc` so a future deferred-completion registry can own it independently of
/// a logical lease.
pub struct PrimaryBlock {
    primary: PrimaryContext,
    ptr: CuDevicePtr,
    capacity: usize,
    generation: std::sync::atomic::AtomicU64,
    closed: AtomicBool,
}
impl Drop for PrimaryBlock {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
pub struct PrimaryBufferLease {
    allocator: Arc<PrimaryCudaAllocator>,
    block: Option<Arc<PrimaryBlock>>,
    bytes: usize,
    generation: u64,
    fences: Mutex<Vec<Arc<PrimaryEventFence>>>,
}
impl PrimaryCudaAllocator {
    fn new(primary: PrimaryContext) -> Arc<Self> {
        Arc::new(Self {
            primary,
            state: Mutex::new(PrimaryPoolState {
                cached: Default::default(),
                cached_bytes: 0,
                deferred: Vec::new(),
                deferred_bytes: 0,
                quarantined: Vec::new(),
                in_use: 0,
                reserved: 0,
                peak: 0,
                closed: false,
            }),
        })
    }
    pub fn cached_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("primary allocator mutex poisoned")
            .cached_bytes
    }
    pub fn in_use_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("primary allocator mutex poisoned")
            .in_use
    }
    pub fn reserved_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("primary allocator mutex poisoned")
            .reserved
    }
    pub fn peak_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("primary allocator mutex poisoned")
            .peak
    }
    pub fn deferred_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("primary allocator mutex poisoned")
            .deferred_bytes
    }
    pub fn deferred_blocks(&self) -> usize {
        self.state
            .lock()
            .expect("primary allocator mutex poisoned")
            .deferred
            .len()
    }
    /// Nonblocking promotion. Driver event queries happen after the pool lock is released.
    pub fn collect_deferred(&self) -> Result<usize, CudaError> {
        let snapshot = {
            let state = self.state.lock().expect("primary allocator mutex poisoned");
            state
                .deferred
                .iter()
                .map(|entry| {
                    (
                        Arc::as_ptr(&entry.block) as usize,
                        entry.generation,
                        entry.fences.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut ready = Vec::new();
        for (key, generation, fences) in snapshot {
            if fences
                .iter()
                .try_fold(true, |all, fence| fence.query().map(|ready| all && ready))?
            {
                ready.push((key, generation));
            }
        }
        Ok(self.promote_deferred(&ready))
    }
    /// Blocking promotion. Driver waits happen after the pool lock is released.
    pub fn wait_deferred(&self) -> Result<usize, CudaError> {
        let snapshot = {
            let state = self.state.lock().expect("primary allocator mutex poisoned");
            state
                .deferred
                .iter()
                .map(|entry| {
                    (
                        Arc::as_ptr(&entry.block) as usize,
                        entry.generation,
                        entry.fences.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        for (_, _, fences) in &snapshot {
            for fence in fences {
                fence.wait()?;
            }
        }
        Ok(self.promote_deferred(
            &snapshot
                .into_iter()
                .map(|(key, generation, _)| (key, generation))
                .collect::<Vec<_>>(),
        ))
    }
    fn promote_deferred(&self, ready: &[(usize, u64)]) -> usize {
        let mut state = self.state.lock().expect("primary allocator mutex poisoned");
        let mut promoted = 0;
        let mut pending = Vec::new();
        for entry in std::mem::take(&mut state.deferred) {
            let key = Arc::as_ptr(&entry.block) as usize;
            if ready
                .iter()
                .any(|&(wanted, generation)| wanted == key && generation == entry.generation)
                && !state.closed
                && entry.block.generation.load(Ordering::Acquire) == entry.generation
            {
                state.deferred_bytes -= entry.block.capacity;
                state.cached_bytes += entry.block.capacity;
                state
                    .cached
                    .entry(entry.block.capacity)
                    .or_default()
                    .push(entry.block);
                promoted += 1;
            } else {
                pending.push(entry);
            }
        }
        state.deferred = pending;
        promoted
    }
    pub fn allocate(
        self: &Arc<Self>,
        bytes: NonZeroUsize,
    ) -> Result<PrimaryBufferLease, CudaError> {
        let requested = bytes.get();
        let capacity = size_class(requested)?;
        let block = {
            let mut state = self.state.lock().expect("primary allocator mutex poisoned");
            if state.closed {
                return Err(CudaError::Closed("primary allocator"));
            }
            let block = state
                .cached
                .range_mut(capacity..)
                .next()
                .and_then(|(_, v)| v.pop());
            if let Some(ref block) = block {
                state.cached_bytes -= block.capacity;
            }
            block
        };
        let reused = block.is_some();
        let block = if let Some(block) = block {
            block
        } else {
            match self
                .primary
                .allocate_primary_block(NonZeroUsize::new(capacity).expect("nonzero class"))
            {
                Ok(block) => Arc::new(block),
                Err(error) if is_oom(&error) => {
                    self.trim()?; // detach/free outside the pool lock, then retry once.
                    Arc::new(self.primary.allocate_primary_block(
                        NonZeroUsize::new(capacity).expect("nonzero class"),
                    )?)
                }
                Err(error) => return Err(error),
            }
        };
        let mut state = self.state.lock().expect("primary allocator mutex poisoned");
        // A concurrent close may only detach cached blocks; a live allocation remains valid.
        state.in_use = state
            .in_use
            .checked_add(requested)
            .ok_or(CudaError::Overflow)?;
        state.reserved = state
            .reserved
            .checked_add(if reused { 0 } else { capacity })
            .ok_or(CudaError::Overflow)?;
        state.peak = state.peak.max(state.in_use);
        let generation = block.generation.fetch_add(1, Ordering::AcqRel) + 1;
        Ok(PrimaryBufferLease {
            allocator: self.clone(),
            block: Some(block),
            bytes: requested,
            generation,
            fences: Mutex::new(Vec::new()),
        })
    }
    pub fn trim(&self) -> Result<(), CudaError> {
        let detached = {
            let mut state = self.state.lock().expect("primary allocator mutex poisoned");
            let cached = std::mem::take(&mut state.cached);
            state.reserved -= state.cached_bytes;
            state.cached_bytes = 0;
            cached.into_values().flatten().collect::<Vec<_>>()
        };
        drop(detached); // Driver frees occur after releasing the mutex.
        Ok(())
    }
    pub fn close(&self) -> Result<(), CudaError> {
        {
            let state = self.state.lock().expect("primary allocator mutex poisoned");
            if state.closed {
                return Ok(());
            }
        }
        self.wait_deferred()?;
        self.state
            .lock()
            .expect("primary allocator mutex poisoned")
            .closed = true;
        self.trim()?;
        // A record+sync failure has no completion proof.  Preserve the blocks
        // rather than free/reuse potentially in-flight device memory.
        let quarantined = std::mem::take(
            &mut self
                .state
                .lock()
                .expect("primary allocator mutex poisoned")
                .quarantined,
        );
        for block in quarantined {
            std::mem::forget(block);
        }
        Ok(())
    }
}
impl Drop for PrimaryCudaAllocator {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
impl PrimaryBufferLease {
    pub fn view(&self) -> Result<BufferView<'_>, CudaError> {
        Ok(BufferView {
            descriptor: {
                let block = self.block.as_ref().ok_or(CudaError::StaleLease)?;
                if block.generation.load(Ordering::Acquire) != self.generation
                    || block.closed.load(Ordering::Acquire)
                {
                    return Err(CudaError::StaleLease);
                }
                CheckedBufferDescriptor {
                    owner: ViewOwner::Primary(&block.primary),
                    ptr: block.ptr,
                    bytes: self.bytes,
                }
            },
            pooled: true,
            primary_lease: Some(self),
        })
    }
    pub fn release(mut self) {
        self.return_block();
    }
    /// Submits a directional primary-context peer copy. The returned token
    /// keeps both logical leases and the peer mapping live until completion.
    pub fn copy_from_peer_async<'a>(
        &'a self,
        dst_offset: usize,
        peer: &'a PeerAccess,
        src: &'a PrimaryBufferLease,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<PeerTransfer<'a>, CudaError> {
        if bytes == 0 {
            return Err(CudaError::InvalidArgument("zero-length peer copy"));
        }
        let dst = self.block.as_ref().ok_or(CudaError::StaleLease)?;
        let source = src.block.as_ref().ok_or(CudaError::StaleLease)?;
        if dst.generation.load(Ordering::Acquire) != self.generation
            || source.generation.load(Ordering::Acquire) != src.generation
        {
            return Err(CudaError::StaleLease);
        }
        let dst_end = dst_offset.checked_add(bytes).ok_or(CudaError::Overflow)?;
        let src_end = src_offset.checked_add(bytes).ok_or(CudaError::Overflow)?;
        if dst_end > self.bytes || src_end > src.bytes {
            return Err(CudaError::InvalidArgument(
                "peer copy range exceeds leased buffer",
            ));
        }
        if !peer.matches(&source.primary, &dst.primary) || !stream.belongs_to_primary(&dst.primary)
        {
            return Err(CudaError::ContextMismatch);
        }
        let d = dst.primary.0.driver.0.dispatch.as_ref();
        let dst_ptr = dst
            .ptr
            .checked_add(u64::try_from(dst_offset).map_err(|_| CudaError::Overflow)?)
            .ok_or(CudaError::Overflow)?;
        let src_ptr = source
            .ptr
            .checked_add(u64::try_from(src_offset).map_err(|_| CudaError::Overflow)?)
            .ok_or(CudaError::Overflow)?;
        if !d.supports_async_transfers() {
            return Err(CudaError::MissingSymbol("cuMemcpyPeerAsync"));
        }
        let fence = Arc::new(dst.primary.event_fence()?);
        let _guard = dst.primary.enter()?;
        check(
            d,
            d.memcpy_peer_async(
                dst_ptr,
                dst.primary.0.raw,
                src_ptr,
                source.primary.0.raw,
                bytes,
                stream.raw,
            ),
        )?;
        if let Err(error) = fence.record(stream) {
            if stream.synchronize().is_err() {
                self.quarantine();
                src.quarantine();
            }
            return Err(error);
        }
        self.attach_fence(fence.clone())?;
        src.attach_peer_fence(fence.clone())?;
        Ok(PeerTransfer {
            fence,
            _destination: self,
            _source: src,
            _peer: peer,
            _stream: stream,
            complete: false,
        })
    }
    pub(crate) fn attach_fence(&self, fence: Arc<PrimaryEventFence>) -> Result<(), CudaError> {
        let block = self.block.as_ref().ok_or(CudaError::StaleLease)?;
        fence.validate_owner(&block.primary)?;
        if block.generation.load(Ordering::Acquire) != self.generation {
            return Err(CudaError::StaleLease);
        }
        let mut fences = self
            .fences
            .lock()
            .expect("primary lease fence mutex poisoned");
        if !fences.iter().any(|old| Arc::ptr_eq(old, &fence)) {
            fences.push(fence);
        }
        Ok(())
    }
    fn attach_peer_fence(&self, fence: Arc<PrimaryEventFence>) -> Result<(), CudaError> {
        let block = self.block.as_ref().ok_or(CudaError::StaleLease)?;
        if block.generation.load(Ordering::Acquire) != self.generation {
            return Err(CudaError::StaleLease);
        }
        let mut fences = self
            .fences
            .lock()
            .expect("primary lease fence mutex poisoned");
        if !fences.iter().any(|old| Arc::ptr_eq(old, &fence)) {
            fences.push(fence);
        }
        Ok(())
    }
    pub(crate) fn primary(&self) -> Result<PrimaryContext, CudaError> {
        Ok(self
            .block
            .as_ref()
            .ok_or(CudaError::StaleLease)?
            .primary
            .clone())
    }
    pub(crate) fn quarantine(&self) {
        if let Some(block) = self.block.as_ref() {
            let mut state = self
                .allocator
                .state
                .lock()
                .expect("primary allocator mutex poisoned");
            // Marked by generation: return_block will transfer it to this list.
            if !state.quarantined.iter().any(|old| Arc::ptr_eq(old, block)) {
                state.quarantined.push(block.clone());
            }
        }
    }
    fn return_block(&mut self) {
        let Some(block) = self.block.take() else {
            return;
        };
        if self
            .allocator
            .state
            .lock()
            .expect("primary allocator mutex poisoned")
            .quarantined
            .iter()
            .any(|old| Arc::ptr_eq(old, &block))
        {
            return;
        }
        let mut state = self
            .allocator
            .state
            .lock()
            .expect("primary allocator mutex poisoned");
        state.in_use -= self.bytes;
        let fences = std::mem::take(
            &mut *self
                .fences
                .lock()
                .expect("primary lease fence mutex poisoned"),
        );
        if state.closed {
            state.reserved -= block.capacity;
            drop(state);
            drop(block);
            return;
        }
        if fences.is_empty() {
            state.cached_bytes += block.capacity;
            state.cached.entry(block.capacity).or_default().push(block);
        } else {
            state.deferred_bytes += block.capacity;
            state.deferred.push(DeferredPrimaryBlock {
                block,
                generation: self.generation,
                fences,
            });
        }
    }
}
/// Completion token for a primary peer copy. It is deliberately non-cloneable
/// and retains both allocator generations and the directional peer session.
#[must_use = "peer transfers must be queried or waited"]
pub struct PeerTransfer<'a> {
    fence: Arc<PrimaryEventFence>,
    _destination: &'a PrimaryBufferLease,
    _source: &'a PrimaryBufferLease,
    _peer: &'a PeerAccess,
    _stream: &'a Stream,
    complete: bool,
}
impl PeerTransfer<'_> {
    pub fn query(&mut self) -> Result<bool, CudaError> {
        if self.complete {
            return Ok(true);
        }
        let done = self.fence.query()?;
        self.complete = done;
        Ok(done)
    }
    pub fn wait(&mut self) -> Result<(), CudaError> {
        if !self.complete {
            self.fence.wait()?;
            self.complete = true;
        }
        Ok(())
    }
}
impl Drop for PeerTransfer<'_> {
    fn drop(&mut self) {
        let _ = self.wait();
    }
}
impl Drop for PrimaryBufferLease {
    fn drop(&mut self) {
        self.return_block();
    }
}
impl PrimaryBlock {
    pub fn len(&self) -> usize {
        self.capacity
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn close(&self) -> Result<(), CudaError> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let _guard = self.primary.enter()?;
        check(
            self.primary.0.driver.0.dispatch.as_ref(),
            self.primary.0.driver.0.dispatch.mem_free(self.ptr),
        )
    }
}
/// A primary-only event resource. It owns a retained primary context, so event
/// destruction happens before the final primary-context release. Owned and
/// mixed `Event` values intentionally remain !Send/!Sync.
pub struct PrimaryEventFence {
    primary: PrimaryContext,
    raw: usize,
    closed: AtomicBool,
}
impl PrimaryEventFence {
    fn live(&self) -> Result<(), CudaError> {
        if self.closed.load(Ordering::Acquire) {
            Err(CudaError::Closed("primary event fence"))
        } else {
            Ok(())
        }
    }
    pub fn query(&self) -> Result<bool, CudaError> {
        self.live()?;
        let _guard = self.primary.enter()?;
        let d = self.primary.0.driver.0.dispatch.as_ref();
        match d.event_query(self.raw as CuEvent) {
            CUDA_SUCCESS => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            code => check(d, code).map(|_| false),
        }
    }
    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        self.live()?;
        if !stream.belongs_to_primary(&self.primary) {
            return Err(CudaError::ContextMismatch);
        }
        let _guard = self.primary.enter()?;
        let d = self.primary.0.driver.0.dispatch.as_ref();
        check(d, d.event_record(self.raw as CuEvent, stream.raw))
    }
    pub fn wait(&self) -> Result<(), CudaError> {
        self.live()?;
        let _guard = self.primary.enter()?;
        let d = self.primary.0.driver.0.dispatch.as_ref();
        check(d, d.event_sync(self.raw as CuEvent))
    }
    pub fn validate_owner(&self, primary: &PrimaryContext) -> Result<(), CudaError> {
        self.live()?;
        if Arc::ptr_eq(&self.primary.0, &primary.0) {
            Ok(())
        } else {
            Err(CudaError::ContextMismatch)
        }
    }
    pub fn close(&self) -> Result<(), CudaError> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let _guard = self.primary.enter()?;
        let d = self.primary.0.driver.0.dispatch.as_ref();
        check(d, d.event_destroy(self.raw as CuEvent))
    }
}
impl Drop for PrimaryEventFence {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
fn is_oom(error: &CudaError) -> bool {
    matches!(error, CudaError::Driver { code: 2, .. })
}
impl DeviceBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn device(&self) -> DeviceId {
        self.owner.device()
    }
    pub(crate) fn belongs_to_primary(&self, primary: &PrimaryContext) -> bool {
        matches!(&self.owner, Owner::Primary(owner) if Arc::ptr_eq(&owner.0, &primary.0))
    }
    fn live(&self) -> Result<(), CudaError> {
        if self.closed.load(Ordering::Acquire) {
            Err(CudaError::Closed("buffer"))
        } else {
            Ok(())
        }
    }
    fn range(&self, offset: usize, bytes: usize) -> Result<CuDevicePtr, CudaError> {
        self.live()?;
        let end = offset.checked_add(bytes).ok_or(CudaError::Overflow)?;
        if end > self.bytes {
            return Err(CudaError::InvalidArgument("copy range exceeds buffer"));
        }
        self.ptr
            .checked_add(u64::try_from(offset).map_err(|_| CudaError::Overflow)?)
            .ok_or(CudaError::Overflow)
    }
    pub fn copy_from(&self, offset: usize, src: &[u8]) -> Result<(), CudaError> {
        let _guard = self.owner.current()?;
        let ptr = self.range(offset, src.len())?;
        let d = self.owner.dispatch();
        check(d, d.memcpy_htod(ptr, src.as_ptr().cast(), src.len()))
    }
    pub fn copy_to(&self, offset: usize, dst: &mut [u8]) -> Result<(), CudaError> {
        let _guard = self.owner.current()?;
        let ptr = self.range(offset, dst.len())?;
        let d = self.owner.dispatch();
        check(d, d.memcpy_dtoh(dst.as_mut_ptr().cast(), ptr, dst.len()))
    }
    pub fn copy_from_device(
        &self,
        offset: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        if !self.owner.same(&src.owner) {
            return Err(CudaError::WrongDevice {
                expected: self.device(),
                actual: src.device(),
            });
        }
        let _guard = self.owner.current()?;
        let dst = self.range(offset, bytes)?;
        let src = src.range(src_offset, bytes)?;
        let d = self.owner.dispatch();
        check(d, d.memcpy_dtod(dst, src, bytes))
    }
    pub fn copy_from_pinned_async<'a>(
        &'a self,
        offset: usize,
        src: &'a PinnedHostBuffer,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<Transfer<'a>, CudaError> {
        self.async_check(
            src.owner.same(&self.owner) && stream.owner.same(&self.owner),
            bytes,
        )?;
        let dst = self.range(offset, bytes)?;
        let src_ptr = src.range(src_offset, bytes)?;
        let event = self.async_event(stream)?;
        check(
            self.owner.dispatch(),
            self.owner
                .dispatch()
                .memcpy_htod_async(dst, src_ptr.cast(), bytes, stream.raw),
        )?;
        event.record(stream)?;
        Ok(Transfer {
            event,
            _stream: stream,
            _device_a: self.view(),
            _device_b: None,
            _host: Some(src),
            complete: false,
        })
    }
    pub fn copy_to_pinned_async<'a>(
        &'a self,
        offset: usize,
        dst: &'a PinnedHostBuffer,
        dst_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<Transfer<'a>, CudaError> {
        self.async_check(
            dst.owner.same(&self.owner) && stream.owner.same(&self.owner),
            bytes,
        )?;
        let src = self.range(offset, bytes)?;
        let dst_ptr = dst.range(dst_offset, bytes)?;
        let event = self.async_event(stream)?;
        check(
            self.owner.dispatch(),
            self.owner
                .dispatch()
                .memcpy_dtoh_async(dst_ptr.cast(), src, bytes, stream.raw),
        )?;
        event.record(stream)?;
        Ok(Transfer {
            event,
            _stream: stream,
            _device_a: self.view(),
            _device_b: None,
            _host: Some(dst),
            complete: false,
        })
    }
    pub fn copy_from_device_async<'a>(
        &'a self,
        offset: usize,
        src: &'a DeviceBuffer,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<Transfer<'a>, CudaError> {
        self.async_check(
            self.owner.same(&src.owner) && stream.owner.same(&self.owner),
            bytes,
        )?;
        let dst = self.range(offset, bytes)?;
        let src_ptr = src.range(src_offset, bytes)?;
        let event = self.async_event(stream)?;
        check(
            self.owner.dispatch(),
            self.owner
                .dispatch()
                .memcpy_dtod_async(dst, src_ptr, bytes, stream.raw),
        )?;
        event.record(stream)?;
        Ok(Transfer {
            event,
            _stream: stream,
            _device_a: self.view(),
            _device_b: Some(src.view()),
            _host: None,
            complete: false,
        })
    }
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn copy_from_pinned_async_profiled<'a>(
        &'a self,
        session: &ProfilingSession,
        name: impl Into<String>,
        primary: &PrimaryContext,
        offset: usize,
        src: &'a PinnedHostBuffer,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<ProfiledTransfer<'a>, CudaError> {
        if !session.is_enabled() {
            return self
                .copy_from_pinned_async(offset, src, src_offset, bytes, stream)
                .map(ProfiledTransfer::Plain);
        }
        self.profile_copy_preflight(primary, stream, src.belongs_to_primary(primary), bytes)?;
        self.range(offset, bytes)?;
        src.range(src_offset, bytes)?;
        let mut timing =
            self.begin_copy_timing(session, name, primary, stream, OperationKind::HtoD, bytes)?;
        match self.copy_from_pinned_async(offset, src, src_offset, bytes, stream) {
            Ok(transfer) => match timing.record_end(stream) {
                Ok(()) => Ok(ProfiledTransfer::Timed { transfer, timing }),
                Err(error) => Err(profile_cuda_error(error)),
            },
            Err(error) => {
                timing.fail_due_to(TimingError::Cuda(error.clone()));
                Err(error)
            }
        }
    }
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn copy_to_pinned_async_profiled<'a>(
        &'a self,
        session: &ProfilingSession,
        name: impl Into<String>,
        primary: &PrimaryContext,
        offset: usize,
        dst: &'a PinnedHostBuffer,
        dst_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<ProfiledTransfer<'a>, CudaError> {
        if !session.is_enabled() {
            return self
                .copy_to_pinned_async(offset, dst, dst_offset, bytes, stream)
                .map(ProfiledTransfer::Plain);
        }
        self.profile_copy_preflight(primary, stream, dst.belongs_to_primary(primary), bytes)?;
        self.range(offset, bytes)?;
        dst.range(dst_offset, bytes)?;
        let mut timing =
            self.begin_copy_timing(session, name, primary, stream, OperationKind::DtoH, bytes)?;
        match self.copy_to_pinned_async(offset, dst, dst_offset, bytes, stream) {
            Ok(transfer) => match timing.record_end(stream) {
                Ok(()) => Ok(ProfiledTransfer::Timed { transfer, timing }),
                Err(error) => Err(profile_cuda_error(error)),
            },
            Err(error) => {
                timing.fail_due_to(TimingError::Cuda(error.clone()));
                Err(error)
            }
        }
    }
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn copy_from_device_async_profiled<'a>(
        &'a self,
        session: &ProfilingSession,
        name: impl Into<String>,
        primary: &PrimaryContext,
        offset: usize,
        src: &'a DeviceBuffer,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<ProfiledTransfer<'a>, CudaError> {
        if !session.is_enabled() {
            return self
                .copy_from_device_async(offset, src, src_offset, bytes, stream)
                .map(ProfiledTransfer::Plain);
        }
        self.profile_copy_preflight(primary, stream, src.belongs_to_primary(primary), bytes)?;
        self.range(offset, bytes)?;
        src.range(src_offset, bytes)?;
        let mut timing =
            self.begin_copy_timing(session, name, primary, stream, OperationKind::DtoD, bytes)?;
        match self.copy_from_device_async(offset, src, src_offset, bytes, stream) {
            Ok(transfer) => match timing.record_end(stream) {
                Ok(()) => Ok(ProfiledTransfer::Timed { transfer, timing }),
                Err(error) => Err(profile_cuda_error(error)),
            },
            Err(error) => {
                timing.fail_due_to(TimingError::Cuda(error.clone()));
                Err(error)
            }
        }
    }
    #[allow(dead_code)]
    fn profile_copy_preflight(
        &self,
        primary: &PrimaryContext,
        stream: &Stream,
        other_primary: bool,
        bytes: usize,
    ) -> Result<(), CudaError> {
        if !self.belongs_to_primary(primary)
            || !stream.belongs_to_primary(primary)
            || !other_primary
        {
            return Err(CudaError::ContextMismatch);
        }
        self.async_check(true, bytes)
    }
    #[allow(dead_code)]
    fn begin_copy_timing<'a>(
        &self,
        session: &ProfilingSession,
        name: impl Into<String>,
        primary: &PrimaryContext,
        stream: &'a Stream,
        kind: OperationKind,
        bytes: usize,
    ) -> Result<TimedSample<'a>, CudaError> {
        TimedSample::begin(
            session,
            Metadata {
                kind,
                name: name.into(),
                owner: primary.identity(),
                device: primary.device(),
                stream: stream.identity(),
                bytes: Some(bytes),
                geometry: None,
                source_key: None,
            },
            primary,
            stream,
            Arc::new(()),
        )
        .map_err(profile_cuda_error)?
        .ok_or(CudaError::InvalidArgument("enabled profiling session"))
    }
    fn async_check(&self, same_owner: bool, bytes: usize) -> Result<(), CudaError> {
        self.live()?;
        if bytes == 0 {
            return Err(CudaError::InvalidArgument("zero-length async copy"));
        }
        if !same_owner {
            return Err(CudaError::ContextMismatch);
        }
        if !self.owner.dispatch().supports_async_transfers() {
            return Err(CudaError::MissingSymbol("cuMemcpy*Async"));
        }
        Ok(())
    }
    fn async_event(&self, stream: &Stream) -> Result<Event, CudaError> {
        stream.live()?;
        self.owner.event()
    }
    pub fn close(&self) -> Result<(), CudaError> {
        self.live()?;
        self.closed.store(true, Ordering::Release);
        let _guard = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().mem_free(self.ptr),
        )
    }
}
impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel)
            && let Ok(_g) = self.owner.current()
        {
            let _ = self.owner.dispatch().mem_free(self.ptr);
        }
    }
}

/// Page-locked host memory owned by a CUDA context owner. It is deliberately
/// !Send/!Sync because the sealed owner sum can contain a thread-affine owned
/// context. Access is bounds checked; the raw pointer never escapes publicly.
pub struct PinnedHostBuffer {
    owner: Owner,
    ptr: *mut u8,
    bytes: usize,
    closed: AtomicBool,
}
impl PinnedHostBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    #[allow(dead_code)]
    fn belongs_to_primary(&self, primary: &PrimaryContext) -> bool {
        matches!(&self.owner, Owner::Primary(owner) if Arc::ptr_eq(&owner.0, &primary.0))
    }
    fn live(&self) -> Result<(), CudaError> {
        if self.closed.load(Ordering::Acquire) {
            Err(CudaError::Closed("pinned host buffer"))
        } else {
            Ok(())
        }
    }
    fn range(&self, offset: usize, bytes: usize) -> Result<*mut u8, CudaError> {
        self.live()?;
        let end = offset.checked_add(bytes).ok_or(CudaError::Overflow)?;
        if bytes == 0 || end > self.bytes {
            return Err(CudaError::InvalidArgument("pinned host range"));
        }
        // SAFETY: range is checked against this Driver-owned allocation.
        Ok(unsafe { self.ptr.add(offset) })
    }
    pub fn write(&self, offset: usize, src: &[u8]) -> Result<(), CudaError> {
        let dst = self.range(offset, src.len())?;
        // SAFETY: `range` checked the destination and slices are non-overlapping.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        Ok(())
    }
    pub fn read(&self, offset: usize, dst: &mut [u8]) -> Result<(), CudaError> {
        let src = self.range(offset, dst.len())?;
        // SAFETY: `range` checked the source and slices are non-overlapping.
        unsafe {
            ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
    pub fn close(&self) -> Result<(), CudaError> {
        self.live()?;
        self.closed.store(true, Ordering::Release);
        check(
            self.owner.dispatch(),
            self.owner.dispatch().mem_free_host(self.ptr.cast()),
        )
    }
}
impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            let _ = self.owner.dispatch().mem_free_host(self.ptr.cast());
        }
    }
}

/// Non-cloneable completion token. Its borrows prevent the stream, device
/// buffers, and pinned staging allocation from being closed or dropped early.
/// `Drop` waits best-effort; callers needing an error must call `wait`.
#[must_use = "async transfers must be queried or waited"]
pub struct Transfer<'a> {
    event: Event,
    _stream: &'a Stream,
    _device_a: BufferView<'a>,
    _device_b: Option<BufferView<'a>>,
    _host: Option<&'a PinnedHostBuffer>,
    complete: bool,
}
impl Transfer<'_> {
    pub(crate) fn mark_complete(&mut self) {
        self.complete = true;
    }
    pub fn query(&mut self) -> Result<bool, CudaError> {
        if self.complete {
            return Ok(true);
        }
        let done = self.event.query()?;
        self.complete = done;
        Ok(done)
    }
    pub fn wait(&mut self) -> Result<(), CudaError> {
        if !self.complete {
            self.event.synchronize()?;
            self.complete = true;
        }
        Ok(())
    }
    pub fn close(mut self) -> Result<(), CudaError> {
        self.wait()
    }
}
impl Drop for Transfer<'_> {
    fn drop(&mut self) {
        if !self.complete {
            let _ = self.event.synchronize();
        }
    }
}

/// Crate-private completion token for an async primary copy with optional
/// profiling. Its timing end marker follows the copy on the same stream, so it
/// is the sole synchronization authority for the profiled branch.
#[allow(dead_code)]
pub(crate) enum ProfiledTransfer<'a> {
    Plain(Transfer<'a>),
    Timed {
        transfer: Transfer<'a>,
        timing: TimedSample<'a>,
    },
}
#[allow(dead_code)]
impl ProfiledTransfer<'_> {
    pub(crate) fn query(&mut self) -> Result<Option<u64>, CudaError> {
        match self {
            Self::Plain(transfer) => transfer.query().map(|ready| ready.then_some(0)),
            Self::Timed { transfer, timing } => match timing.query().map_err(profile_cuda_error)? {
                Some(duration) => {
                    transfer.mark_complete();
                    Ok(Some(duration))
                }
                None => Ok(None),
            },
        }
    }
    pub(crate) fn wait(&mut self) -> Result<Option<u64>, CudaError> {
        match self {
            Self::Plain(transfer) => {
                transfer.wait()?;
                Ok(None)
            }
            Self::Timed { transfer, timing } => {
                let duration = timing.wait().map_err(profile_cuda_error)?;
                transfer.mark_complete();
                Ok(Some(duration))
            }
        }
    }
    pub(crate) fn collect(self) -> Result<Option<u64>, CudaError> {
        match self {
            Self::Plain(mut transfer) => {
                transfer.wait()?;
                Ok(None)
            }
            Self::Timed {
                mut transfer,
                timing,
            } => {
                let duration = timing.collect().map_err(profile_cuda_error)?;
                transfer.mark_complete();
                Ok(Some(duration))
            }
        }
    }
}
fn profile_cuda_error(error: TimingError) -> CudaError {
    match error {
        TimingError::Cuda(error) => error,
        _ => CudaError::InvalidArgument("profiling timing state"),
    }
}

/// A primary-context stream-capture session. It borrows its stream and every
/// explicitly retained resource, preventing premature cleanup through replay.
pub struct Capture<'a> {
    stream: &'a Stream,
    retained: Vec<CaptureResource<'a>>,
    active: bool,
}
#[allow(dead_code)] // retained solely to keep captured Driver pointers alive.
enum CaptureResource<'a> {
    Buffer(&'a DeviceBuffer),
    View(BufferView<'a>),
    Pinned(&'a PinnedHostBuffer),
}
pub struct CudaGraph<'a> {
    owner: Owner,
    raw: CuGraph,
    retained: Vec<CaptureResource<'a>>,
    closed: AtomicBool,
}
pub struct GraphExec<'a> {
    owner: Owner,
    raw: CuGraphExec,
    #[allow(dead_code)] // keeps all captured resources alive through graph-exec drop.
    retained: Vec<CaptureResource<'a>>,
    closed: AtomicBool,
}
impl<'a> Capture<'a> {
    /// Retains the logical view itself, so the originating lease cannot be
    /// released while a captured graph or graph executable can replay it.
    pub fn retain_view(&mut self, buffer: BufferView<'a>) -> Result<(), CudaError> {
        if !self.active {
            return Err(CudaError::Closed("capture"));
        }
        if !ViewOwner::Mixed(&self.stream.owner).same(buffer.descriptor.owner) {
            return Err(CudaError::ContextMismatch);
        }
        self.retained.push(CaptureResource::View(buffer));
        Ok(())
    }
    pub fn retain_buffer(&mut self, buffer: &'a DeviceBuffer) -> Result<(), CudaError> {
        if !self.active {
            return Err(CudaError::Closed("capture"));
        }
        if !self.stream.owner.same(&buffer.owner) {
            return Err(CudaError::ContextMismatch);
        }
        self.retained.push(CaptureResource::Buffer(buffer));
        Ok(())
    }
    pub fn retain_pinned(&mut self, pinned: &'a PinnedHostBuffer) -> Result<(), CudaError> {
        if !self.active {
            return Err(CudaError::Closed("capture"));
        }
        if !self.stream.owner.same(&pinned.owner) {
            return Err(CudaError::ContextMismatch);
        }
        self.retained.push(CaptureResource::Pinned(pinned));
        Ok(())
    }
    pub fn finish(mut self) -> Result<CudaGraph<'a>, CudaError> {
        if !self.active {
            return Err(CudaError::Closed("capture"));
        }
        let mut raw = ptr::null_mut();
        check(
            self.stream.owner.dispatch(),
            self.stream
                .owner
                .dispatch()
                .stream_end_capture(self.stream.raw, &mut raw),
        )?;
        self.active = false;
        Ok(CudaGraph {
            owner: self.stream.owner.clone(),
            raw,
            retained: std::mem::take(&mut self.retained),
            closed: AtomicBool::new(false),
        })
    }
}
impl Drop for Capture<'_> {
    fn drop(&mut self) {
        if self.active {
            let mut graph = ptr::null_mut();
            let _ = self
                .stream
                .owner
                .dispatch()
                .stream_end_capture(self.stream.raw, &mut graph);
            if !graph.is_null() {
                let _ = self.stream.owner.dispatch().graph_destroy(graph);
            }
        }
    }
}
impl<'a> CudaGraph<'a> {
    pub fn instantiate(mut self) -> Result<GraphExec<'a>, CudaError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(CudaError::Closed("graph"));
        }
        let mut raw = ptr::null_mut();
        check(
            self.owner.dispatch(),
            self.owner.dispatch().graph_instantiate(&mut raw, self.raw),
        )?;
        let destroyed = check(
            self.owner.dispatch(),
            self.owner.dispatch().graph_destroy(self.raw),
        );
        self.closed.store(true, Ordering::Release);
        destroyed?;
        Ok(GraphExec {
            owner: self.owner.clone(),
            raw,
            retained: std::mem::take(&mut self.retained),
            closed: AtomicBool::new(false),
        })
    }
}
impl Drop for CudaGraph<'_> {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            let _ = self.owner.dispatch().graph_destroy(self.raw);
        }
    }
}
impl<'a> GraphExec<'a> {
    pub fn launch(&self, stream: &Stream) -> Result<(), CudaError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(CudaError::Closed("graph exec"));
        }
        if !self.owner.same(&stream.owner) {
            return Err(CudaError::ContextMismatch);
        }
        check(
            self.owner.dispatch(),
            self.owner.dispatch().graph_launch(self.raw, stream.raw),
        )
    }
    pub fn close(&self) -> Result<(), CudaError> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Err(CudaError::Closed("graph exec"));
        }
        check(
            self.owner.dispatch(),
            self.owner.dispatch().graph_exec_destroy(self.raw),
        )
    }
}
impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            let _ = self.owner.dispatch().graph_exec_destroy(self.raw);
        }
    }
}

pub struct Stream {
    owner: Owner,
    raw: CuStream,
    closed: AtomicBool,
}
impl Stream {
    fn same_view_owner(&self, owner: ViewOwner<'_>) -> bool {
        ViewOwner::Mixed(&self.owner).same(owner)
    }
    fn event_for_view_owner(&self, owner: ViewOwner<'_>) -> Result<Event, CudaError> {
        if !self.same_view_owner(owner) {
            return Err(CudaError::ContextMismatch);
        }
        self.owner.event()
    }
    #[allow(dead_code)] // stable metadata identity for crate-private profiling.
    pub(crate) fn identity(&self) -> usize {
        self as *const Self as usize
    }
    #[allow(dead_code)] // validates crate-private profiled PTX launches.
    #[allow(dead_code)] // validates crate-private profiled PTX launches.
    pub(crate) fn belongs_to_primary(&self, primary: &PrimaryContext) -> bool {
        matches!(&self.owner, Owner::Primary(owner) if Arc::ptr_eq(&owner.0, &primary.0))
    }
    pub(crate) fn same_stream(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
    fn live(&self) -> Result<(), CudaError> {
        if self.closed.load(Ordering::Acquire) {
            Err(CudaError::Closed("stream"))
        } else {
            Ok(())
        }
    }
    pub fn begin_capture(&self) -> Result<Capture<'_>, CudaError> {
        self.live()?;
        if !self.owner.is_primary() {
            return Err(CudaError::InvalidArgument(
                "graphs require a primary-context stream",
            ));
        }
        if !self.owner.dispatch().supports_graphs() {
            return Err(CudaError::MissingSymbol("cuStreamBeginCapture"));
        }
        check(
            self.owner.dispatch(),
            self.owner.dispatch().stream_begin_capture(self.raw, 0),
        )?;
        Ok(Capture {
            stream: self,
            retained: Vec::new(),
            active: true,
        })
    }
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.live()?;
        let _g = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().stream_sync(self.raw),
        )
    }
    pub fn wait(&self, event: &Event) -> Result<(), CudaError> {
        self.live()?;
        event.live()?;
        if !self.owner.same(&event.owner) {
            return Err(CudaError::ContextMismatch);
        }
        let _g = self.owner.current()?;
        let d = self.owner.dispatch();
        check(d, d.stream_wait_event(self.raw, event.raw, 0))
    }
    pub fn close(&self) -> Result<(), CudaError> {
        self.live()?;
        self.closed.store(true, Ordering::Release);
        let _g = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().stream_destroy(self.raw),
        )
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel)
            && let Ok(_g) = self.owner.current()
        {
            let _ = self.owner.dispatch().stream_destroy(self.raw);
        }
    }
}
pub struct Event {
    owner: Owner,
    raw: CuEvent,
    closed: AtomicBool,
}
impl Event {
    fn live(&self) -> Result<(), CudaError> {
        if self.closed.load(Ordering::Acquire) {
            Err(CudaError::Closed("event"))
        } else {
            Ok(())
        }
    }
    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        self.live()?;
        stream.live()?;
        if !self.owner.same(&stream.owner) {
            return Err(CudaError::ContextMismatch);
        }
        let _g = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().event_record(self.raw, stream.raw),
        )
    }
    pub fn query(&self) -> Result<bool, CudaError> {
        self.live()?;
        let _g = self.owner.current()?;
        let r = self.owner.dispatch().event_query(self.raw);
        if r == CUDA_SUCCESS {
            Ok(true)
        } else if r == CUDA_ERROR_NOT_READY {
            Ok(false)
        } else {
            check(self.owner.dispatch(), r).map(|_| false)
        }
    }
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.live()?;
        let _g = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().event_sync(self.raw),
        )
    }
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32, CudaError> {
        start.live()?;
        end.live()?;
        if !start.owner.same(&end.owner) {
            return Err(CudaError::ContextMismatch);
        }
        let _g = start.owner.current()?;
        let mut ms = 0.;
        let d = start.owner.dispatch();
        check(d, d.event_elapsed(&mut ms, start.raw, end.raw)?)?;
        Ok(ms)
    }
    pub fn close(&self) -> Result<(), CudaError> {
        self.live()?;
        self.closed.store(true, Ordering::Release);
        let _g = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().event_destroy(self.raw),
        )
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel)
            && let Ok(_g) = self.owner.current()
        {
            let _ = self.owner.dispatch().event_destroy(self.raw);
        }
    }
}

pub struct CudaModule {
    owner: Owner,
    raw: CuModule,
    closed: AtomicBool,
    metadata: ModuleLoadMetadata,
}
impl CudaModule {
    pub fn load_metadata(&self) -> &ModuleLoadMetadata {
        &self.metadata
    }
    pub(crate) fn device(&self) -> DeviceId {
        self.owner.device()
    }
    pub(crate) fn belongs_to_primary(&self, primary: &PrimaryContext) -> bool {
        matches!(&self.owner, Owner::Primary(owner) if Arc::ptr_eq(&owner.0, &primary.0))
    }
    fn live(&self) -> Result<(), CudaError> {
        if self.closed.load(Ordering::Acquire) {
            Err(CudaError::Closed("module"))
        } else {
            Ok(())
        }
    }
    pub fn function(&self, name: &CStr) -> Result<Function, CudaError> {
        self.live()?;
        let _g = self.owner.current()?;
        let mut raw = ptr::null_mut();
        let d = self.owner.dispatch();
        check(d, d.module_function(&mut raw, self.raw, name))?;
        Ok(Function {
            owner: self.owner.clone(),
            raw,
        })
    }
    /// Explicitly unloads the module.  It is idempotent in the sense that a
    /// second close reports `Closed` and never calls the driver twice.
    pub fn close(&self) -> Result<(), CudaError> {
        self.live()?;
        self.closed.store(true, Ordering::Release);
        let _g = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().module_unload(self.raw),
        )
    }
}
impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel)
            && let Ok(_g) = self.owner.current()
        {
            let _ = self.owner.dispatch().module_unload(self.raw);
        }
    }
}
pub struct Function {
    owner: Owner,
    raw: CuFunction,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
}
impl LaunchConfig {
    pub fn validate(self, max_threads: u32) -> Result<(), CudaError> {
        if self.grid.contains(&0) || self.block.contains(&0) {
            return Err(CudaError::InvalidArgument("zero grid or block dimension"));
        }
        let threads = self
            .block
            .into_iter()
            .try_fold(1_u32, |a, b| a.checked_mul(b))
            .ok_or(CudaError::Overflow)?;
        if threads > max_threads {
            return Err(CudaError::InvalidArgument(
                "block exceeds device thread limit",
            ));
        }
        Ok(())
    }
}
impl Function {
    /// `args` owns the pointed-to argument values through this synchronous call.
    pub fn launch(
        &self,
        config: LaunchConfig,
        stream: &Stream,
        args: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        if !self.owner.same(&stream.owner) {
            return Err(CudaError::ContextMismatch);
        }
        config.validate(self.owner.device_capability_threads()?)?;
        let _g = self.owner.current()?;
        check(
            self.owner.dispatch(),
            self.owner.dispatch().launch(
                self.raw,
                config.grid,
                config.block,
                config.shared_bytes,
                stream.raw,
                args.as_mut_ptr(),
            ),
        )
    }
}
impl Context {
    fn device_capability_threads(&self) -> Result<u32, CudaError> {
        let d = self.inner.driver.device(self.device())?;
        Ok(d.capability()?.max_threads_per_block)
    }
}

// Native dynamic loading. The function-pointer casts are confined to this one
// audited boundary: dlsym guarantees a symbol with the declared CUDA C ABI.
// Tests use `Dispatch`, never this conversion.
struct NativeDispatch {
    _library: Library,
    table: NativeTable,
    graph: NativeGraphTable,
    peer: NativePeerTable,
}
struct NativePeerTable {
    can_access: Option<unsafe extern "C" fn(*mut c_int, CuDevice, CuDevice) -> CuResult>,
    enable: Option<unsafe extern "C" fn(CuContext, c_uint) -> CuResult>,
    disable: Option<unsafe extern "C" fn(CuContext) -> CuResult>,
    copy_async: Option<
        unsafe extern "C" fn(
            CuDevicePtr,
            CuContext,
            CuDevicePtr,
            CuContext,
            usize,
            CuStream,
        ) -> CuResult,
    >,
}
struct NativeGraphTable {
    begin: Option<unsafe extern "C" fn(CuStream, c_uint) -> CuResult>,
    end: Option<unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult>,
    instantiate: Option<
        unsafe extern "C" fn(
            *mut CuGraphExec,
            CuGraph,
            *mut c_void,
            *mut c_char,
            usize,
        ) -> CuResult,
    >,
    launch: Option<unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult>,
    destroy: Option<unsafe extern "C" fn(CuGraph) -> CuResult>,
    exec_destroy: Option<unsafe extern "C" fn(CuGraphExec) -> CuResult>,
}
macro_rules! table { ($($n:ident : $t:ty),* $(,)?) => { struct NativeTable { $($n: $t,)* } }; }
table!(driver_version: unsafe extern "C" fn(*mut c_int)->CuResult, init: unsafe extern "C" fn(c_uint)->CuResult, device_count: unsafe extern "C" fn(*mut c_int)->CuResult, device_get: unsafe extern "C" fn(*mut CuDevice,c_int)->CuResult, device_name: unsafe extern "C" fn(*mut c_char,c_int,CuDevice)->CuResult, device_cc: unsafe extern "C" fn(*mut c_int,*mut c_int,CuDevice)->CuResult, device_memory: unsafe extern "C" fn(*mut usize,CuDevice)->CuResult, device_attribute: unsafe extern "C" fn(*mut c_int,c_int,CuDevice)->CuResult, ctx_create: unsafe extern "C" fn(*mut CuContext,c_uint,CuDevice)->CuResult, ctx_destroy: unsafe extern "C" fn(CuContext)->CuResult, ctx_get_current: unsafe extern "C" fn(*mut CuContext)->CuResult, ctx_set_current: unsafe extern "C" fn(CuContext)->CuResult, primary_ctx_retain: unsafe extern "C" fn(*mut CuContext,CuDevice)->CuResult, primary_ctx_release: unsafe extern "C" fn(CuDevice)->CuResult, primary_ctx_get_state: unsafe extern "C" fn(CuDevice,*mut c_uint,*mut c_int)->CuResult, primary_ctx_set_flags: unsafe extern "C" fn(CuDevice,c_uint)->CuResult, ctx_push_current: unsafe extern "C" fn(CuContext)->CuResult, ctx_pop_current: unsafe extern "C" fn(*mut CuContext)->CuResult, mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr,usize)->CuResult, mem_free: unsafe extern "C" fn(CuDevicePtr)->CuResult, memcpy_htod: unsafe extern "C" fn(CuDevicePtr,*const c_void,usize)->CuResult, memcpy_dtoh: unsafe extern "C" fn(*mut c_void,CuDevicePtr,usize)->CuResult, memcpy_dtod: unsafe extern "C" fn(CuDevicePtr,CuDevicePtr,usize)->CuResult, memcpy_htod_async: Option<unsafe extern "C" fn(CuDevicePtr,*const c_void,usize,CuStream)->CuResult>, memcpy_dtoh_async: Option<unsafe extern "C" fn(*mut c_void,CuDevicePtr,usize,CuStream)->CuResult>, memcpy_dtod_async: Option<unsafe extern "C" fn(CuDevicePtr,CuDevicePtr,usize,CuStream)->CuResult>, mem_host_alloc: Option<unsafe extern "C" fn(*mut *mut c_void,usize,c_uint)->CuResult>, mem_free_host: Option<unsafe extern "C" fn(*mut c_void)->CuResult>, stream_create: unsafe extern "C" fn(*mut CuStream,c_uint)->CuResult, stream_destroy: unsafe extern "C" fn(CuStream)->CuResult, stream_sync: unsafe extern "C" fn(CuStream)->CuResult, event_create: unsafe extern "C" fn(*mut CuEvent,c_uint)->CuResult, event_destroy: unsafe extern "C" fn(CuEvent)->CuResult, event_record: unsafe extern "C" fn(CuEvent,CuStream)->CuResult, event_query: unsafe extern "C" fn(CuEvent)->CuResult, event_sync: unsafe extern "C" fn(CuEvent)->CuResult, stream_wait_event: unsafe extern "C" fn(CuStream,CuEvent,c_uint)->CuResult, event_elapsed: Option<unsafe extern "C" fn(*mut f32,CuEvent,CuEvent)->CuResult>, module_load_data: unsafe extern "C" fn(*mut CuModule,*const c_void)->CuResult, module_load_data_ex: Option<unsafe extern "C" fn(*mut CuModule,*const c_void,c_uint,*mut u32,*mut *mut c_void)->CuResult>, module_unload: unsafe extern "C" fn(CuModule)->CuResult, module_function: unsafe extern "C" fn(*mut CuFunction,CuModule,*const c_char)->CuResult, launch: unsafe extern "C" fn(CuFunction,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,CuStream,*mut *mut c_void,*mut *mut c_void)->CuResult, error_name: unsafe extern "C" fn(CuResult,*mut *const c_char)->CuResult, error_string: unsafe extern "C" fn(CuResult,*mut *const c_char)->CuResult);
impl NativeDispatch {
    fn load() -> Result<Self, CudaError> {
        let library = Library::open()?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let p = library.symbol(concat!($name, "\0").as_bytes())?;
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }
        let table = NativeTable {
            driver_version: sym!(
                "cuDriverGetVersion",
                unsafe extern "C" fn(*mut c_int) -> CuResult
            ),
            init: sym!("cuInit", unsafe extern "C" fn(c_uint) -> CuResult),
            device_count: sym!(
                "cuDeviceGetCount",
                unsafe extern "C" fn(*mut c_int) -> CuResult
            ),
            device_get: sym!(
                "cuDeviceGet",
                unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult
            ),
            device_name: sym!(
                "cuDeviceGetName",
                unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult
            ),
            device_cc: sym!(
                "cuDeviceComputeCapability",
                unsafe extern "C" fn(*mut c_int, *mut c_int, CuDevice) -> CuResult
            ),
            device_memory: sym!(
                "cuDeviceTotalMem_v2",
                unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult
            ),
            device_attribute: sym!(
                "cuDeviceGetAttribute",
                unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult
            ),
            ctx_create: sym!(
                "cuCtxCreate_v2",
                unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult
            ),
            ctx_destroy: sym!(
                "cuCtxDestroy_v2",
                unsafe extern "C" fn(CuContext) -> CuResult
            ),
            ctx_get_current: sym!(
                "cuCtxGetCurrent",
                unsafe extern "C" fn(*mut CuContext) -> CuResult
            ),
            ctx_set_current: sym!(
                "cuCtxSetCurrent",
                unsafe extern "C" fn(CuContext) -> CuResult
            ),
            primary_ctx_retain: sym!(
                "cuDevicePrimaryCtxRetain",
                unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult
            ),
            primary_ctx_release: sym!(
                "cuDevicePrimaryCtxRelease",
                unsafe extern "C" fn(CuDevice) -> CuResult
            ),
            primary_ctx_get_state: sym!(
                "cuDevicePrimaryCtxGetState",
                unsafe extern "C" fn(CuDevice, *mut c_uint, *mut c_int) -> CuResult
            ),
            primary_ctx_set_flags: sym!(
                "cuDevicePrimaryCtxSetFlags",
                unsafe extern "C" fn(CuDevice, c_uint) -> CuResult
            ),
            ctx_push_current: sym!(
                "cuCtxPushCurrent_v2",
                unsafe extern "C" fn(CuContext) -> CuResult
            ),
            ctx_pop_current: sym!(
                "cuCtxPopCurrent_v2",
                unsafe extern "C" fn(*mut CuContext) -> CuResult
            ),
            mem_alloc: sym!(
                "cuMemAlloc_v2",
                unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult
            ),
            mem_free: sym!(
                "cuMemFree_v2",
                unsafe extern "C" fn(CuDevicePtr) -> CuResult
            ),
            memcpy_htod: sym!(
                "cuMemcpyHtoD_v2",
                unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult
            ),
            memcpy_dtoh: sym!(
                "cuMemcpyDtoH_v2",
                unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult
            ),
            memcpy_dtod: sym!(
                "cuMemcpyDtoD_v2",
                unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize) -> CuResult
            ),
            memcpy_htod_async: library
                .symbol(b"cuMemcpyHtoDAsync_v2\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(
                            CuDevicePtr,
                            *const c_void,
                            usize,
                            CuStream,
                        ) -> CuResult,
                    >(p)
                }),
            memcpy_dtoh_async: library
                .symbol(b"cuMemcpyDtoHAsync_v2\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult,
                    >(p)
                }),
            memcpy_dtod_async: library
                .symbol(b"cuMemcpyDtoDAsync_v2\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize, CuStream) -> CuResult,
                    >(p)
                }),
            mem_host_alloc: library.symbol(b"cuMemHostAlloc\0").ok().map(|p| unsafe {
                std::mem::transmute::<
                    *mut c_void,
                    unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CuResult,
                >(p)
            }),
            mem_free_host: library.symbol(b"cuMemFreeHost\0").ok().map(|p| unsafe {
                std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_void) -> CuResult>(p)
            }),
            stream_create: sym!(
                "cuStreamCreate",
                unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult
            ),
            stream_destroy: sym!(
                "cuStreamDestroy_v2",
                unsafe extern "C" fn(CuStream) -> CuResult
            ),
            stream_sync: sym!(
                "cuStreamSynchronize",
                unsafe extern "C" fn(CuStream) -> CuResult
            ),
            event_create: sym!(
                "cuEventCreate",
                unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult
            ),
            event_destroy: sym!(
                "cuEventDestroy_v2",
                unsafe extern "C" fn(CuEvent) -> CuResult
            ),
            event_record: sym!(
                "cuEventRecord",
                unsafe extern "C" fn(CuEvent, CuStream) -> CuResult
            ),
            event_query: sym!("cuEventQuery", unsafe extern "C" fn(CuEvent) -> CuResult),
            event_sync: sym!(
                "cuEventSynchronize",
                unsafe extern "C" fn(CuEvent) -> CuResult
            ),
            stream_wait_event: sym!(
                "cuStreamWaitEvent",
                unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> CuResult
            ),
            // `cuEventElapsedTime(float*, CUevent, CUevent)` is deliberately
            // optional: timing must not make ordinary events unavailable.
            event_elapsed: library
                .symbol(b"cuEventElapsedTime\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
                    >(p)
                }),
            module_load_data: sym!(
                "cuModuleLoadData",
                unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult
            ),
            module_load_data_ex: library
                .symbol(b"cuModuleLoadDataEx\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(
                            *mut CuModule,
                            *const c_void,
                            c_uint,
                            *mut u32,
                            *mut *mut c_void,
                        ) -> CuResult,
                    >(p)
                }),
            module_unload: sym!("cuModuleUnload", unsafe extern "C" fn(CuModule) -> CuResult),
            module_function: sym!(
                "cuModuleGetFunction",
                unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult
            ),
            launch: sym!(
                "cuLaunchKernel",
                unsafe extern "C" fn(
                    CuFunction,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    CuStream,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> CuResult
            ),
            error_name: sym!(
                "cuGetErrorName",
                unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult
            ),
            error_string: sym!(
                "cuGetErrorString",
                unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult
            ),
        };
        let peer = NativePeerTable {
            can_access: library
                .symbol(b"cuDeviceCanAccessPeer\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            enable: library
                .symbol(b"cuCtxEnablePeerAccess\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            disable: library
                .symbol(b"cuCtxDisablePeerAccess\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            copy_async: library
                .symbol(b"cuMemcpyPeerAsync\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
        };
        // CUDA's legacy cuGraphInstantiate ABI is stable and permits a null
        // error-node/log buffer for this static foundation.
        let graph = NativeGraphTable {
            begin: library
                .symbol(b"cuStreamBeginCapture\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            end: library
                .symbol(b"cuStreamEndCapture\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            instantiate: library
                .symbol(b"cuGraphInstantiate\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            launch: library
                .symbol(b"cuGraphLaunch\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            destroy: library
                .symbol(b"cuGraphDestroy\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
            exec_destroy: library
                .symbol(b"cuGraphExecDestroy\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) }),
        };
        Ok(Self {
            _library: library,
            table,
            graph,
            peer,
        })
    }
}
macro_rules! call { ($self:ident.$method:ident($($arg:expr),*)) => { unsafe { ($self.table.$method)($($arg),*) } }; }
impl Dispatch for NativeDispatch {
    fn driver_version(&self, o: &mut c_int) -> CuResult {
        call!(self.driver_version(o))
    }
    fn init(&self, x: c_uint) -> CuResult {
        call!(self.init(x))
    }
    fn device_count(&self, o: &mut c_int) -> CuResult {
        call!(self.device_count(o))
    }
    fn device_get(&self, o: &mut CuDevice, x: c_int) -> CuResult {
        call!(self.device_get(o, x))
    }
    fn device_name(&self, o: &mut [c_char], x: CuDevice) -> CuResult {
        call!(self.device_name(o.as_mut_ptr(), o.len() as c_int, x))
    }
    fn device_cc(&self, a: &mut c_int, b: &mut c_int, x: CuDevice) -> CuResult {
        call!(self.device_cc(a, b, x))
    }
    fn device_memory(&self, o: &mut usize, x: CuDevice) -> CuResult {
        call!(self.device_memory(o, x))
    }
    fn device_attribute(&self, o: &mut c_int, a: c_int, x: CuDevice) -> CuResult {
        call!(self.device_attribute(o, a, x))
    }
    fn ctx_create(&self, o: &mut CuContext, f: c_uint, d: CuDevice) -> CuResult {
        call!(self.ctx_create(o, f, d))
    }
    fn ctx_destroy(&self, x: CuContext) -> CuResult {
        call!(self.ctx_destroy(x))
    }
    fn ctx_get_current(&self, o: &mut CuContext) -> CuResult {
        call!(self.ctx_get_current(o))
    }
    fn ctx_set_current(&self, x: CuContext) -> CuResult {
        call!(self.ctx_set_current(x))
    }
    fn primary_ctx_retain(&self, o: &mut CuContext, d: CuDevice) -> CuResult {
        call!(self.primary_ctx_retain(o, d))
    }
    fn primary_ctx_release(&self, d: CuDevice) -> CuResult {
        call!(self.primary_ctx_release(d))
    }
    fn primary_ctx_get_state(&self, d: CuDevice, f: &mut c_uint, a: &mut c_int) -> CuResult {
        call!(self.primary_ctx_get_state(d, f, a))
    }
    fn primary_ctx_set_flags(&self, d: CuDevice, f: c_uint) -> CuResult {
        call!(self.primary_ctx_set_flags(d, f))
    }
    fn ctx_push_current(&self, c: CuContext) -> CuResult {
        call!(self.ctx_push_current(c))
    }
    fn ctx_pop_current(&self, o: &mut CuContext) -> CuResult {
        call!(self.ctx_pop_current(o))
    }
    fn mem_alloc(&self, o: &mut CuDevicePtr, x: usize) -> CuResult {
        call!(self.mem_alloc(o, x))
    }
    fn mem_free(&self, x: CuDevicePtr) -> CuResult {
        call!(self.mem_free(x))
    }
    fn memcpy_htod(&self, a: CuDevicePtr, b: *const c_void, c: usize) -> CuResult {
        call!(self.memcpy_htod(a, b, c))
    }
    fn memcpy_dtoh(&self, a: *mut c_void, b: CuDevicePtr, c: usize) -> CuResult {
        call!(self.memcpy_dtoh(a, b, c))
    }
    fn memcpy_dtod(&self, a: CuDevicePtr, b: CuDevicePtr, c: usize) -> CuResult {
        call!(self.memcpy_dtod(a, b, c))
    }
    fn device_can_access_peer(
        &self,
        out: &mut c_int,
        source: CuDevice,
        destination: CuDevice,
    ) -> CuResult {
        self.peer
            .can_access
            .map_or(801, |f| unsafe { f(out, source, destination) })
    }
    fn ctx_enable_peer_access(&self, peer: CuContext, flags: c_uint) -> CuResult {
        self.peer.enable.map_or(801, |f| unsafe { f(peer, flags) })
    }
    fn ctx_disable_peer_access(&self, peer: CuContext) -> CuResult {
        self.peer.disable.map_or(801, |f| unsafe { f(peer) })
    }
    fn memcpy_peer_async(
        &self,
        dst: CuDevicePtr,
        dst_ctx: CuContext,
        src: CuDevicePtr,
        src_ctx: CuContext,
        bytes: usize,
        stream: CuStream,
    ) -> CuResult {
        self.peer.copy_async.map_or(801, |f| unsafe {
            f(dst, dst_ctx, src, src_ctx, bytes, stream)
        })
    }
    fn supports_async_transfers(&self) -> bool {
        self.table.memcpy_htod_async.is_some()
            && self.table.memcpy_dtoh_async.is_some()
            && self.table.memcpy_dtod_async.is_some()
    }
    fn supports_pinned_host_memory(&self) -> bool {
        self.table.mem_host_alloc.is_some() && self.table.mem_free_host.is_some()
    }
    fn memcpy_htod_async(
        &self,
        a: CuDevicePtr,
        b: *const c_void,
        c: usize,
        s: CuStream,
    ) -> CuResult {
        self.table
            .memcpy_htod_async
            .map_or(801, |f| unsafe { f(a, b, c, s) })
    }
    fn memcpy_dtoh_async(&self, a: *mut c_void, b: CuDevicePtr, c: usize, s: CuStream) -> CuResult {
        self.table
            .memcpy_dtoh_async
            .map_or(801, |f| unsafe { f(a, b, c, s) })
    }
    fn memcpy_dtod_async(&self, a: CuDevicePtr, b: CuDevicePtr, c: usize, s: CuStream) -> CuResult {
        self.table
            .memcpy_dtod_async
            .map_or(801, |f| unsafe { f(a, b, c, s) })
    }
    fn mem_host_alloc(&self, o: &mut *mut c_void, n: usize, f: c_uint) -> CuResult {
        self.table
            .mem_host_alloc
            .map_or(801, |fun| unsafe { fun(o, n, f) })
    }
    fn mem_free_host(&self, p: *mut c_void) -> CuResult {
        self.table.mem_free_host.map_or(801, |f| unsafe { f(p) })
    }
    fn supports_graphs(&self) -> bool {
        self.graph.begin.is_some()
            && self.graph.end.is_some()
            && self.graph.instantiate.is_some()
            && self.graph.launch.is_some()
            && self.graph.destroy.is_some()
            && self.graph.exec_destroy.is_some()
    }
    fn stream_begin_capture(&self, s: CuStream, mode: c_uint) -> CuResult {
        self.graph.begin.map_or(801, |f| unsafe { f(s, mode) })
    }
    fn stream_end_capture(&self, s: CuStream, g: &mut CuGraph) -> CuResult {
        self.graph.end.map_or(801, |f| unsafe { f(s, g) })
    }
    fn graph_instantiate(&self, e: &mut CuGraphExec, g: CuGraph) -> CuResult {
        self.graph.instantiate.map_or(801, |f| unsafe {
            f(e, g, ptr::null_mut(), ptr::null_mut(), 0)
        })
    }
    fn graph_launch(&self, e: CuGraphExec, s: CuStream) -> CuResult {
        self.graph.launch.map_or(801, |f| unsafe { f(e, s) })
    }
    fn graph_destroy(&self, g: CuGraph) -> CuResult {
        self.graph.destroy.map_or(801, |f| unsafe { f(g) })
    }
    fn graph_exec_destroy(&self, e: CuGraphExec) -> CuResult {
        self.graph.exec_destroy.map_or(801, |f| unsafe { f(e) })
    }
    fn stream_create(&self, o: &mut CuStream, x: c_uint) -> CuResult {
        call!(self.stream_create(o, x))
    }
    fn stream_destroy(&self, x: CuStream) -> CuResult {
        call!(self.stream_destroy(x))
    }
    fn stream_sync(&self, x: CuStream) -> CuResult {
        call!(self.stream_sync(x))
    }
    fn event_create(&self, o: &mut CuEvent, x: c_uint) -> CuResult {
        call!(self.event_create(o, x))
    }
    fn event_destroy(&self, x: CuEvent) -> CuResult {
        call!(self.event_destroy(x))
    }
    fn event_record(&self, a: CuEvent, b: CuStream) -> CuResult {
        call!(self.event_record(a, b))
    }
    fn event_query(&self, x: CuEvent) -> CuResult {
        call!(self.event_query(x))
    }
    fn event_sync(&self, x: CuEvent) -> CuResult {
        call!(self.event_sync(x))
    }
    fn stream_wait_event(&self, a: CuStream, b: CuEvent, c: c_uint) -> CuResult {
        call!(self.stream_wait_event(a, b, c))
    }
    fn event_elapsed(&self, o: &mut f32, a: CuEvent, b: CuEvent) -> Result<CuResult, CudaError> {
        self.table
            .event_elapsed
            .map(|f| unsafe { f(o, a, b) })
            .ok_or(CudaError::MissingSymbol("cuEventElapsedTime"))
    }
    fn module_load_data(&self, o: &mut CuModule, p: *const c_void) -> CuResult {
        call!(self.module_load_data(o, p))
    }
    fn module_load_data_ex(
        &self,
        o: &mut CuModule,
        image: *const c_void,
        options: &[u32],
        values: &mut [*mut c_void],
    ) -> CuResult {
        match self.table.module_load_data_ex {
            Some(f) => unsafe {
                f(
                    o,
                    image,
                    options.len() as c_uint,
                    options.as_ptr().cast_mut(),
                    values.as_mut_ptr(),
                )
            },
            None => self.module_load_data(o, image),
        }
    }
    fn supports_module_load_data_ex(&self) -> bool {
        self.table.module_load_data_ex.is_some()
    }
    fn module_unload(&self, x: CuModule) -> CuResult {
        call!(self.module_unload(x))
    }
    fn module_function(&self, o: &mut CuFunction, m: CuModule, n: &CStr) -> CuResult {
        call!(self.module_function(o, m, n.as_ptr()))
    }
    fn launch(
        &self,
        f: CuFunction,
        g: [u32; 3],
        b: [u32; 3],
        s: u32,
        st: CuStream,
        a: *mut *mut c_void,
    ) -> CuResult {
        call!(self.launch(
            f,
            g[0],
            g[1],
            g[2],
            b[0],
            b[1],
            b[2],
            s,
            st,
            a,
            ptr::null_mut()
        ))
    }
    fn error_name(&self, c: CuResult) -> Option<String> {
        let mut p = ptr::null();
        if call!(self.error_name(c, &mut p)) == 0 && !p.is_null() {
            Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        } else {
            None
        }
    }
    fn error_string(&self, c: CuResult) -> Option<String> {
        let mut p = ptr::null();
        if call!(self.error_string(c, &mut p)) == 0 && !p.is_null() {
            Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        } else {
            None
        }
    }
}

struct Library(*mut c_void);
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
impl Library {
    fn open() -> Result<Self, CudaError> {
        #[cfg(target_os = "macos")]
        let names: &[&str] = &["libcuda.dylib"];
        #[cfg(target_os = "linux")]
        let names: &[&str] = &["libcuda.so.1", "libcuda.so"];
        #[cfg(target_os = "windows")]
        let names: &[&str] = &["nvcuda.dll"];
        let mut details = Vec::new();
        for &name in names {
            let c = CString::new(name).unwrap();
            let h = unsafe { platform::open(c.as_ptr()) };
            if !h.is_null() {
                return Ok(Self(h));
            }
            details.push(platform::last_error());
        }
        Err(CudaError::LibraryNotFound {
            tried: names
                .iter()
                .map(|x| match *x {
                    "libcuda.dylib" => "libcuda.dylib",
                    "libcuda.so.1" => "libcuda.so.1",
                    "libcuda.so" => "libcuda.so",
                    _ => "nvcuda.dll",
                })
                .collect(),
            detail: details.join("; "),
        })
    }
    fn symbol(&self, n: &'static [u8]) -> Result<*mut c_void, CudaError> {
        let p = unsafe { platform::symbol(self.0, n.as_ptr().cast()) };
        if p.is_null() {
            Err(CudaError::MissingSymbol(
                std::str::from_utf8(&n[..n.len() - 1]).unwrap_or("<invalid>"),
            ))
        } else {
            Ok(p)
        }
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe { platform::close(self.0) }
    }
}
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod platform {
    use super::*;
    unsafe extern "C" {
        fn dlopen(n: *const c_char, f: c_int) -> *mut c_void;
        fn dlsym(h: *mut c_void, n: *const c_char) -> *mut c_void;
        fn dlclose(h: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }
    const RTLD_NOW: c_int = 2;
    pub unsafe fn open(n: *const c_char) -> *mut c_void {
        unsafe { dlopen(n, RTLD_NOW) }
    }
    pub unsafe fn symbol(h: *mut c_void, n: *const c_char) -> *mut c_void {
        unsafe { dlsym(h, n) }
    }
    pub unsafe fn close(h: *mut c_void) {
        unsafe { dlclose(h) };
    }
    pub fn last_error() -> String {
        unsafe {
            let p = dlerror();
            if p.is_null() {
                "unknown dynamic loader error".into()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}
#[cfg(target_os = "windows")]
mod platform {
    use super::*;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(n: *const c_char) -> *mut c_void;
        fn GetProcAddress(h: *mut c_void, n: *const c_char) -> *mut c_void;
        fn FreeLibrary(h: *mut c_void) -> i32;
    }
    pub unsafe fn open(n: *const c_char) -> *mut c_void {
        unsafe { LoadLibraryA(n) }
    }
    pub unsafe fn symbol(h: *mut c_void, n: *const c_char) -> *mut c_void {
        unsafe { GetProcAddress(h, n) }
    }
    pub unsafe fn close(h: *mut c_void) {
        unsafe { FreeLibrary(h) };
    }
    pub fn last_error() -> String {
        "LoadLibraryA failed".into()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{
        Mutex,
        atomic::{AtomicI32, AtomicU32},
    };

    pub(crate) struct Mock {
        calls: Mutex<Vec<&'static str>>,
        current: Mutex<usize>,
        fail_alloc: AtomicBool,
        module_result: AtomicI32,
        ex: AtomicBool,
        ex_result: AtomicI32,
        capture_active: AtomicBool,
        elapsed_supported: AtomicBool,
        elapsed_result: AtomicI32,
        elapsed_millis: AtomicU32,
        event_ready: AtomicBool,
        peer_capable: AtomicBool,
        peer_result: AtomicI32,
    }
    impl Default for Mock {
        fn default() -> Self {
            Self {
                calls: Mutex::new(vec![]),
                current: Mutex::new(0),
                fail_alloc: AtomicBool::new(false),
                module_result: AtomicI32::new(0),
                ex: AtomicBool::new(false),
                ex_result: AtomicI32::new(0),
                capture_active: AtomicBool::new(false),
                elapsed_supported: AtomicBool::new(false),
                elapsed_result: AtomicI32::new(0),
                elapsed_millis: AtomicU32::new(1.5_f32.to_bits()),
                event_ready: AtomicBool::new(false),
                peer_capable: AtomicBool::new(true),
                peer_result: AtomicI32::new(0),
            }
        }
    }
    impl Mock {
        fn call(&self, name: &'static str) {
            self.calls.lock().unwrap().push(name);
        }
        pub(crate) fn calls(&self) -> Vec<&'static str> {
            self.calls.lock().unwrap().clone()
        }
        pub(crate) fn set_module_result(&self, result: i32) {
            self.module_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_elapsed_support(&self, supported: bool) {
            self.elapsed_supported.store(supported, Ordering::Release);
        }
        pub(crate) fn set_elapsed_result(&self, result: CuResult) {
            self.elapsed_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_elapsed_millis(&self, milliseconds: f32) {
            self.elapsed_millis
                .store(milliseconds.to_bits(), Ordering::Release);
        }
        pub(crate) fn set_event_ready(&self, ready: bool) {
            self.event_ready.store(ready, Ordering::Release);
        }
        pub(crate) fn set_peer_capable(&self, capable: bool) {
            self.peer_capable.store(capable, Ordering::Release);
        }
    }
    impl Dispatch for Mock {
        fn driver_version(&self, out: &mut c_int) -> CuResult {
            self.call("version");
            *out = 12000;
            0
        }
        fn init(&self, _: c_uint) -> CuResult {
            self.call("init");
            0
        }
        fn device_count(&self, out: &mut c_int) -> CuResult {
            self.call("count");
            *out = 1;
            0
        }
        fn device_get(&self, out: &mut CuDevice, _: c_int) -> CuResult {
            self.call("get");
            *out = 7;
            0
        }
        fn device_name(&self, out: &mut [c_char], _: CuDevice) -> CuResult {
            self.call("name");
            out[..5].copy_from_slice(&[b'm' as i8, b'o' as i8, b'c' as i8, b'k' as i8, 0]);
            0
        }
        fn device_cc(&self, a: &mut c_int, b: &mut c_int, _: CuDevice) -> CuResult {
            *a = 8;
            *b = 0;
            0
        }
        fn device_memory(&self, out: &mut usize, _: CuDevice) -> CuResult {
            *out = 4096;
            0
        }
        fn device_attribute(&self, out: &mut c_int, _: c_int, _: CuDevice) -> CuResult {
            *out = 1024;
            0
        }
        fn ctx_create(&self, out: &mut CuContext, _: c_uint, _: CuDevice) -> CuResult {
            self.call("ctx_create");
            *out = 0x11usize as CuContext;
            0
        }
        fn ctx_destroy(&self, _: CuContext) -> CuResult {
            self.call("ctx_destroy");
            0
        }
        fn ctx_get_current(&self, out: &mut CuContext) -> CuResult {
            self.call("ctx_get");
            *out = *self.current.lock().unwrap() as CuContext;
            0
        }
        fn ctx_set_current(&self, context: CuContext) -> CuResult {
            self.call("ctx_set");
            *self.current.lock().unwrap() = context as usize;
            0
        }
        fn primary_ctx_retain(&self, out: &mut CuContext, _: CuDevice) -> CuResult {
            self.call("primary_retain");
            *out = 0x77usize as CuContext;
            0
        }
        fn primary_ctx_release(&self, _: CuDevice) -> CuResult {
            self.call("primary_release");
            0
        }
        fn ctx_push_current(&self, _: CuContext) -> CuResult {
            self.call("ctx_push");
            0
        }
        fn ctx_pop_current(&self, out: &mut CuContext) -> CuResult {
            self.call("ctx_pop");
            *out = 0x77usize as CuContext;
            0
        }
        fn mem_alloc(&self, out: &mut CuDevicePtr, _: usize) -> CuResult {
            self.call("alloc");
            if self.fail_alloc.load(Ordering::Acquire) {
                2
            } else {
                *out = 0x1000;
                0
            }
        }
        fn mem_free(&self, _: CuDevicePtr) -> CuResult {
            self.call("free");
            0
        }
        fn memcpy_htod(&self, _: CuDevicePtr, _: *const c_void, _: usize) -> CuResult {
            self.call("htod");
            0
        }
        fn memcpy_dtoh(&self, _: *mut c_void, _: CuDevicePtr, _: usize) -> CuResult {
            self.call("dtoh");
            0
        }
        fn memcpy_dtod(&self, _: CuDevicePtr, _: CuDevicePtr, _: usize) -> CuResult {
            self.call("dtod");
            0
        }
        fn device_can_access_peer(&self, out: &mut c_int, _: CuDevice, _: CuDevice) -> CuResult {
            self.call("peer_can");
            *out = self.peer_capable.load(Ordering::Acquire) as c_int;
            self.peer_result.load(Ordering::Acquire)
        }
        fn ctx_enable_peer_access(&self, _: CuContext, _: c_uint) -> CuResult {
            self.call("peer_enable");
            self.peer_result.load(Ordering::Acquire)
        }
        fn ctx_disable_peer_access(&self, _: CuContext) -> CuResult {
            self.call("peer_disable");
            self.peer_result.load(Ordering::Acquire)
        }
        fn memcpy_peer_async(
            &self,
            _: CuDevicePtr,
            _: CuContext,
            _: CuDevicePtr,
            _: CuContext,
            _: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("peer_copy");
            self.peer_result.load(Ordering::Acquire)
        }
        fn supports_async_transfers(&self) -> bool {
            true
        }
        fn supports_pinned_host_memory(&self) -> bool {
            true
        }
        fn memcpy_htod_async(
            &self,
            _: CuDevicePtr,
            _: *const c_void,
            _: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("htod_async");
            0
        }
        fn memcpy_dtoh_async(
            &self,
            _: *mut c_void,
            _: CuDevicePtr,
            _: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("dtoh_async");
            0
        }
        fn memcpy_dtod_async(
            &self,
            _: CuDevicePtr,
            _: CuDevicePtr,
            _: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("dtod_async");
            0
        }
        fn mem_host_alloc(&self, out: &mut *mut c_void, _: usize, _: c_uint) -> CuResult {
            self.call("host_alloc");
            *out = 0x88usize as *mut c_void;
            0
        }
        fn mem_free_host(&self, _: *mut c_void) -> CuResult {
            self.call("host_free");
            0
        }
        fn supports_graphs(&self) -> bool {
            true
        }
        fn stream_begin_capture(&self, _: CuStream, _: c_uint) -> CuResult {
            self.call("capture_begin");
            if self.capture_active.swap(true, Ordering::AcqRel) {
                2
            } else {
                0
            }
        }
        fn stream_end_capture(&self, _: CuStream, out: &mut CuGraph) -> CuResult {
            self.call("capture_end");
            self.capture_active.store(false, Ordering::Release);
            *out = 0x99usize as CuGraph;
            0
        }
        fn graph_instantiate(&self, out: &mut CuGraphExec, _: CuGraph) -> CuResult {
            self.call("graph_instantiate");
            *out = 0xaausize as CuGraphExec;
            0
        }
        fn graph_launch(&self, _: CuGraphExec, _: CuStream) -> CuResult {
            self.call("graph_launch");
            0
        }
        fn graph_destroy(&self, _: CuGraph) -> CuResult {
            self.call("graph_destroy");
            0
        }
        fn graph_exec_destroy(&self, _: CuGraphExec) -> CuResult {
            self.call("graph_exec_destroy");
            0
        }
        fn stream_create(&self, out: &mut CuStream, _: c_uint) -> CuResult {
            self.call("stream_create");
            *out = 0x22usize as CuStream;
            0
        }
        fn stream_destroy(&self, _: CuStream) -> CuResult {
            self.call("stream_destroy");
            0
        }
        fn stream_sync(&self, _: CuStream) -> CuResult {
            self.call("stream_sync");
            0
        }
        fn event_create(&self, out: &mut CuEvent, _: c_uint) -> CuResult {
            self.call("event_create");
            *out = 0x33usize as CuEvent;
            0
        }
        fn event_destroy(&self, _: CuEvent) -> CuResult {
            self.call("event_destroy");
            0
        }
        fn event_record(&self, _: CuEvent, _: CuStream) -> CuResult {
            self.call("event_record");
            0
        }
        fn event_query(&self, _: CuEvent) -> CuResult {
            if self.event_ready.load(Ordering::Acquire) {
                CUDA_SUCCESS
            } else {
                CUDA_ERROR_NOT_READY
            }
        }
        fn event_sync(&self, _: CuEvent) -> CuResult {
            self.call("event_sync");
            0
        }
        fn stream_wait_event(&self, _: CuStream, _: CuEvent, _: c_uint) -> CuResult {
            self.call("stream_wait");
            0
        }
        fn event_elapsed(
            &self,
            out: &mut f32,
            _: CuEvent,
            _: CuEvent,
        ) -> Result<CuResult, CudaError> {
            if !self.elapsed_supported.load(Ordering::Acquire) {
                return Err(CudaError::MissingSymbol("cuEventElapsedTime"));
            }
            self.call("event_elapsed");
            *out = f32::from_bits(self.elapsed_millis.load(Ordering::Acquire));
            Ok(self.elapsed_result.load(Ordering::Acquire))
        }
        fn module_load_data(&self, out: &mut CuModule, _: *const c_void) -> CuResult {
            self.call("module_load");
            *out = 0x44usize as CuModule;
            self.module_result.load(Ordering::Acquire)
        }
        fn supports_module_load_data_ex(&self) -> bool {
            self.ex.load(Ordering::Acquire)
        }
        fn module_load_data_ex(
            &self,
            out: &mut CuModule,
            _: *const c_void,
            options: &[u32],
            values: &mut [*mut c_void],
        ) -> CuResult {
            self.call("module_load_ex");
            assert_eq!(
                options,
                [
                    CU_JIT_OPTIMIZATION_LEVEL,
                    CU_JIT_TARGET_FROM_CUCONTEXT,
                    CU_JIT_INFO_LOG_BUFFER,
                    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                    CU_JIT_ERROR_LOG_BUFFER,
                    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
                ]
            );
            unsafe {
                assert_eq!(*(values[0] as *const u32), 4);
                let info = values[2] as *mut u8;
                let info_len = values[3] as *mut usize;
                let err = values[4] as *mut u8;
                let err_len = values[5] as *mut usize;
                assert_ne!(info, err);
                assert_eq!(*info_len, 4096);
                assert_eq!(*err_len, 4096);
                std::ptr::copy_nonoverlapping(b"info\0tail".as_ptr(), info, 9);
                std::ptr::copy_nonoverlapping(b"error-full".as_ptr(), err, 10);
                *info_len = 9;
                *err_len = 10;
            }
            *out = 0x44usize as CuModule;
            self.ex_result.load(Ordering::Acquire)
        }
        fn module_unload(&self, _: CuModule) -> CuResult {
            self.call("module_unload");
            0
        }
        fn module_function(&self, out: &mut CuFunction, _: CuModule, _: &CStr) -> CuResult {
            self.call("function");
            *out = 0x55usize as CuFunction;
            0
        }
        fn launch(
            &self,
            _: CuFunction,
            _: [u32; 3],
            _: [u32; 3],
            _: u32,
            _: CuStream,
            _: *mut *mut c_void,
        ) -> CuResult {
            self.call("launch");
            0
        }
        fn error_name(&self, code: CuResult) -> Option<String> {
            Some(
                if code == 2 {
                    "CUDA_ERROR_OUT_OF_MEMORY"
                } else {
                    "CUDA_ERROR_UNKNOWN"
                }
                .into(),
            )
        }
        fn error_string(&self, _: CuResult) -> Option<String> {
            Some("mock failure".into())
        }
    }
    pub(crate) fn context(mock: &Arc<Mock>) -> Context {
        Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .create_context()
            .unwrap()
    }

    #[test]
    fn mock_context_restores_previous_and_resources_cleanup() {
        let mock = Arc::new(Mock::default());
        *mock.current.lock().unwrap() = 0x99;
        let ctx = context(&mock);
        {
            let _guard = ctx.enter().unwrap();
            assert_eq!(*mock.current.lock().unwrap(), 0x11);
        }
        assert_eq!(*mock.current.lock().unwrap(), 0x99);
        let buffer = ctx.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        buffer.copy_from(8, &[1; 8]).unwrap();
        assert!(matches!(
            buffer.copy_from(9, &[1; 8]),
            Err(CudaError::InvalidArgument(_))
        ));
        let stream = ctx.stream().unwrap();
        let event = ctx.event().unwrap();
        event.record(&stream).unwrap();
        assert!(!event.query().unwrap());
        drop(event);
        drop(stream);
        drop(buffer);
        drop(ctx);
        let calls = mock.calls();
        assert!(calls.windows(2).any(|pair| pair == ["ctx_set", "ctx_set"]));
        assert!(
            calls.contains(&"free")
                && calls.contains(&"stream_destroy")
                && calls.contains(&"event_destroy")
                && calls.contains(&"ctx_destroy")
        );
    }

    #[test]
    fn mock_rejects_zero_alloc_and_use_after_close() {
        let mock = Arc::new(Mock::default());
        let ctx = context(&mock);
        assert!(NonZeroUsize::new(0).is_none());
        let buffer = ctx.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        buffer.close().unwrap();
        assert!(matches!(
            buffer.copy_from(0, &[1]),
            Err(CudaError::Closed("buffer"))
        ));
        assert!(matches!(buffer.close(), Err(CudaError::Closed("buffer"))));
    }

    #[test]
    fn event_timing_is_optional_without_changing_event_lifecycle() {
        let mock = Arc::new(Mock::default());
        let ctx = context(&mock);
        let stream = ctx.stream().unwrap();
        let start = ctx.event().unwrap();
        let end = ctx.event().unwrap();

        start.record(&stream).unwrap();
        assert!(!start.query().unwrap());
        stream.wait(&start).unwrap();
        end.record(&stream).unwrap();
        end.synchronize().unwrap();
        assert!(matches!(
            Event::elapsed_ms(&start, &end),
            Err(CudaError::MissingSymbol("cuEventElapsedTime"))
        ));
        start.close().unwrap();
        end.close().unwrap();
        stream.close().unwrap();

        let calls = mock.calls();
        for required in [
            "event_create",
            "event_record",
            "stream_wait",
            "event_sync",
            "event_destroy",
        ] {
            assert!(calls.contains(&required), "missing {required}");
        }
        assert!(!calls.contains(&"event_elapsed"));
    }

    #[test]
    fn mock_event_timing_has_typed_success_and_driver_failure() {
        let mock = Arc::new(Mock::default());
        mock.set_elapsed_support(true);
        let ctx = context(&mock);
        let start = ctx.event().unwrap();
        let end = ctx.event().unwrap();

        assert_eq!(Event::elapsed_ms(&start, &end).unwrap(), 1.5);
        mock.set_elapsed_result(2);
        assert!(matches!(
            Event::elapsed_ms(&start, &end),
            Err(CudaError::Driver { code: 2, .. })
        ));
    }

    #[test]
    fn error_mapping_and_launch_validation_are_precise() {
        let mock = Arc::new(Mock::default());
        mock.fail_alloc.store(true, Ordering::Release);
        let ctx = context(&mock);
        let err = match ctx.allocate(NonZeroUsize::new(4).unwrap()) {
            Ok(_) => panic!("mock allocation should fail"),
            Err(err) => err,
        };
        assert!(
            matches!(err, CudaError::Driver { code: 2, ref name, .. } if name == "CUDA_ERROR_OUT_OF_MEMORY")
        );
        assert!(matches!(
            LaunchConfig {
                grid: [1, 1, 1],
                block: [1025, 1, 1],
                shared_bytes: 0
            }
            .validate(1024),
            Err(CudaError::InvalidArgument(_))
        ));
    }

    #[test]
    fn module_load_ex_option_layout_logs_and_failure_are_bounded() {
        let mock = Arc::new(Mock::default());
        mock.ex.store(true, Ordering::Release);
        let ctx = context(&mock);
        let ptx = CString::new(".version 7.0").unwrap();
        let module = ctx
            .module_from_ptx_with_options(
                &ptx,
                ModuleLoadOptions {
                    optimization_level: 4,
                    log_bytes: 4096,
                    capture_logs: true,
                },
            )
            .unwrap();
        assert_eq!(module.load_metadata().info_log, "info");
        assert_eq!(module.load_metadata().error_log, "error-full");
        assert!(module.load_metadata().used_load_data_ex);
        drop(module);
        assert!(mock.calls().contains(&"module_load_ex"));
        mock.ex_result.store(200, Ordering::Release);
        let error = match ctx.module_from_ptx_with_options(
            &ptx,
            ModuleLoadOptions {
                optimization_level: 4,
                log_bytes: 4096,
                capture_logs: true,
            },
        ) {
            Ok(_) => panic!("expected JIT failure"),
            Err(e) => e,
        };
        assert!(
            matches!(error,CudaError::JitCompile{code:200,ref info_log,ref error_log,..} if info_log=="info" && error_log=="error-full")
        );
    }
    #[test]
    fn primary_context_clones_retain_once_and_pop_before_final_release() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let clone = primary.clone();
        {
            let _guard = clone.enter().unwrap();
        }
        drop(clone);
        drop(primary);
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|&&x| x == "primary_retain").count(), 1);
        assert_eq!(calls.iter().filter(|&&x| x == "primary_release").count(), 1);
        assert!(calls.windows(2).any(|x| x == ["ctx_push", "ctx_pop"]));
    }
    #[test]
    fn allocator_reuses_exact_owner_scoped_leases() {
        let mock = Arc::new(Mock::default());
        let ctx = context(&mock);
        let allocator = ctx.allocator();
        let lease = allocator.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let first = lease.view().unwrap().device_ptr().unwrap();
        lease.release();
        let second = allocator.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        assert_eq!(second.view().unwrap().device_ptr().unwrap(), first);
        assert_eq!(allocator.in_use_bytes(), 8);
        drop(second);
        assert_eq!(allocator.cached_bytes(), 256);
        allocator.trim().unwrap();
        assert_eq!(allocator.cached_bytes(), 0);
    }

    #[test]
    fn pooled_views_enforce_logical_length_and_primary_pool_is_shareable() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PrimaryCudaAllocator>();
        assert_send_sync::<PrimaryBlock>();
        assert_send_sync::<PrimaryBufferLease>();
        assert_send_sync::<PrimaryEventFence>();
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let allocator = primary.allocator();
        let lease = allocator.allocate(NonZeroUsize::new(1).unwrap()).unwrap();
        assert_eq!(lease.view().unwrap().len(), 1);
        assert!(matches!(
            lease.view().unwrap().copy_from(1, &[7]),
            Err(CudaError::InvalidArgument(_))
        ));
        lease.release();
        assert_eq!(allocator.cached_bytes(), 256);
        allocator.trim().unwrap();
        assert_eq!(allocator.cached_bytes(), 0);
        assert_eq!(allocator.in_use_bytes(), 0);
        assert_eq!(allocator.reserved_bytes(), 0);
    }

    #[test]
    fn primary_pool_reuse_advances_block_generation() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let allocator = primary.allocator();
        let lease = allocator.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let block = lease.block.as_ref().unwrap().clone();
        let generation = lease.generation;
        lease.release();
        let next = allocator.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        assert!(next.generation > generation);
        assert_ne!(block.generation.load(Ordering::Acquire), generation);
    }

    #[test]
    fn primary_deferred_blocks_are_not_reused_until_collected() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let allocator = primary.allocator();
        let stream = primary.stream().unwrap();
        let lease = allocator.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let fence = Arc::new(primary.event_fence().unwrap());
        fence.record(&stream).unwrap();
        lease.attach_fence(fence).unwrap();
        lease.release();
        assert_eq!(allocator.deferred_blocks(), 1);
        assert_eq!(allocator.cached_bytes(), 0);
        assert_eq!(allocator.collect_deferred().unwrap(), 0);
        let other = allocator.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        assert_eq!(allocator.reserved_bytes(), 512);
        drop(other);
        mock.set_event_ready(true);
        assert_eq!(allocator.collect_deferred().unwrap(), 1);
        assert_eq!(allocator.deferred_blocks(), 0);
        assert_eq!(allocator.cached_bytes(), 512);
        assert_eq!(allocator.wait_deferred().unwrap(), 0);
    }

    #[test]
    fn peer_access_and_pooled_copy_are_directional_and_deferred() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let device = driver.device(DeviceId(0)).unwrap();
        let source = device.retain_primary_context().unwrap();
        let destination = device.retain_primary_context().unwrap();
        let peer = source.peer_access_to(&destination).unwrap();
        let src_pool = source.allocator();
        let dst_pool = destination.allocator();
        let src = src_pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let dst = dst_pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let stream = destination.stream().unwrap();
        let mut transfer = dst
            .copy_from_peer_async(0, &peer, &src, 0, 8, &stream)
            .unwrap();
        assert!(!transfer.query().unwrap());
        transfer.wait().unwrap();
        drop(transfer);
        drop(src);
        drop(dst);
        assert_eq!(src_pool.deferred_blocks(), 1);
        assert_eq!(dst_pool.deferred_blocks(), 1);
        mock.set_event_ready(true);
        assert_eq!(src_pool.collect_deferred().unwrap(), 1);
        assert_eq!(dst_pool.collect_deferred().unwrap(), 1);
        assert!(destination.peer_access_to(&source).is_ok());
        assert!(matches!(
            source.peer_access_to(&source),
            Err(CudaError::InvalidArgument(_))
        ));
        mock.set_peer_capable(false);
        assert!(matches!(
            source.peer_access_to(&destination),
            Err(CudaError::InvalidArgument(_))
        ));
        drop(peer);
        let calls = mock.calls();
        assert!(calls.contains(&"peer_copy") && calls.contains(&"peer_disable"));
    }

    #[test]
    fn primary_fence_is_owner_scoped_and_cleans_before_primary_release() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let device = driver.device(DeviceId(0)).unwrap();
        let first = device.retain_primary_context().unwrap();
        let second = device.retain_primary_context().unwrap();
        let fence = first.event_fence().unwrap();
        assert!(matches!(
            fence.validate_owner(&second),
            Err(CudaError::ContextMismatch)
        ));
        assert!(!fence.query().unwrap());
        drop(fence);
        drop(first);
        drop(second);
        let calls = mock.calls();
        let destroy = calls.iter().position(|x| *x == "event_destroy").unwrap();
        let release = calls.iter().rposition(|x| *x == "primary_release").unwrap();
        assert!(destroy < release);
    }

    #[test]
    fn primary_resources_cleanup_before_last_release_and_reject_owned_owner() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let device = driver.device(DeviceId(0)).unwrap();
        let primary = device.retain_primary_context().unwrap();
        let owned = device.create_context().unwrap();
        let buffer = primary.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let stream = primary.stream().unwrap();
        let event = primary.event().unwrap();
        let module = primary
            .module_from_ptx(&CString::new(".version 7.0").unwrap())
            .unwrap();
        let other = owned.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        assert!(matches!(
            buffer.copy_from_device(0, &other, 0, 1),
            Err(CudaError::WrongDevice { .. })
        ));
        drop(module);
        drop(event);
        drop(stream);
        drop(buffer);
        drop(primary);
        let calls = mock.calls();
        let release = calls.iter().position(|x| *x == "primary_release").unwrap();
        for cleanup in ["free", "stream_destroy", "event_destroy", "module_unload"] {
            assert!(calls.iter().position(|x| *x == cleanup).unwrap() < release);
        }
    }

    #[test]
    fn async_pinned_transfers_hold_borrows_until_wait() {
        let mock = Arc::new(Mock::default());
        let ctx = context(&mock);
        let buffer = ctx.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let other = ctx.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let host = ctx.allocate_pinned(NonZeroUsize::new(8).unwrap()).unwrap();
        let stream = ctx.stream().unwrap();
        let mut h2d = buffer
            .copy_from_pinned_async(0, &host, 0, 8, &stream)
            .unwrap();
        assert!(!h2d.query().unwrap());
        h2d.wait().unwrap();
        let mut d2h = buffer
            .copy_to_pinned_async(0, &host, 0, 8, &stream)
            .unwrap();
        d2h.wait().unwrap();
        let mut d2d = buffer
            .copy_from_device_async(0, &other, 0, 8, &stream)
            .unwrap();
        d2d.wait().unwrap();
        assert!(matches!(
            buffer.copy_from_pinned_async(0, &host, 0, 0, &stream),
            Err(CudaError::InvalidArgument(_))
        ));
        assert!(matches!(
            buffer.copy_from_pinned_async(7, &host, 0, 2, &stream),
            Err(CudaError::InvalidArgument(_))
        ));
        drop(d2d);
        drop(d2h);
        drop(h2d);
        drop(host);
        drop(stream);
        drop(other);
        drop(buffer);
        let calls = mock.calls();
        assert!(
            calls.contains(&"htod_async")
                && calls.contains(&"dtoh_async")
                && calls.contains(&"dtod_async")
        );
        assert!(calls.contains(&"host_alloc") && calls.contains(&"host_free"));
    }

    #[test]
    fn primary_graph_capture_replays_and_releases_before_context() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let stream = primary.stream().unwrap();
        let buffer = primary.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let host = primary
            .allocate_pinned(NonZeroUsize::new(8).unwrap())
            .unwrap();
        let mut capture = stream.begin_capture().unwrap();
        capture.retain_buffer(&buffer).unwrap();
        capture.retain_pinned(&host).unwrap();
        let graph = capture.finish().unwrap();
        let exec = graph.instantiate().unwrap();
        exec.launch(&stream).unwrap();
        exec.launch(&stream).unwrap();
        exec.close().unwrap();
        drop(exec);
        drop(host);
        drop(buffer);
        drop(stream);
        drop(primary);
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|&&x| x == "graph_launch").count(), 2);
        let destroy = calls
            .iter()
            .position(|x| *x == "graph_exec_destroy")
            .unwrap();
        let release = calls.iter().position(|x| *x == "primary_release").unwrap();
        assert!(destroy < release);
    }

    #[test]
    fn graph_capture_rejects_nested_and_cross_owner_and_abandons_cleanly() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let a = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let b = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let stream = a.stream().unwrap();
        let foreign = b.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        let mut capture = stream.begin_capture().unwrap();
        assert!(matches!(
            stream.begin_capture(),
            Err(CudaError::Driver { code: 2, .. })
        ));
        assert!(matches!(
            capture.retain_buffer(&foreign),
            Err(CudaError::ContextMismatch)
        ));
        drop(capture);
        let calls = mock.calls();
        assert!(
            calls
                .windows(2)
                .any(|pair| pair == ["capture_end", "graph_destroy"])
        );
    }
}
