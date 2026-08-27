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
    panic::{AssertUnwindSafe, catch_unwind},
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
type CuLinkState = *mut c_void;
type CuJitInputType = c_uint;
type CuFunction = *mut c_void;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;
const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NOT_READY: CuResult = 600;
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: CuResult = 704;
const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: CuResult = 705;
const CU_CTX_SCHED_AUTO: c_uint = 0;
const CU_EVENT_DEFAULT: c_uint = 0;
const CU_STREAM_DEFAULT: c_uint = 0;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
const CU_JIT_INPUT_PTX: CuJitInputType = 1;
const CU_JIT_INPUT_LIBRARY: CuJitInputType = 4;
// CUDA Driver API `CUjitInputType`: CU_JIT_INPUT_NVVM = 5.
// <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html>
const CU_JIT_INPUT_NVVM: CuJitInputType = 5;
const LINKED_MODULE_IDENTITY_VERSION: u32 = 1;

/// The closed set of in-memory CUDA link inputs accepted by this runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkInputKind {
    Ptx,
    Library,
    Nvvm,
}

/// One owned, ordered input for the CUDA Driver linker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkInput {
    kind: LinkInputKind,
    name: CString,
    bytes: Vec<u8>,
}
impl LinkInput {
    pub fn ptx(name: &str, bytes: Vec<u8>) -> Result<Self, CudaError> {
        Self::new(LinkInputKind::Ptx, name, bytes)
    }
    /// Adds caller-supplied immutable CUDA library bytes to an ordered link.
    pub fn library(name: &str, bytes: Vec<u8>) -> Result<Self, CudaError> {
        Self::new(LinkInputKind::Library, name, bytes)
    }
    /// Adds caller-supplied immutable NVVM bitcode; no host discovery occurs.
    pub fn nvvm(name: &str, bytes: Vec<u8>) -> Result<Self, CudaError> {
        Self::new(LinkInputKind::Nvvm, name, bytes)
    }
    fn new(kind: LinkInputKind, name: &str, bytes: Vec<u8>) -> Result<Self, CudaError> {
        let name = CString::new(name).map_err(|_| CudaError::InvalidArgument("link input name"))?;
        let input = Self {
            kind,
            name,
            bytes,
        };
        input.validate()?;
        Ok(input)
    }
    fn validate(&self) -> Result<(), CudaError> {
        if self.name.as_bytes().is_empty() || self.bytes.is_empty() {
            return Err(CudaError::InvalidArgument("nonempty link input"));
        }
        match self.kind {
            LinkInputKind::Ptx | LinkInputKind::Library | LinkInputKind::Nvvm => Ok(()),
        }
    }
    fn input_type(&self) -> CuJitInputType {
        match self.kind {
            LinkInputKind::Ptx => CU_JIT_INPUT_PTX,
            LinkInputKind::Library => CU_JIT_INPUT_LIBRARY,
            LinkInputKind::Nvvm => CU_JIT_INPUT_NVVM,
        }
    }
}

/// Versioned deterministic identity for an ordered linked-module input set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedModuleIdentity {
    version: u32,
    cache_key: String,
}
impl LinkedModuleIdentity {
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }
    pub fn from_cache_key(cache_key: &str) -> Result<Self, CudaError> {
        let Some((prefix, fingerprint)) = cache_key.rsplit_once(':') else {
            return Err(CudaError::InvalidArgument("linked module cache key"));
        };
        let Some(version) = prefix.strip_prefix("cuda-link-v") else {
            return Err(CudaError::InvalidArgument("linked module cache key"));
        };
        if version.parse::<u32>().ok() != Some(LINKED_MODULE_IDENTITY_VERSION)
            || fingerprint.len() != 16
            || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(CudaError::InvalidArgument("linked module cache key"));
        }
        Ok(Self {
            version: LINKED_MODULE_IDENTITY_VERSION,
            cache_key: cache_key.into(),
        })
    }
}

/// Validates and fingerprints ordered link inputs without loading a module.
pub fn linked_module_identity(inputs: &[LinkInput]) -> Result<LinkedModuleIdentity, CudaError> {
    validate_link_inputs(inputs)?;
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in LINKED_MODULE_IDENTITY_VERSION
        .to_le_bytes()
        .into_iter()
        .chain((inputs.len() as u64).to_le_bytes())
    {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    for input in inputs {
        let kind = match input.kind {
            LinkInputKind::Ptx => 1_u8,
            LinkInputKind::Library => 4_u8,
            LinkInputKind::Nvvm => 5_u8,
        };
        for byte in [kind]
            .into_iter()
            .chain((input.name.as_bytes().len() as u64).to_le_bytes())
            .chain(input.name.as_bytes().iter().copied())
            .chain((input.bytes.len() as u64).to_le_bytes())
            .chain(input.bytes.iter().copied())
        {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    Ok(LinkedModuleIdentity {
        version: LINKED_MODULE_IDENTITY_VERSION,
        cache_key: format!("cuda-link-v{LINKED_MODULE_IDENTITY_VERSION}:{hash:016x}"),
    })
}

fn validate_link_inputs(inputs: &[LinkInput]) -> Result<(), CudaError> {
    if inputs.is_empty() {
        return Err(CudaError::InvalidArgument("nonempty link inputs"));
    }
    for (index, input) in inputs.iter().enumerate() {
        input.validate()?;
        if inputs[..index]
            .iter()
            .any(|previous| previous.name == input.name)
        {
            return Err(CudaError::InvalidArgument("unique link input names"));
        }
    }
    Ok(())
}

/// A CUDA ordinal, distinct from arbitrary signed integers at the public API.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DeviceId(pub u32);

/// A stable Rust owner for one retained CUDA primary context.
///
/// This is diagnostic metadata for alternate `Dispatch` implementations. It
/// deliberately does not identify a CUDA context: distinct owners may have
/// the same raw CUDA context and device handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PrimaryOwner {
    pub identity: usize,
    pub device: DeviceId,
}

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
    /// Records a successfully retained Rust primary-context owner.
    ///
    /// This metadata hook cannot affect CUDA ownership or currentness.
    fn primary_owner_register(&self, _owner: PrimaryOwner) {}
    /// Records final release of a Rust primary-context owner.
    ///
    /// This metadata hook cannot affect CUDA ownership or currentness.
    fn primary_owner_unregister(&self, _owner: PrimaryOwner) {}
    /// Records that `owner` became current on the calling thread after CUDA
    /// successfully pushed its raw context.
    fn primary_owner_enter(&self, _owner: PrimaryOwner) {}
    /// Records that `owner` stopped being current on the calling thread after
    /// CUDA successfully popped its raw context.
    fn primary_owner_exit(&self, _owner: PrimaryOwner) {}
    /// Returns the metadata owner currently observed on the calling thread.
    ///
    /// Native dispatches return `None`; this is never used to validate CUDA
    /// currentness.
    fn primary_owner_current(&self) -> Option<PrimaryOwner> {
        None
    }
    /// Records the stable owners authorized for one peer-copy submission.
    ///
    /// This diagnostic metadata disambiguates colliding raw contexts in
    /// deterministic mocks; it cannot authorize or alter CUDA peer access.
    fn primary_owner_peer_copy(&self, _source: PrimaryOwner, _destination: PrimaryOwner) {}
    /// Registers a test-only semantic local-add launch contract for a loaded
    /// function. Native dispatch ignores it; it never changes the CUDA ABI.
    fn primary_owner_register_collective_add(
        &self,
        _owner: PrimaryOwner,
        _function: usize,
        _source_key: &str,
        _dtype: crate::DType,
        _abi_version: u32,
    ) {
    }
    /// Test-only generic PTX semantic registration. Native dispatch deliberately ignores it.
    #[allow(private_interfaces)]
    fn primary_owner_register_generic_kernel(
        &self,
        _owner: PrimaryOwner,
        _function: usize,
        _key: &str,
        _semantics: std::sync::Arc<crate::ptx::GenericKernelSemantics>,
    ) {
    }
    fn primary_owner_unregister_generic_kernel(&self, _owner: PrimaryOwner, _function: usize) {}
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
    /// Peer copies have an independent optional Driver symbol; ordinary async
    /// memcpy support must not be used as a substitute capability check.
    fn supports_peer_async_transfers(&self) -> bool {
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
    fn link_create(
        &self,
        _options: &[u32],
        _values: &mut [*mut c_void],
        _state: &mut CuLinkState,
    ) -> Result<CuResult, CudaError> {
        Err(CudaError::MissingSymbol("cuLinkCreate"))
    }
    fn link_add_data(
        &self,
        _state: CuLinkState,
        _input: CuJitInputType,
        _data: *const c_void,
        _bytes: usize,
        _name: &CStr,
        _options: &[u32],
        _values: &mut [*mut c_void],
    ) -> Result<CuResult, CudaError> {
        Err(CudaError::MissingSymbol("cuLinkAddData"))
    }
    fn link_complete(
        &self,
        _state: CuLinkState,
        _image: &mut *mut c_void,
        _bytes: &mut usize,
    ) -> Result<CuResult, CudaError> {
        Err(CudaError::MissingSymbol("cuLinkComplete"))
    }
    fn link_destroy(&self, _state: CuLinkState) -> Result<CuResult, CudaError> {
        Err(CudaError::MissingSymbol("cuLinkDestroy"))
    }
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

// Observation is strictly diagnostic. An alternate Dispatch must not be able
// to turn metadata bookkeeping into a CUDA operation failure or a double-panic
// during RAII cleanup.
fn observe_primary(f: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(f));
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
        let primary = PrimaryContext(Arc::new(PrimaryInner {
            driver: self.driver.clone(),
            device: self.id,
            raw,
            raw_device: self.raw,
        }));
        observe_primary(|| d.primary_owner_register(primary.owner()));
        Ok(primary)
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
        // A primary owner outlives all resources and enter guards that retain
        // it. Release CUDA first; then remove only the diagnostic record.
        // Both are best effort during unwinding, so Drop never reports errors.
        let dispatch = self.driver.0.dispatch.as_ref();
        let _ = dispatch.primary_ctx_release(self.raw_device);
        observe_primary(|| {
            dispatch.primary_owner_unregister(PrimaryOwner {
                identity: self as *const Self as usize,
                device: self.device,
            })
        });
    }
}
/// Shareable retained primary context. Currentness is guarded per thread.
#[derive(Clone)]
pub struct PrimaryContext(Arc<PrimaryInner>);
impl PrimaryContext {
    pub(crate) fn register_generic_kernel_semantics(
        &self,
        function: usize,
        key: &str,
        semantics: std::sync::Arc<crate::ptx::GenericKernelSemantics>,
    ) {
        observe_primary(|| {
            self.0
                .driver
                .0
                .dispatch
                .primary_owner_register_generic_kernel(self.owner(), function, key, semantics)
        });
    }
    pub(crate) fn unregister_generic_kernel_semantics(&self, function: usize) {
        observe_primary(|| {
            self.0
                .driver
                .0
                .dispatch
                .primary_owner_unregister_generic_kernel(self.owner(), function)
        });
    }
    pub(crate) fn owner(&self) -> PrimaryOwner {
        PrimaryOwner {
            identity: self.identity(),
            device: self.device(),
        }
    }
    #[allow(dead_code)]
    pub(crate) fn register_collective_add_semantics(
        &self,
        function: usize,
        source_key: &str,
        dtype: crate::DType,
        abi_version: u32,
    ) {
        let dispatch = self.0.driver.0.dispatch.as_ref();
        observe_primary(|| {
            dispatch.primary_owner_register_collective_add(
                self.owner(),
                function,
                source_key,
                dtype,
                abi_version,
            )
        });
    }
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
        let enabled = d.ctx_enable_peer_access(destination.0.raw, 0);
        if enabled != CUDA_SUCCESS && enabled != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            check(d, enabled)?;
        }
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
    /// Makes this primary context current on the calling thread.
    ///
    /// Metadata observation is updated only after a successful CUDA push. On
    /// drop, a successful CUDA pop removes the matching observation; if the
    /// pop fails, the observation is deliberately left unchanged because CUDA
    /// currentness is then unknown. Drop otherwise ignores Driver and
    /// observation-hook failures; hook panics are contained as well.
    pub fn enter(&self) -> Result<PrimaryContextGuard, CudaError> {
        let d = self.0.driver.0.dispatch.as_ref();
        check(d, d.ctx_push_current(self.0.raw))?;
        observe_primary(|| d.primary_owner_enter(self.owner()));
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
    #[allow(dead_code)]
    pub(crate) fn ptx_sm(&self) -> Result<u32, CudaError> {
        Ok(self.0.driver.device(self.device())?.capability()?.sm())
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
    /// Links ordered in-memory PTX inputs without changing the legacy PTX cache path.
    pub fn module_from_link_inputs(&self, inputs: &[LinkInput]) -> Result<CudaModule, CudaError> {
        Owner::Primary(self.clone()).module_from_link_inputs(inputs)
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
        let d = self.source.0.driver.0.dispatch.as_ref();
        let disabled = d.ctx_disable_peer_access(self.destination.0.raw);
        if disabled == CUDA_SUCCESS || disabled == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED {
            Ok(())
        } else {
            check(d, disabled)
        }
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
            let dispatch = self.primary.0.driver.0.dispatch.as_ref();
            if dispatch.ctx_pop_current(&mut popped) == CUDA_SUCCESS {
                debug_assert_eq!(popped, self.primary.0.raw);
                observe_primary(|| dispatch.primary_owner_exit(self.primary.owner()));
            }
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
    /// Links ordered in-memory PTX inputs without changing the legacy PTX cache path.
    pub fn module_from_link_inputs(&self, inputs: &[LinkInput]) -> Result<CudaModule, CudaError> {
        Owner::Owned(self.clone()).module_from_link_inputs(inputs)
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
            if raw.is_null() {
                return Err(CudaError::InvalidArgument(
                    "module load returned null handle",
                ));
            }
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
        if raw.is_null() {
            return Err(CudaError::InvalidArgument(
                "module load returned null handle",
            ));
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

    fn module_from_link_inputs(&self, inputs: &[LinkInput]) -> Result<CudaModule, CudaError> {
        validate_link_inputs(inputs)?;
        let _guard = self.current()?;
        let dispatch = self.dispatch();
        let mut link = LinkState::create(dispatch)?;
        for input in inputs {
            link.add(input)?;
        }
        let image = link.complete()?;
        let mut raw = ptr::null_mut();
        if let Err(error) = check(dispatch, dispatch.module_load_data(&mut raw, image.cast())) {
            return Err(error);
        }
        if let Err(error) = link.destroy() {
            let _ = dispatch.module_unload(raw);
            return Err(error);
        }
        Ok(CudaModule {
            owner: self.clone(),
            raw,
            closed: AtomicBool::new(false),
            metadata: ModuleLoadMetadata {
                used_load_data_ex: false,
                info_log: String::new(),
                error_log: String::new(),
            },
        })
    }
}

struct LinkState<'a> {
    dispatch: &'a dyn Dispatch,
    raw: CuLinkState,
}

impl<'a> LinkState<'a> {
    fn create(dispatch: &'a dyn Dispatch) -> Result<Self, CudaError> {
        let mut raw = ptr::null_mut();
        check(dispatch, dispatch.link_create(&[], &mut [], &mut raw)?)?;
        Ok(Self { dispatch, raw })
    }
    fn add(&self, input: &LinkInput) -> Result<(), CudaError> {
        check(
            self.dispatch,
            self.dispatch.link_add_data(
                self.raw,
                input.input_type(),
                input.bytes.as_ptr().cast(),
                input.bytes.len(),
                input.name.as_c_str(),
                &[],
                &mut [],
            )?,
        )
    }
    fn complete(&self) -> Result<*mut c_void, CudaError> {
        let mut image = ptr::null_mut();
        let mut bytes = 0;
        check(
            self.dispatch,
            self.dispatch
                .link_complete(self.raw, &mut image, &mut bytes)?,
        )?;
        if image.is_null() || bytes == 0 {
            return Err(CudaError::InvalidArgument("nonempty linked image"));
        }
        Ok(image)
    }
    fn destroy(&mut self) -> Result<(), CudaError> {
        let raw = std::mem::replace(&mut self.raw, ptr::null_mut());
        check(self.dispatch, self.dispatch.link_destroy(raw)?)
    }
}

impl Drop for LinkState<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let raw = std::mem::replace(&mut self.raw, ptr::null_mut());
            let _ = self.dispatch.link_destroy(raw);
        }
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
        let _guard = self.descriptor.owner.current()?;
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
        let _guard = self.descriptor.owner.current()?;
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
        let _guard = self.descriptor.owner.current()?;
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
/// Read-only, owner-scoped accounting for one exact primary allocator handle.
/// `pool_id` distinguishes independently constructed pools for the same primary
/// context; clone the allocator handle when callers need shared accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrimaryPoolStats {
    pub pool_id: usize,
    pub owner_id: usize,
    pub device: DeviceId,
    pub logical_leased_bytes: usize,
    pub cached_blocks: usize,
    pub cached_bytes: usize,
    pub deferred_blocks: usize,
    pub deferred_bytes: usize,
    pub quarantined_blocks: usize,
    pub quarantined_bytes: usize,
    pub peak_in_use_bytes: usize,
    pub peak_in_use_blocks: usize,
}
struct PrimaryPoolState {
    cached: std::collections::BTreeMap<usize, Vec<Arc<PrimaryBlock>>>,
    cached_bytes: usize,
    deferred: Vec<DeferredPrimaryBlock>,
    deferred_bytes: usize,
    quarantined: Vec<Arc<PrimaryBlock>>,
    in_use: usize,
    in_use_blocks: usize,
    reserved: usize,
    peak: usize,
    peak_blocks: usize,
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
                in_use_blocks: 0,
                reserved: 0,
                peak: 0,
                peak_blocks: 0,
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
    /// Snapshots this allocator's own shared state without invoking the Driver.
    pub fn stats(&self) -> PrimaryPoolStats {
        let state = self.state.lock().expect("primary allocator mutex poisoned");
        PrimaryPoolStats {
            pool_id: self as *const Self as usize,
            owner_id: self.primary.identity(),
            device: self.primary.device(),
            logical_leased_bytes: state.in_use,
            cached_blocks: state.cached.values().map(Vec::len).sum(),
            cached_bytes: state.cached_bytes,
            deferred_blocks: state.deferred.len(),
            deferred_bytes: state.deferred_bytes,
            quarantined_blocks: state.quarantined.len(),
            quarantined_bytes: state.quarantined.iter().map(|b| b.capacity).sum(),
            peak_in_use_bytes: state.peak,
            peak_in_use_blocks: state.peak_blocks,
        }
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
        state.in_use_blocks = state
            .in_use_blocks
            .checked_add(1)
            .ok_or(CudaError::Overflow)?;
        state.peak_blocks = state.peak_blocks.max(state.in_use_blocks);
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
    /// Sealed runtime-preflight metadata. No pointer or mutable allocation state escapes.
    #[allow(dead_code)] // consumed by the forthcoming sharded CUDA executor.
    pub(crate) fn execution_metadata(&self) -> Result<(usize, usize, u64, usize), CudaError> {
        let block = self.block.as_ref().ok_or(CudaError::StaleLease)?;
        if block.generation.load(Ordering::Acquire) != self.generation
            || block.closed.load(Ordering::Acquire)
        {
            return Err(CudaError::StaleLease);
        }
        Ok((
            block.primary.identity(),
            self.bytes,
            self.generation,
            Arc::as_ptr(block) as usize,
        ))
    }
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
        if !d.supports_peer_async_transfers() {
            return Err(CudaError::MissingSymbol("cuMemcpyPeerAsync"));
        }
        let fence = Arc::new(dst.primary.event_fence()?);
        let _guard = dst.primary.enter()?;
        observe_primary(|| d.primary_owner_peer_copy(source.primary.owner(), dst.primary.owner()));
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
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn copy_from_peer_async_profiled<'a>(
        &'a self,
        session: &ProfilingSession,
        name: impl Into<String>,
        dst_offset: usize,
        peer: &'a PeerAccess,
        src: &'a PrimaryBufferLease,
        src_offset: usize,
        bytes: usize,
        stream: &'a Stream,
    ) -> Result<ProfiledPeerTransfer<'a>, CudaError> {
        if !session.is_enabled() {
            return self
                .copy_from_peer_async(dst_offset, peer, src, src_offset, bytes, stream)
                .map(ProfiledPeerTransfer::Plain);
        }
        let dst = self.block.as_ref().ok_or(CudaError::StaleLease)?;
        let source = src.block.as_ref().ok_or(CudaError::StaleLease)?;
        if bytes == 0
            || !peer.matches(&source.primary, &dst.primary)
            || !stream.belongs_to_primary(&dst.primary)
        {
            return Err(CudaError::ContextMismatch);
        }
        let mut timing = TimedSample::begin(
            session,
            Metadata {
                kind: OperationKind::PeerCopy,
                name: name.into(),
                owner: dst.primary.identity(),
                device: dst.primary.device(),
                stream: stream.identity(),
                bytes: Some(bytes),
                geometry: None,
                source_key: None,
                peer: Some(crate::cuda_profile::PeerMetadata {
                    source_owner: source.primary.identity(),
                    source_device: source.primary.device(),
                    destination_owner: dst.primary.identity(),
                    destination_device: dst.primary.device(),
                }),
            },
            &dst.primary,
            stream,
            Arc::new(()),
        )
        .map_err(profile_cuda_error)?
        .ok_or(CudaError::InvalidArgument("enabled profiling session"))?;
        match self.copy_from_peer_async(dst_offset, peer, src, src_offset, bytes, stream) {
            Ok(transfer) => {
                timing.record_end(stream).map_err(profile_cuda_error)?;
                Ok(ProfiledPeerTransfer::Timed { transfer, timing })
            }
            Err(error) => {
                timing.fail_due_to(TimingError::Cuda(error.clone()));
                Err(error)
            }
        }
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
        state.in_use_blocks -= 1;
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

#[allow(dead_code)]
pub(crate) enum ProfiledPeerTransfer<'a> {
    Plain(PeerTransfer<'a>),
    Timed {
        transfer: PeerTransfer<'a>,
        timing: TimedSample<'a>,
    },
}
#[allow(dead_code)]
impl ProfiledPeerTransfer<'_> {
    pub(crate) fn wait(&mut self) -> Result<Option<u64>, CudaError> {
        match self {
            Self::Plain(transfer) => {
                transfer.wait()?;
                Ok(None)
            }
            Self::Timed { transfer, timing } => {
                let value = timing.wait().map_err(profile_cuda_error)?;
                transfer.complete = true;
                Ok(Some(value))
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
                let value = timing.collect().map_err(profile_cuda_error)?;
                transfer.complete = true;
                Ok(Some(value))
            }
        }
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
        let _guard = self.owner.current()?;
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
        let _guard = self.owner.current()?;
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
        let _guard = self.owner.current()?;
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
                peer: None,
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
        // CUDA success without a graph handle is not a usable capture. Reject
        // it before constructing an RAII owner that could later pass a null
        // handle back into the Driver.
        if raw.is_null() {
            return Err(CudaError::InvalidArgument("capture returned null graph"));
        }
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
        if raw.is_null() {
            return Err(CudaError::InvalidArgument(
                "instantiate returned null graph exec",
            ));
        }
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
    #[allow(dead_code)]
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
        if raw.is_null() {
            return Err(CudaError::InvalidArgument(
                "function lookup returned null handle",
            ));
        }
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
    pub(crate) fn identity(&self) -> usize {
        self.raw as usize
    }
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
impl NativePeerTable {
    /// Isolated optional resolver seam: peer omissions never affect the
    /// mandatory Driver table construction.
    fn resolve(mut symbol: impl FnMut(&'static [u8]) -> Option<*mut c_void>) -> Self {
        Self {
            can_access: symbol(b"cuDeviceCanAccessPeer\0")
                .map(|p| unsafe { std::mem::transmute(p) }),
            enable: symbol(b"cuCtxEnablePeerAccess\0").map(|p| unsafe { std::mem::transmute(p) }),
            disable: symbol(b"cuCtxDisablePeerAccess\0").map(|p| unsafe { std::mem::transmute(p) }),
            copy_async: symbol(b"cuMemcpyPeerAsync\0").map(|p| unsafe { std::mem::transmute(p) }),
        }
    }
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
table!(driver_version: unsafe extern "C" fn(*mut c_int)->CuResult, init: unsafe extern "C" fn(c_uint)->CuResult, device_count: unsafe extern "C" fn(*mut c_int)->CuResult, device_get: unsafe extern "C" fn(*mut CuDevice,c_int)->CuResult, device_name: unsafe extern "C" fn(*mut c_char,c_int,CuDevice)->CuResult, device_cc: unsafe extern "C" fn(*mut c_int,*mut c_int,CuDevice)->CuResult, device_memory: unsafe extern "C" fn(*mut usize,CuDevice)->CuResult, device_attribute: unsafe extern "C" fn(*mut c_int,c_int,CuDevice)->CuResult, ctx_create: unsafe extern "C" fn(*mut CuContext,c_uint,CuDevice)->CuResult, ctx_destroy: unsafe extern "C" fn(CuContext)->CuResult, ctx_get_current: unsafe extern "C" fn(*mut CuContext)->CuResult, ctx_set_current: unsafe extern "C" fn(CuContext)->CuResult, primary_ctx_retain: unsafe extern "C" fn(*mut CuContext,CuDevice)->CuResult, primary_ctx_release: unsafe extern "C" fn(CuDevice)->CuResult, primary_ctx_get_state: unsafe extern "C" fn(CuDevice,*mut c_uint,*mut c_int)->CuResult, primary_ctx_set_flags: unsafe extern "C" fn(CuDevice,c_uint)->CuResult, ctx_push_current: unsafe extern "C" fn(CuContext)->CuResult, ctx_pop_current: unsafe extern "C" fn(*mut CuContext)->CuResult, mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr,usize)->CuResult, mem_free: unsafe extern "C" fn(CuDevicePtr)->CuResult, memcpy_htod: unsafe extern "C" fn(CuDevicePtr,*const c_void,usize)->CuResult, memcpy_dtoh: unsafe extern "C" fn(*mut c_void,CuDevicePtr,usize)->CuResult, memcpy_dtod: unsafe extern "C" fn(CuDevicePtr,CuDevicePtr,usize)->CuResult, memcpy_htod_async: Option<unsafe extern "C" fn(CuDevicePtr,*const c_void,usize,CuStream)->CuResult>, memcpy_dtoh_async: Option<unsafe extern "C" fn(*mut c_void,CuDevicePtr,usize,CuStream)->CuResult>, memcpy_dtod_async: Option<unsafe extern "C" fn(CuDevicePtr,CuDevicePtr,usize,CuStream)->CuResult>, mem_host_alloc: Option<unsafe extern "C" fn(*mut *mut c_void,usize,c_uint)->CuResult>, mem_free_host: Option<unsafe extern "C" fn(*mut c_void)->CuResult>, stream_create: unsafe extern "C" fn(*mut CuStream,c_uint)->CuResult, stream_destroy: unsafe extern "C" fn(CuStream)->CuResult, stream_sync: unsafe extern "C" fn(CuStream)->CuResult, event_create: unsafe extern "C" fn(*mut CuEvent,c_uint)->CuResult, event_destroy: unsafe extern "C" fn(CuEvent)->CuResult, event_record: unsafe extern "C" fn(CuEvent,CuStream)->CuResult, event_query: unsafe extern "C" fn(CuEvent)->CuResult, event_sync: unsafe extern "C" fn(CuEvent)->CuResult, stream_wait_event: unsafe extern "C" fn(CuStream,CuEvent,c_uint)->CuResult, event_elapsed: Option<unsafe extern "C" fn(*mut f32,CuEvent,CuEvent)->CuResult>, module_load_data: unsafe extern "C" fn(*mut CuModule,*const c_void)->CuResult, module_load_data_ex: Option<unsafe extern "C" fn(*mut CuModule,*const c_void,c_uint,*mut u32,*mut *mut c_void)->CuResult>, link_create: Option<unsafe extern "C" fn(c_uint,*mut u32,*mut *mut c_void,*mut CuLinkState)->CuResult>, link_add_data: Option<unsafe extern "C" fn(CuLinkState,CuJitInputType,*mut c_void,usize,*const c_char,c_uint,*mut u32,*mut *mut c_void)->CuResult>, link_complete: Option<unsafe extern "C" fn(CuLinkState,*mut *mut c_void,*mut usize)->CuResult>, link_destroy: Option<unsafe extern "C" fn(CuLinkState)->CuResult>, module_unload: unsafe extern "C" fn(CuModule)->CuResult, module_function: unsafe extern "C" fn(*mut CuFunction,CuModule,*const c_char)->CuResult, launch: unsafe extern "C" fn(CuFunction,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,CuStream,*mut *mut c_void,*mut *mut c_void)->CuResult, error_name: unsafe extern "C" fn(CuResult,*mut *const c_char)->CuResult, error_string: unsafe extern "C" fn(CuResult,*mut *const c_char)->CuResult);
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
            link_create: library
                .symbol(b"cuLinkCreate_v2\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(
                            c_uint,
                            *mut u32,
                            *mut *mut c_void,
                            *mut CuLinkState,
                        ) -> CuResult,
                    >(p)
                }),
            link_add_data: library
                .symbol(b"cuLinkAddData_v2\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(
                            CuLinkState,
                            CuJitInputType,
                            *mut c_void,
                            usize,
                            *const c_char,
                            c_uint,
                            *mut u32,
                            *mut *mut c_void,
                        ) -> CuResult,
                    >(p)
                }),
            link_complete: library
                .symbol(b"cuLinkComplete\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(CuLinkState, *mut *mut c_void, *mut usize) -> CuResult,
                    >(p)
                }),
            link_destroy: library
                .symbol(b"cuLinkDestroy\0")
                .ok()
                .map(|p| unsafe {
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(CuLinkState) -> CuResult,
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
        let peer = NativePeerTable::resolve(|name| library.symbol(name).ok());
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
    fn supports_peer_async_transfers(&self) -> bool {
        self.peer.copy_async.is_some()
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
    fn link_create(
        &self,
        options: &[u32],
        values: &mut [*mut c_void],
        state: &mut CuLinkState,
    ) -> Result<CuResult, CudaError> {
        self.table
            .link_create
            .map(|f| unsafe {
                f(
                    options.len() as c_uint,
                    options.as_ptr().cast_mut(),
                    values.as_mut_ptr(),
                    state,
                )
            })
            .ok_or(CudaError::MissingSymbol("cuLinkCreate_v2"))
    }
    fn link_add_data(
        &self,
        state: CuLinkState,
        input: CuJitInputType,
        data: *const c_void,
        bytes: usize,
        name: &CStr,
        options: &[u32],
        values: &mut [*mut c_void],
    ) -> Result<CuResult, CudaError> {
        self.table
            .link_add_data
            .map(|f| unsafe {
                f(
                    state,
                    input,
                    data.cast_mut(),
                    bytes,
                    name.as_ptr(),
                    options.len() as c_uint,
                    options.as_ptr().cast_mut(),
                    values.as_mut_ptr(),
                )
            })
            .ok_or(CudaError::MissingSymbol("cuLinkAddData_v2"))
    }
    fn link_complete(
        &self,
        state: CuLinkState,
        image: &mut *mut c_void,
        bytes: &mut usize,
    ) -> Result<CuResult, CudaError> {
        self.table
            .link_complete
            .map(|f| unsafe { f(state, image, bytes) })
            .ok_or(CudaError::MissingSymbol("cuLinkComplete"))
    }
    fn link_destroy(&self, state: CuLinkState) -> Result<CuResult, CudaError> {
        self.table
            .link_destroy
            .map(|f| unsafe { f(state) })
            .ok_or(CudaError::MissingSymbol("cuLinkDestroy"))
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
    use std::collections::{HashMap, HashSet};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicUsize},
    };
    use std::thread::ThreadId;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct MockAllocationDescriptor {
        pub(crate) base: CuDevicePtr,
        pub(crate) generation: u64,
        pub(crate) device: DeviceId,
    }

    struct MockAllocation {
        base: CuDevicePtr,
        bytes: usize,
        data: Vec<u8>,
        generation: u64,
        alive: bool,
        device: DeviceId,
    }
    #[derive(Clone)]
    struct MockCollectiveAdd {
        source_key: String,
        dtype: crate::DType,
        abi_version: u32,
    }

    pub(crate) struct Mock {
        calls: Mutex<Vec<&'static str>>,
        current: Mutex<usize>,
        primary_owners: Mutex<HashMap<usize, DeviceId>>,
        primary_current: Mutex<HashMap<ThreadId, Vec<PrimaryOwner>>>,
        primary_peer_copy: Mutex<HashMap<ThreadId, (PrimaryOwner, PrimaryOwner)>>,
        allocations: Mutex<HashMap<usize, Vec<MockAllocation>>>,
        host_allocations: Mutex<HashMap<usize, Box<[u8]>>>,
        collective_adds: Mutex<HashMap<(usize, usize), MockCollectiveAdd>>,
        generic_kernels: Mutex<HashMap<(usize, usize), Arc<crate::ptx::GenericKernelSemantics>>>,
        link_states: Mutex<HashSet<usize>>,
        link_input_types: Mutex<Vec<CuJitInputType>>,
        next_allocation_generation: AtomicU64,
        next_function: AtomicUsize,
        next_link_state: AtomicUsize,
        launch_result: AtomicI32,
        launch_fail_after: AtomicUsize,
        launch_fail_result: AtomicI32,
        fail_alloc: AtomicBool,
        push_result: AtomicI32,
        pop_result: AtomicI32,
        module_result: AtomicI32,
        link_create_result: AtomicI32,
        link_add_data_result: AtomicI32,
        link_complete_result: AtomicI32,
        link_destroy_result: AtomicI32,
        ex: AtomicBool,
        ex_result: AtomicI32,
        null_module: AtomicBool,
        null_function: AtomicBool,
        capture_active: AtomicBool,
        capture_null_graph: AtomicBool,
        instantiate_null_exec: AtomicBool,
        elapsed_supported: AtomicBool,
        elapsed_result: AtomicI32,
        elapsed_millis: AtomicU32,
        event_ready: AtomicBool,
        peer_capable: AtomicBool,
        peer_async_supported: AtomicBool,
        peer_result: AtomicI32,
        peer_fail_after: AtomicUsize,
        peer_fail_result: AtomicI32,
        dtod_fail_after: AtomicUsize,
        dtod_fail_result: AtomicI32,
        peer_enable_result: AtomicI32,
        peer_disable_result: AtomicI32,
        event_record_result: AtomicI32,
        stream_wait_result: AtomicI32,
        stream_sync_result: AtomicI32,
    }
    impl Default for Mock {
        fn default() -> Self {
            Self {
                calls: Mutex::new(vec![]),
                current: Mutex::new(0),
                primary_owners: Mutex::new(HashMap::new()),
                primary_current: Mutex::new(HashMap::new()),
                primary_peer_copy: Mutex::new(HashMap::new()),
                allocations: Mutex::new(HashMap::new()),
                host_allocations: Mutex::new(HashMap::new()),
                collective_adds: Mutex::new(HashMap::new()),
                generic_kernels: Mutex::new(HashMap::new()),
                link_states: Mutex::new(HashSet::new()),
                link_input_types: Mutex::new(vec![]),
                next_allocation_generation: AtomicU64::new(1),
                next_function: AtomicUsize::new(0x55),
                next_link_state: AtomicUsize::new(0x66),
                launch_result: AtomicI32::new(0),
                launch_fail_after: AtomicUsize::new(usize::MAX),
                launch_fail_result: AtomicI32::new(0),
                fail_alloc: AtomicBool::new(false),
                push_result: AtomicI32::new(0),
                pop_result: AtomicI32::new(0),
                module_result: AtomicI32::new(0),
                link_create_result: AtomicI32::new(0),
                link_add_data_result: AtomicI32::new(0),
                link_complete_result: AtomicI32::new(0),
                link_destroy_result: AtomicI32::new(0),
                ex: AtomicBool::new(false),
                ex_result: AtomicI32::new(0),
                null_module: AtomicBool::new(false),
                null_function: AtomicBool::new(false),
                capture_active: AtomicBool::new(false),
                capture_null_graph: AtomicBool::new(false),
                instantiate_null_exec: AtomicBool::new(false),
                elapsed_supported: AtomicBool::new(false),
                elapsed_result: AtomicI32::new(0),
                elapsed_millis: AtomicU32::new(1.5_f32.to_bits()),
                event_ready: AtomicBool::new(false),
                peer_capable: AtomicBool::new(true),
                peer_async_supported: AtomicBool::new(true),
                peer_result: AtomicI32::new(0),
                peer_fail_after: AtomicUsize::new(usize::MAX),
                peer_fail_result: AtomicI32::new(0),
                dtod_fail_after: AtomicUsize::new(usize::MAX),
                dtod_fail_result: AtomicI32::new(0),
                peer_enable_result: AtomicI32::new(0),
                peer_disable_result: AtomicI32::new(0),
                event_record_result: AtomicI32::new(0),
                stream_wait_result: AtomicI32::new(0),
                stream_sync_result: AtomicI32::new(0),
            }
        }
    }
    impl Mock {
        const INVALID_MEMORY: CuResult = 1;
        #[allow(dead_code)]
        pub(crate) fn generic_kernel_count(&self) -> usize {
            self.generic_kernels.lock().unwrap().len()
        }

        fn call(&self, name: &'static str) {
            self.calls.lock().unwrap().push(name);
        }
        fn allocation_range(
            allocations: &[MockAllocation],
            ptr: CuDevicePtr,
            bytes: usize,
        ) -> Option<(usize, usize)> {
            allocations
                .iter()
                .enumerate()
                .find_map(|(index, allocation)| {
                    let offset = ptr.checked_sub(allocation.base)?;
                    let offset = usize::try_from(offset).ok()?;
                    let end = offset.checked_add(bytes)?;
                    (allocation.alive && end <= allocation.bytes).then_some((index, offset))
                })
        }
        fn current_primary(&self) -> Option<PrimaryOwner> {
            self.primary_owner_current()
        }
        fn copy_from_host(
            &self,
            owner: PrimaryOwner,
            dst: CuDevicePtr,
            src: *const c_void,
            bytes: usize,
        ) -> CuResult {
            let source = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), bytes) };
            let mut all = self.allocations.lock().unwrap();
            let Some(records) = all.get_mut(&owner.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some((index, offset)) = Self::allocation_range(records, dst, bytes) else {
                return Self::INVALID_MEMORY;
            };
            records[index].data[offset..offset + bytes].copy_from_slice(source);
            CUDA_SUCCESS
        }
        fn copy_to_host(
            &self,
            dst: *mut c_void,
            owner: PrimaryOwner,
            src: CuDevicePtr,
            bytes: usize,
        ) -> CuResult {
            let mut all = self.allocations.lock().unwrap();
            let Some(records) = all.get_mut(&owner.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some((index, offset)) = Self::allocation_range(records, src, bytes) else {
                return Self::INVALID_MEMORY;
            };
            let target = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), bytes) };
            target.copy_from_slice(&records[index].data[offset..offset + bytes]);
            CUDA_SUCCESS
        }
        fn copy_within_owner(
            &self,
            owner: PrimaryOwner,
            dst: CuDevicePtr,
            src: CuDevicePtr,
            bytes: usize,
        ) -> CuResult {
            let mut all = self.allocations.lock().unwrap();
            let Some(records) = all.get_mut(&owner.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some((source_index, source_offset)) = Self::allocation_range(records, src, bytes)
            else {
                return Self::INVALID_MEMORY;
            };
            let source = records[source_index].data[source_offset..source_offset + bytes].to_vec();
            let Some((destination_index, destination_offset)) =
                Self::allocation_range(records, dst, bytes)
            else {
                return Self::INVALID_MEMORY;
            };
            records[destination_index].data[destination_offset..destination_offset + bytes]
                .copy_from_slice(&source);
            CUDA_SUCCESS
        }
        fn copy_between_owners(
            &self,
            destination: PrimaryOwner,
            dst: CuDevicePtr,
            source: PrimaryOwner,
            src: CuDevicePtr,
            bytes: usize,
        ) -> CuResult {
            let mut all = self.allocations.lock().unwrap();
            let Some(source_records) = all.get(&source.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some((source_index, source_offset)) =
                Self::allocation_range(source_records, src, bytes)
            else {
                return Self::INVALID_MEMORY;
            };
            let data =
                source_records[source_index].data[source_offset..source_offset + bytes].to_vec();
            let Some(destination_records) = all.get_mut(&destination.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some((destination_index, destination_offset)) =
                Self::allocation_range(destination_records, dst, bytes)
            else {
                return Self::INVALID_MEMORY;
            };
            destination_records[destination_index].data
                [destination_offset..destination_offset + bytes]
                .copy_from_slice(&data);
            CUDA_SUCCESS
        }
        fn collective_add(
            &self,
            owner: PrimaryOwner,
            dtype: crate::DType,
            dst: CuDevicePtr,
            src: CuDevicePtr,
            count: usize,
        ) -> CuResult {
            let bytes = match count.checked_mul(dtype.itemsize()) {
                Some(bytes) => bytes,
                None => return Self::INVALID_MEMORY,
            };
            let mut all = self.allocations.lock().unwrap();
            let Some(records) = all.get_mut(&owner.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some((source_index, source_offset)) = Self::allocation_range(records, src, bytes)
            else {
                return Self::INVALID_MEMORY;
            };
            let source = records[source_index].data[source_offset..source_offset + bytes].to_vec();
            let Some((destination_index, destination_offset)) =
                Self::allocation_range(records, dst, bytes)
            else {
                return Self::INVALID_MEMORY;
            };
            let destination = &mut records[destination_index].data
                [destination_offset..destination_offset + bytes];
            for (dst, src) in destination
                .chunks_exact_mut(dtype.itemsize())
                .zip(source.chunks_exact(dtype.itemsize()))
            {
                let old = dst.to_vec();
                match dtype {
                    crate::DType::I8 => {
                        dst[0] = (i8::from_ne_bytes([dst[0]])
                            .wrapping_add(i8::from_ne_bytes([src[0]])))
                            as u8
                    }
                    crate::DType::U8 => dst[0] = dst[0].wrapping_add(src[0]),
                    crate::DType::I32 => dst.copy_from_slice(
                        &i32::from_ne_bytes(old.try_into().unwrap())
                            .wrapping_add(i32::from_ne_bytes(src.try_into().unwrap()))
                            .to_ne_bytes(),
                    ),
                    crate::DType::U32 => dst.copy_from_slice(
                        &u32::from_ne_bytes(old.try_into().unwrap())
                            .wrapping_add(u32::from_ne_bytes(src.try_into().unwrap()))
                            .to_ne_bytes(),
                    ),
                    crate::DType::I64 => dst.copy_from_slice(
                        &i64::from_ne_bytes(old.try_into().unwrap())
                            .wrapping_add(i64::from_ne_bytes(src.try_into().unwrap()))
                            .to_ne_bytes(),
                    ),
                    crate::DType::U64 => dst.copy_from_slice(
                        &u64::from_ne_bytes(old.try_into().unwrap())
                            .wrapping_add(u64::from_ne_bytes(src.try_into().unwrap()))
                            .to_ne_bytes(),
                    ),
                    crate::DType::F32 => dst.copy_from_slice(
                        &(f32::from_ne_bytes(old.try_into().unwrap())
                            + f32::from_ne_bytes(src.try_into().unwrap()))
                        .to_ne_bytes(),
                    ),
                    crate::DType::F64 => dst.copy_from_slice(
                        &(f64::from_ne_bytes(old.try_into().unwrap())
                            + f64::from_ne_bytes(src.try_into().unwrap()))
                        .to_ne_bytes(),
                    ),
                    _ => return Self::INVALID_MEMORY,
                }
            }
            CUDA_SUCCESS
        }
        /// Test-only execution of renderer-retained semantics.  The production
        /// dispatch never takes this path: it submits the PTX image to CUDA.
        fn generic_kernel_launch(
            &self,
            owner: PrimaryOwner,
            function: CuFunction,
            args: *mut *mut c_void,
        ) -> CuResult {
            let Some(semantics) = self
                .generic_kernels
                .lock()
                .unwrap()
                .get(&(owner.identity, function as usize))
                .cloned()
            else {
                return CUDA_SUCCESS;
            };
            if args.is_null() {
                return Self::INVALID_MEMORY;
            }
            let mut words = Vec::with_capacity(semantics.buffers.len() + 1);
            unsafe {
                for index in 0..=semantics.buffers.len() {
                    let word = *args.add(index);
                    if word.is_null() {
                        return Self::INVALID_MEMORY;
                    }
                    words.push(*(word as *const u64));
                }
            }
            if words.last().copied() != Some(semantics.extent as u64) {
                return Self::INVALID_MEMORY;
            }
            if semantics.extent == 0 {
                return CUDA_SUCCESS;
            }
            let output_index = match &semantics.program {
                crate::ptx::KernelSemanticProgram::UOp(program) => match program
                    .sources()
                    .iter()
                    .find(|node| matches!(node.kind(), crate::UOpKind::Store))
                    .and_then(|store| store.sources().first())
                    .map(|index| index.arg())
                {
                    Some(crate::UArg::BufferIndex { buffer, .. }) => *buffer,
                    _ if matches!(program.kind(), crate::UOpKind::Random) => match program.arg() {
                        crate::UArg::Random(plan) => plan.output.index() as u64,
                        _ => return Self::INVALID_MEMORY,
                    },
                    _ => return Self::INVALID_MEMORY,
                },
                crate::ptx::KernelSemanticProgram::Matmul(plan) => plan.output.index() as u64,
                crate::ptx::KernelSemanticProgram::TiledMatmul(payload) => {
                    payload.matmul.output.index() as u64
                }
                crate::ptx::KernelSemanticProgram::TensorCoreMatmul(payload) => {
                    payload.matmul.output.index() as u64
                }
            };
            let mut bindings = crate::KernelBindings::default();
            let mut values = Vec::new();
            let mut output = None;
            {
                let all = self.allocations.lock().unwrap();
                let Some(records) = all.get(&owner.identity) else {
                    return Self::INVALID_MEMORY;
                };
                for (abi, pointer) in semantics.buffers.iter().zip(&words) {
                    let Ok(elements) = abi.source_shape.numel() else {
                        return Self::INVALID_MEMORY;
                    };
                    if elements != abi.elements {
                        return Self::INVALID_MEMORY;
                    }
                    let Some(bytes) = abi.elements.checked_mul(abi.dtype.itemsize()) else {
                        return Self::INVALID_MEMORY;
                    };
                    let Some((record, offset)) = Self::allocation_range(records, *pointer, bytes)
                    else {
                        return Self::INVALID_MEMORY;
                    };
                    if abi.mutable {
                        if abi.id != output_index || output.is_some() {
                            return Self::INVALID_MEMORY;
                        }
                        output = Some((
                            record,
                            offset,
                            records[record].base,
                            records[record].generation,
                        ));
                    }
                    let Ok(value) = crate::TensorData::from_le_bytes(
                        abi.source_shape.clone(),
                        abi.dtype,
                        &records[record].data[offset..offset + bytes],
                    ) else {
                        return Self::INVALID_MEMORY;
                    };
                    let Ok(desc) = crate::KernelBufferDesc::concrete(
                        abi.id,
                        if abi.mutable {
                            crate::BufferRole::Output
                        } else {
                            crate::BufferRole::Input
                        },
                        abi.source_shape.clone(),
                        abi.dtype,
                        abi.mutable,
                    ) else {
                        return Self::INVALID_MEMORY;
                    };
                    values.push(value.clone());
                    if bindings.insert(&desc, value).is_err() {
                        return Self::INVALID_MEMORY;
                    }
                }
            }
            let Some((out_record, out_offset, out_base, out_generation)) = output else {
                return Self::INVALID_MEMORY;
            };
            // Do not give the test evaluator memmove-like alias semantics:
            // generic local kernels have distinct input/output bindings.
            for (abi, pointer) in semantics.buffers.iter().zip(&words) {
                if abi.mutable {
                    continue;
                }
                let bytes = match abi.elements.checked_mul(abi.dtype.itemsize()) {
                    Some(bytes) => bytes,
                    None => return Self::INVALID_MEMORY,
                };
                if *pointer == words[semantics.buffers.iter().position(|x| x.mutable).unwrap()]
                    && bytes != 0
                {
                    return Self::INVALID_MEMORY;
                }
            }
            let result = match &semantics.program {
                crate::ptx::KernelSemanticProgram::UOp(program) => {
                    crate::kernel::execute_lowered_elementwise(program, &bindings)
                }
                crate::ptx::KernelSemanticProgram::Matmul(plan) => {
                    if semantics.buffers.len() != 3
                        || semantics.buffers[0].id != plan.lhs.index() as u64
                        || semantics.buffers[1].id != plan.rhs.index() as u64
                        || semantics.buffers[2].id != plan.output.index() as u64
                    {
                        return Self::INVALID_MEMORY;
                    }
                    plan.execute(&values[0], &values[1])
                        .map_err(|_| crate::Error::InvalidIndex)
                }
                crate::ptx::KernelSemanticProgram::TiledMatmul(payload) => {
                    let plan = &payload.matmul;
                    if semantics.buffers.len() != 3
                        || semantics.buffers[0].id != plan.lhs.index() as u64
                        || semantics.buffers[1].id != plan.rhs.index() as u64
                        || semantics.buffers[2].id != plan.output.index() as u64
                    {
                        return Self::INVALID_MEMORY;
                    }
                    payload
                        .simulate(&values[0], &values[1])
                        .map_err(|_| crate::Error::InvalidIndex)
                }
                crate::ptx::KernelSemanticProgram::TensorCoreMatmul(payload) => {
                    let plan = &payload.matmul;
                    if semantics.buffers.len() != 3
                        || semantics.buffers[0].id != plan.lhs.index() as u64
                        || semantics.buffers[1].id != plan.rhs.index() as u64
                        || semantics.buffers[2].id != plan.output.index() as u64
                    {
                        return Self::INVALID_MEMORY;
                    }
                    payload
                        .simulate(&values[0], &values[1])
                        .map_err(|_| crate::Error::InvalidIndex)
                }
            };
            let Ok(result) = result else {
                return Self::INVALID_MEMORY;
            };
            let Ok(result_bytes) = result.to_le_bytes() else {
                return Self::INVALID_MEMORY;
            };
            let Some(output_abi) = semantics.buffers.iter().find(|abi| abi.mutable) else {
                return Self::INVALID_MEMORY;
            };
            let Some(expected) = output_abi.elements.checked_mul(output_abi.dtype.itemsize())
            else {
                return Self::INVALID_MEMORY;
            };
            if result.dtype() != output_abi.dtype || result_bytes.len() != expected {
                return Self::INVALID_MEMORY;
            }
            let mut all = self.allocations.lock().unwrap();
            let Some(records) = all.get_mut(&owner.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some(record) = records.get_mut(out_record) else {
                return Self::INVALID_MEMORY;
            };
            if !record.alive || record.base != out_base || record.generation != out_generation {
                return Self::INVALID_MEMORY;
            }
            let Some(end) = out_offset.checked_add(result_bytes.len()) else {
                return Self::INVALID_MEMORY;
            };
            if end > record.bytes {
                return Self::INVALID_MEMORY;
            }
            record.data[out_offset..end].copy_from_slice(&result_bytes);
            CUDA_SUCCESS
        }
        pub(crate) fn calls(&self) -> Vec<&'static str> {
            self.calls.lock().unwrap().clone()
        }
        pub(crate) fn live_link_state_count(&self) -> usize {
            self.link_states.lock().unwrap().len()
        }
        pub(crate) fn link_input_types(&self) -> Vec<CuJitInputType> {
            self.link_input_types.lock().unwrap().clone()
        }
        pub(crate) fn registered_primary_owner(&self, identity: usize) -> Option<DeviceId> {
            self.primary_owners.lock().unwrap().get(&identity).copied()
        }
        pub(crate) fn current_primary_owner(&self) -> Option<PrimaryOwner> {
            self.primary_owner_current()
        }
        pub(crate) fn current_primary_owner_on(&self, thread: ThreadId) -> Option<PrimaryOwner> {
            self.primary_current
                .lock()
                .unwrap()
                .get(&thread)
                .and_then(|owners| owners.last().copied())
        }
        pub(crate) fn allocation_descriptor(
            &self,
            owner: PrimaryOwner,
            base: CuDevicePtr,
        ) -> Option<MockAllocationDescriptor> {
            self.allocations
                .lock()
                .unwrap()
                .get(&owner.identity)?
                .iter()
                .find(|allocation| allocation.base == base)
                .map(|allocation| MockAllocationDescriptor {
                    base: allocation.base,
                    generation: allocation.generation,
                    device: allocation.device,
                })
        }
        pub(crate) fn allocation_snapshot(
            &self,
            owner: PrimaryOwner,
            descriptor: MockAllocationDescriptor,
        ) -> Option<Vec<u8>> {
            self.allocations
                .lock()
                .unwrap()
                .get(&owner.identity)?
                .iter()
                .find(|allocation| {
                    allocation.base == descriptor.base
                        && allocation.generation == descriptor.generation
                        && allocation.device == descriptor.device
                        && allocation.alive
                })
                .map(|allocation| allocation.data.clone())
        }
        pub(crate) fn write_allocation(
            &self,
            owner: PrimaryOwner,
            descriptor: MockAllocationDescriptor,
            offset: usize,
            bytes: &[u8],
        ) -> Result<(), CudaError> {
            let mut all = self.allocations.lock().unwrap();
            let allocation = all
                .get_mut(&owner.identity)
                .and_then(|allocations| {
                    allocations.iter_mut().find(|allocation| {
                        allocation.base == descriptor.base
                            && allocation.generation == descriptor.generation
                            && allocation.device == descriptor.device
                            && allocation.alive
                    })
                })
                .ok_or(CudaError::Closed("mock allocation"))?;
            let end = offset.checked_add(bytes.len()).ok_or(CudaError::Overflow)?;
            if end > allocation.bytes {
                return Err(CudaError::InvalidArgument("mock allocation range"));
            }
            allocation.data[offset..end].copy_from_slice(bytes);
            Ok(())
        }
        pub(crate) fn live_allocation_count(&self, owner: PrimaryOwner) -> usize {
            self.allocations
                .lock()
                .unwrap()
                .get(&owner.identity)
                .map_or(0, |allocations| {
                    allocations.iter().filter(|x| x.alive).count()
                })
        }
        pub(crate) fn set_push_result(&self, result: CuResult) {
            self.push_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_pop_result(&self, result: CuResult) {
            self.pop_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_launch_result(&self, result: CuResult) {
            self.launch_result.store(result, Ordering::Release);
        }
        /// Fails allocation before mock storage is made live.
        pub(crate) fn set_allocation_failure(&self, fail: bool) {
            self.fail_alloc.store(fail, Ordering::Release);
        }
        /// Fails one collective add after `successful_calls` more launches.
        pub(crate) fn fail_launch_after(&self, successful_calls: usize, result: CuResult) {
            self.launch_fail_result.store(result, Ordering::Release);
            self.launch_fail_after
                .store(successful_calls, Ordering::Release);
        }
        pub(crate) fn set_module_result(&self, result: i32) {
            self.module_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_link_create_result(&self, result: CuResult) {
            self.link_create_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_link_add_data_result(&self, result: CuResult) {
            self.link_add_data_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_link_complete_result(&self, result: CuResult) {
            self.link_complete_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_link_destroy_result(&self, result: CuResult) {
            self.link_destroy_result.store(result, Ordering::Release);
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
        pub(crate) fn set_peer_async_supported(&self, supported: bool) {
            self.peer_async_supported
                .store(supported, Ordering::Release);
        }
        pub(crate) fn set_peer_result(&self, result: CuResult) {
            self.peer_result.store(result, Ordering::Release);
        }
        /// Fails one peer copy after `successful_calls` more peer submissions.
        pub(crate) fn fail_peer_after(&self, successful_calls: usize, result: CuResult) {
            self.peer_fail_result.store(result, Ordering::Release);
            self.peer_fail_after
                .store(successful_calls, Ordering::Release);
        }
        /// Fails one same-owner DtoD submission before it mutates mock bytes.
        pub(crate) fn fail_dtod_after(&self, successful_calls: usize, result: CuResult) {
            self.dtod_fail_result.store(result, Ordering::Release);
            self.dtod_fail_after
                .store(successful_calls, Ordering::Release);
        }
        pub(crate) fn set_peer_enable_result(&self, result: CuResult) {
            self.peer_enable_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_peer_disable_result(&self, result: CuResult) {
            self.peer_disable_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_event_record_result(&self, result: CuResult) {
            self.event_record_result.store(result, Ordering::Release);
        }
        /// Configures the mock `cuStreamWaitEvent` result without changing
        /// event ownership or readiness. A failed wait never establishes a
        /// dependency, matching the Driver submission boundary.
        pub(crate) fn set_stream_wait_result(&self, result: CuResult) {
            self.stream_wait_result.store(result, Ordering::Release);
        }
        pub(crate) fn set_stream_sync_result(&self, result: CuResult) {
            self.stream_sync_result.store(result, Ordering::Release);
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
            self.push_result.load(Ordering::Acquire)
        }
        fn ctx_pop_current(&self, out: &mut CuContext) -> CuResult {
            self.call("ctx_pop");
            *out = 0x77usize as CuContext;
            self.pop_result.load(Ordering::Acquire)
        }
        fn primary_owner_register(&self, owner: PrimaryOwner) {
            let old = self
                .primary_owners
                .lock()
                .unwrap()
                .insert(owner.identity, owner.device);
            assert!(old.is_none(), "primary owner was registered twice");
        }
        fn primary_owner_unregister(&self, owner: PrimaryOwner) {
            let removed = self.primary_owners.lock().unwrap().remove(&owner.identity);
            assert_eq!(removed, Some(owner.device), "unknown primary owner");
            // Resource RAII normally frees every allocation before the final
            // primary release. Clean any remaining deterministic mock storage
            // so a leaked test allocation cannot outlive its owner.
            self.allocations.lock().unwrap().remove(&owner.identity);
            self.primary_peer_copy
                .lock()
                .unwrap()
                .retain(|_, pair| pair.0 != owner && pair.1 != owner);
        }
        fn primary_owner_enter(&self, owner: PrimaryOwner) {
            assert_eq!(
                self.registered_primary_owner(owner.identity),
                Some(owner.device),
                "entered unregistered primary owner"
            );
            self.primary_current
                .lock()
                .unwrap()
                .entry(std::thread::current().id())
                .or_default()
                .push(owner);
        }
        fn primary_owner_exit(&self, owner: PrimaryOwner) {
            let thread = std::thread::current().id();
            let mut currents = self.primary_current.lock().unwrap();
            let owners = currents
                .get_mut(&thread)
                .expect("exited primary owner without a current stack");
            assert_eq!(owners.pop(), Some(owner), "primary owner stack mismatch");
            if owners.is_empty() {
                currents.remove(&thread);
            }
        }
        fn primary_owner_current(&self) -> Option<PrimaryOwner> {
            self.current_primary_owner_on(std::thread::current().id())
        }
        fn primary_owner_peer_copy(&self, source: PrimaryOwner, destination: PrimaryOwner) {
            self.primary_peer_copy
                .lock()
                .unwrap()
                .insert(std::thread::current().id(), (source, destination));
        }
        fn primary_owner_register_collective_add(
            &self,
            owner: PrimaryOwner,
            function: usize,
            source_key: &str,
            dtype: crate::DType,
            abi_version: u32,
        ) {
            self.collective_adds.lock().unwrap().insert(
                (owner.identity, function),
                MockCollectiveAdd {
                    source_key: source_key.into(),
                    dtype,
                    abi_version,
                },
            );
        }
        fn primary_owner_register_generic_kernel(
            &self,
            owner: PrimaryOwner,
            function: usize,
            _: &str,
            semantics: Arc<crate::ptx::GenericKernelSemantics>,
        ) {
            let mut kernels = self.generic_kernels.lock().unwrap();
            if let Some(old) = kernels.get(&(owner.identity, function)) {
                assert_eq!(
                    old.key, semantics.key,
                    "incompatible generic semantic registration"
                );
            } else {
                kernels.insert((owner.identity, function), semantics);
            }
        }
        fn primary_owner_unregister_generic_kernel(&self, owner: PrimaryOwner, function: usize) {
            self.generic_kernels
                .lock()
                .unwrap()
                .remove(&(owner.identity, function));
        }
        fn mem_alloc(&self, out: &mut CuDevicePtr, bytes: usize) -> CuResult {
            self.call("alloc");
            if self.fail_alloc.load(Ordering::Acquire) {
                2
            } else {
                let Some(owner) = self.current_primary() else {
                    *out = 0x1000;
                    return CUDA_SUCCESS;
                };
                let mut data = Vec::new();
                if data.try_reserve_exact(bytes).is_err() {
                    return 2;
                }
                data.resize(bytes, 0);
                let mut all = self.allocations.lock().unwrap();
                let records = all.entry(owner.identity).or_default();
                let base = records
                    .last()
                    .and_then(|allocation| {
                        allocation
                            .base
                            .checked_add(u64::try_from(allocation.bytes).ok()?)
                            .and_then(|end| end.checked_add(0x100))
                    })
                    .unwrap_or(0x1000);
                *out = base;
                records.push(MockAllocation {
                    base,
                    bytes,
                    data,
                    generation: self
                        .next_allocation_generation
                        .fetch_add(1, Ordering::AcqRel),
                    alive: true,
                    device: owner.device,
                });
                0
            }
        }
        fn mem_free(&self, ptr: CuDevicePtr) -> CuResult {
            self.call("free");
            let Some(owner) = self.current_primary() else {
                return CUDA_SUCCESS;
            };
            let mut all = self.allocations.lock().unwrap();
            let Some(records) = all.get_mut(&owner.identity) else {
                return Self::INVALID_MEMORY;
            };
            let Some(allocation) = records
                .iter_mut()
                .find(|allocation| allocation.base == ptr && allocation.alive)
            else {
                return Self::INVALID_MEMORY;
            };
            allocation.alive = false;
            CUDA_SUCCESS
        }
        fn memcpy_htod(&self, dst: CuDevicePtr, src: *const c_void, bytes: usize) -> CuResult {
            self.call("htod");
            self.current_primary().map_or(CUDA_SUCCESS, |owner| {
                self.copy_from_host(owner, dst, src, bytes)
            })
        }
        fn memcpy_dtoh(&self, dst: *mut c_void, src: CuDevicePtr, bytes: usize) -> CuResult {
            self.call("dtoh");
            self.current_primary().map_or(CUDA_SUCCESS, |owner| {
                self.copy_to_host(dst, owner, src, bytes)
            })
        }
        fn memcpy_dtod(&self, dst: CuDevicePtr, src: CuDevicePtr, bytes: usize) -> CuResult {
            self.call("dtod");
            if self
                .dtod_fail_after
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                    (left != usize::MAX).then(|| if left == 0 { usize::MAX } else { left - 1 })
                })
                .ok()
                == Some(0)
            {
                return self.dtod_fail_result.load(Ordering::Acquire);
            }
            self.current_primary().map_or(CUDA_SUCCESS, |owner| {
                self.copy_within_owner(owner, dst, src, bytes)
            })
        }
        fn device_can_access_peer(&self, out: &mut c_int, _: CuDevice, _: CuDevice) -> CuResult {
            self.call("peer_can");
            *out = self.peer_capable.load(Ordering::Acquire) as c_int;
            self.peer_result.load(Ordering::Acquire)
        }
        fn ctx_enable_peer_access(&self, _: CuContext, _: c_uint) -> CuResult {
            self.call("peer_enable");
            self.peer_enable_result.load(Ordering::Acquire)
        }
        fn ctx_disable_peer_access(&self, _: CuContext) -> CuResult {
            self.call("peer_disable");
            self.peer_disable_result.load(Ordering::Acquire)
        }
        fn memcpy_peer_async(
            &self,
            dst: CuDevicePtr,
            dst_context: CuContext,
            src: CuDevicePtr,
            src_context: CuContext,
            bytes: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("peer_copy");
            let result = self.peer_result.load(Ordering::Acquire);
            if result != CUDA_SUCCESS {
                return result;
            }
            if self
                .peer_fail_after
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                    (left != usize::MAX).then(|| if left == 0 { usize::MAX } else { left - 1 })
                })
                .ok()
                == Some(0)
            {
                return self.peer_fail_result.load(Ordering::Acquire);
            }
            let Some((source, destination)) = self
                .primary_peer_copy
                .lock()
                .unwrap()
                .get(&std::thread::current().id())
                .copied()
            else {
                return Self::INVALID_MEMORY;
            };
            if self.current_primary() != Some(destination)
                || self.registered_primary_owner(source.identity) != Some(source.device)
                || self.registered_primary_owner(destination.identity) != Some(destination.device)
                || dst_context != 0x77usize as CuContext
                || src_context != 0x77usize as CuContext
            {
                return Self::INVALID_MEMORY;
            }
            self.copy_between_owners(destination, dst, source, src, bytes)
        }
        fn supports_async_transfers(&self) -> bool {
            true
        }
        fn supports_peer_async_transfers(&self) -> bool {
            self.peer_async_supported.load(Ordering::Acquire)
        }
        fn supports_pinned_host_memory(&self) -> bool {
            true
        }
        fn memcpy_htod_async(
            &self,
            dst: CuDevicePtr,
            src: *const c_void,
            bytes: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("htod_async");
            self.current_primary().map_or(CUDA_SUCCESS, |owner| {
                self.copy_from_host(owner, dst, src, bytes)
            })
        }
        fn memcpy_dtoh_async(
            &self,
            dst: *mut c_void,
            src: CuDevicePtr,
            bytes: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("dtoh_async");
            self.current_primary().map_or(CUDA_SUCCESS, |owner| {
                self.copy_to_host(dst, owner, src, bytes)
            })
        }
        fn memcpy_dtod_async(
            &self,
            dst: CuDevicePtr,
            src: CuDevicePtr,
            bytes: usize,
            _: CuStream,
        ) -> CuResult {
            self.call("dtod_async");
            if self
                .dtod_fail_after
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                    (left != usize::MAX).then(|| if left == 0 { usize::MAX } else { left - 1 })
                })
                .ok()
                == Some(0)
            {
                return self.dtod_fail_result.load(Ordering::Acquire);
            }
            self.current_primary().map_or(CUDA_SUCCESS, |owner| {
                self.copy_within_owner(owner, dst, src, bytes)
            })
        }
        fn mem_host_alloc(&self, out: &mut *mut c_void, bytes: usize, _: c_uint) -> CuResult {
            self.call("host_alloc");
            let mut data = Vec::new();
            if data.try_reserve_exact(bytes).is_err() {
                return 2;
            }
            data.resize(bytes, 0);
            let mut data = data.into_boxed_slice();
            *out = data.as_mut_ptr().cast();
            self.host_allocations
                .lock()
                .unwrap()
                .insert(*out as usize, data);
            CUDA_SUCCESS
        }
        fn mem_free_host(&self, ptr: *mut c_void) -> CuResult {
            self.call("host_free");
            self.host_allocations
                .lock()
                .unwrap()
                .remove(&(ptr as usize))
                .map_or(Self::INVALID_MEMORY, |_| CUDA_SUCCESS)
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
            *out = if self.capture_null_graph.load(Ordering::Acquire) {
                ptr::null_mut()
            } else {
                0x99usize as CuGraph
            };
            0
        }
        fn graph_instantiate(&self, out: &mut CuGraphExec, _: CuGraph) -> CuResult {
            self.call("graph_instantiate");
            *out = if self.instantiate_null_exec.load(Ordering::Acquire) {
                ptr::null_mut()
            } else {
                0xaausize as CuGraphExec
            };
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
            self.stream_sync_result.load(Ordering::Acquire)
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
            self.event_record_result.load(Ordering::Acquire)
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
            self.stream_wait_result.load(Ordering::Acquire)
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
            *out = if self.null_module.load(Ordering::Acquire) {
                ptr::null_mut()
            } else {
                0x44usize as CuModule
            };
            self.module_result.load(Ordering::Acquire)
        }
        fn link_create(
            &self,
            _: &[u32],
            _: &mut [*mut c_void],
            state: &mut CuLinkState,
        ) -> Result<CuResult, CudaError> {
            self.call("link_create");
            let result = self.link_create_result.load(Ordering::Acquire);
            if result == CUDA_SUCCESS {
                let raw = self.next_link_state.fetch_add(1, Ordering::AcqRel);
                self.link_states.lock().unwrap().insert(raw);
                *state = raw as CuLinkState;
            }
            Ok(result)
        }
        fn link_add_data(
            &self,
            state: CuLinkState,
            input: CuJitInputType,
            _: *const c_void,
            _: usize,
            _: &CStr,
            _: &[u32],
            _: &mut [*mut c_void],
        ) -> Result<CuResult, CudaError> {
            self.call("link_add_data");
            self.link_input_types.lock().unwrap().push(input);
            if !self.link_states.lock().unwrap().contains(&(state as usize)) {
                return Ok(Self::INVALID_MEMORY);
            }
            Ok(self.link_add_data_result.load(Ordering::Acquire))
        }
        fn link_complete(
            &self,
            state: CuLinkState,
            image: &mut *mut c_void,
            bytes: &mut usize,
        ) -> Result<CuResult, CudaError> {
            self.call("link_complete");
            if !self.link_states.lock().unwrap().contains(&(state as usize)) {
                return Ok(Self::INVALID_MEMORY);
            }
            let result = self.link_complete_result.load(Ordering::Acquire);
            if result == CUDA_SUCCESS {
                *image = 0x77usize as *mut c_void;
                *bytes = 1;
            }
            Ok(result)
        }
        fn link_destroy(&self, state: CuLinkState) -> Result<CuResult, CudaError> {
            self.call("link_destroy");
            let result = self.link_destroy_result.load(Ordering::Acquire);
            self.link_states.lock().unwrap().remove(&(state as usize));
            Ok(result)
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
            *out = if self.null_module.load(Ordering::Acquire) {
                ptr::null_mut()
            } else {
                0x44usize as CuModule
            };
            self.ex_result.load(Ordering::Acquire)
        }
        fn module_unload(&self, _: CuModule) -> CuResult {
            self.call("module_unload");
            0
        }
        fn module_function(&self, out: &mut CuFunction, _: CuModule, _: &CStr) -> CuResult {
            self.call("function");
            *out = if self.null_function.load(Ordering::Acquire) {
                ptr::null_mut()
            } else {
                self.next_function.fetch_add(1, Ordering::AcqRel) as CuFunction
            };
            0
        }
        fn launch(
            &self,
            function: CuFunction,
            _: [u32; 3],
            _: [u32; 3],
            _: u32,
            _: CuStream,
            args: *mut *mut c_void,
        ) -> CuResult {
            self.call("launch");
            let result = self.launch_result.load(Ordering::Acquire);
            if result != CUDA_SUCCESS {
                return result;
            }
            if self
                .launch_fail_after
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                    (left != usize::MAX).then(|| if left == 0 { usize::MAX } else { left - 1 })
                })
                .ok()
                == Some(0)
            {
                return self.launch_fail_result.load(Ordering::Acquire);
            }
            let Some(owner) = self.current_primary() else {
                return CUDA_SUCCESS;
            };
            if self
                .generic_kernels
                .lock()
                .unwrap()
                .contains_key(&(owner.identity, function as usize))
            {
                return self.generic_kernel_launch(owner, function, args);
            }
            let Some(contract) = self
                .collective_adds
                .lock()
                .unwrap()
                .get(&(owner.identity, function as usize))
                .cloned()
            else {
                return CUDA_SUCCESS;
            };
            if contract.abi_version != 1 || contract.source_key.is_empty() || args.is_null() {
                return Self::INVALID_MEMORY;
            }
            let words = unsafe {
                let mut values = [0_u64; 5];
                for (index, value) in values.iter_mut().enumerate() {
                    let word = *args.add(index);
                    if word.is_null() {
                        return Self::INVALID_MEMORY;
                    }
                    *value = *(word as *const u64);
                }
                values
            };
            let dst =
                words[0].saturating_add(words[2].saturating_mul(contract.dtype.itemsize() as u64));
            let src =
                words[1].saturating_add(words[3].saturating_mul(contract.dtype.itemsize() as u64));
            let Ok(count) = usize::try_from(words[4]) else {
                return Self::INVALID_MEMORY;
            };
            self.collective_add(owner, contract.dtype, dst, src, count)
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
    fn mock_stream_wait_failure_is_typed_atomic_and_retryable() {
        let mock = Arc::new(Mock::default());
        let ctx = context(&mock);
        let stream = ctx.stream().unwrap();
        let event = ctx.event().unwrap();

        mock.set_stream_wait_result(2);
        assert!(matches!(
            stream.wait(&event),
            Err(CudaError::Driver { code: 2, .. })
        ));
        // A submission failure has not consumed either resource; an unchanged
        // owned event and stream may be retried after the Driver recovers.
        mock.set_stream_wait_result(CUDA_SUCCESS);
        stream.wait(&event).unwrap();

        let foreign = context(&mock);
        let foreign_stream = foreign.stream().unwrap();
        let before = mock
            .calls()
            .into_iter()
            .filter(|call| *call == "stream_wait")
            .count();
        assert!(matches!(
            foreign_stream.wait(&event),
            Err(CudaError::ContextMismatch)
        ));
        assert_eq!(
            mock.calls()
                .into_iter()
                .filter(|call| *call == "stream_wait")
                .count(),
            before,
            "owner validation must reject before the Driver wait call"
        );
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
        assert!(calls.contains(&"ctx_push") && calls.contains(&"ctx_pop"));
    }

    #[test]
    fn primary_owner_observation_distinguishes_colliding_raw_handles() {
        let mock = Arc::new(Mock::default());
        let device = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap();
        let first = device.retain_primary_context().unwrap();
        let second = device.retain_primary_context().unwrap();
        assert_ne!(first.identity(), second.identity());
        assert_eq!(
            mock.registered_primary_owner(first.identity()),
            Some(DeviceId(0))
        );
        assert_eq!(
            mock.registered_primary_owner(second.identity()),
            Some(DeviceId(0))
        );

        {
            let _first = first.enter().unwrap();
            assert_eq!(mock.current_primary_owner(), Some(first.owner()));
            {
                let _second = second.enter().unwrap();
                assert_eq!(mock.current_primary_owner(), Some(second.owner()));
            }
            assert_eq!(mock.current_primary_owner(), Some(first.owner()));
        }
        assert_eq!(mock.current_primary_owner(), None);
        let first_id = first.identity();
        let second_id = second.identity();
        drop(first);
        drop(second);
        assert_eq!(mock.registered_primary_owner(first_id), None);
        assert_eq!(mock.registered_primary_owner(second_id), None);
        assert!(
            mock.calls()
                .iter()
                .all(|call| !call.starts_with("primary_owner"))
        );
    }

    #[test]
    fn failed_primary_push_and_pop_leave_observation_coherent() {
        let mock = Arc::new(Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        mock.set_push_result(2);
        assert!(primary.enter().is_err());
        assert_eq!(mock.current_primary_owner(), None);

        mock.set_push_result(CUDA_SUCCESS);
        mock.set_pop_result(2);
        {
            let _guard = primary.enter().unwrap();
            assert_eq!(mock.current_primary_owner(), Some(primary.owner()));
        }
        // A failed CUDA pop leaves real currentness unknown, so the metadata
        // intentionally retains the owner rather than claiming restoration.
        assert_eq!(mock.current_primary_owner(), Some(primary.owner()));
    }

    #[test]
    fn primary_owner_observation_is_independent_per_thread() {
        use std::sync::{Barrier, mpsc};

        let mock = Arc::new(Mock::default());
        let device = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap();
        let first = device.retain_primary_context().unwrap();
        let second = device.retain_primary_context().unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let (send, receive) = mpsc::channel();
        let mut workers = Vec::new();
        for primary in [first.clone(), second.clone()] {
            let barrier = barrier.clone();
            let send = send.clone();
            workers.push(std::thread::spawn(move || {
                let owner = primary.owner();
                let guard = primary.enter().unwrap();
                send.send((std::thread::current().id(), owner)).unwrap();
                barrier.wait();
                barrier.wait();
                drop(guard);
            }));
        }
        drop(send);
        let observed = [receive.recv().unwrap(), receive.recv().unwrap()];
        barrier.wait();
        for (thread, owner) in &observed {
            assert_eq!(mock.current_primary_owner_on(*thread), Some(*owner));
        }
        assert_eq!(mock.current_primary_owner(), None);
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn mock_primary_memory_is_owner_scoped_despite_colliding_raw_handles() {
        let mock = Arc::new(Mock::default());
        let device = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap();
        let first = device.retain_primary_context().unwrap();
        let second = device.retain_primary_context().unwrap();
        let first_buffer = first.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let second_buffer = second.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        assert_eq!(first_buffer.ptr, second_buffer.ptr);
        first_buffer.copy_from(1, &[1, 2, 3]).unwrap();
        second_buffer.copy_from(1, &[9, 8, 7]).unwrap();
        let mut first_bytes = [0; 3];
        let mut second_bytes = [0; 3];
        first_buffer.copy_to(1, &mut first_bytes).unwrap();
        second_buffer.copy_to(1, &mut second_bytes).unwrap();
        assert_eq!(first_bytes, [1, 2, 3]);
        assert_eq!(second_bytes, [9, 8, 7]);
        let descriptor = mock
            .allocation_descriptor(first.owner(), first_buffer.ptr)
            .unwrap();
        mock.write_allocation(first.owner(), descriptor, 4, &[6])
            .unwrap();
        assert_eq!(
            mock.allocation_snapshot(first.owner(), descriptor).unwrap()[1..4],
            [1, 2, 3]
        );
    }

    #[test]
    fn mock_primary_memory_mutates_sync_and_async_copies_at_submission() {
        let mock = Arc::new(Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let destination = primary.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let source = primary.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let stream = primary.stream().unwrap();
        let input = primary
            .allocate_pinned(NonZeroUsize::new(8).unwrap())
            .unwrap();
        let output = primary
            .allocate_pinned(NonZeroUsize::new(8).unwrap())
            .unwrap();
        input.write(1, &[4, 5, 6]).unwrap();
        let mut h2d = destination
            .copy_from_pinned_async(2, &input, 1, 3, &stream)
            .unwrap();
        let mut observed = [0; 3];
        destination.copy_to(2, &mut observed).unwrap();
        assert_eq!(observed, [4, 5, 6]);
        h2d.wait().unwrap();
        let mut d2h = destination
            .copy_to_pinned_async(2, &output, 3, 3, &stream)
            .unwrap();
        let mut host_bytes = [0; 3];
        output.read(3, &mut host_bytes).unwrap();
        assert_eq!(host_bytes, [4, 5, 6]);
        d2h.wait().unwrap();
        source.copy_from(1, &[7, 8]).unwrap();
        let mut d2d = destination
            .copy_from_device_async(5, &source, 1, 2, &stream)
            .unwrap();
        destination.copy_to(5, &mut observed[..2]).unwrap();
        assert_eq!(&observed[..2], [7, 8]);
        d2d.wait().unwrap();
        assert!(mock.calls().contains(&"htod_async"));
        assert!(mock.calls().contains(&"dtoh_async"));
        assert!(mock.calls().contains(&"dtod_async"));
    }

    #[test]
    fn mock_peer_memory_uses_stable_pair_for_colliding_contexts() {
        let mock = Arc::new(Mock::default());
        let device = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap();
        let source = device.retain_primary_context().unwrap();
        let destination = device.retain_primary_context().unwrap();
        let source_pool = source.allocator();
        let destination_pool = destination.allocator();
        let source_lease = source_pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let destination_lease = destination_pool
            .allocate(NonZeroUsize::new(8).unwrap())
            .unwrap();
        let source_view = source_lease.view().unwrap();
        let destination_view = destination_lease.view().unwrap();
        assert_eq!(
            source_view.device_ptr().unwrap(),
            destination_view.device_ptr().unwrap()
        );
        source_view.copy_from(1, &[3, 4, 5]).unwrap();
        let forward = source.peer_access_to(&destination).unwrap();
        let destination_stream = destination.stream().unwrap();
        let mut transfer = destination_lease
            .copy_from_peer_async(2, &forward, &source_lease, 1, 3, &destination_stream)
            .unwrap();
        transfer.wait().unwrap();
        let mut bytes = [0; 3];
        destination_view.copy_to(2, &mut bytes).unwrap();
        assert_eq!(bytes, [3, 4, 5]);

        destination_view.copy_from(1, &[8, 7]).unwrap();
        let reverse = destination.peer_access_to(&source).unwrap();
        let source_stream = source.stream().unwrap();
        let mut reverse_transfer = source_lease
            .copy_from_peer_async(4, &reverse, &destination_lease, 1, 2, &source_stream)
            .unwrap();
        reverse_transfer.wait().unwrap();
        source_view.copy_to(4, &mut bytes[..2]).unwrap();
        assert_eq!(&bytes[..2], [8, 7]);
    }

    #[test]
    fn mock_primary_memory_rejects_invalid_lifecycle_and_cleans_before_unregister() {
        let mock = Arc::new(Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let owner = primary.owner();
        mock.set_push_result(CUDA_SUCCESS);
        mock.fail_alloc.store(true, Ordering::Release);
        assert!(primary.allocate(NonZeroUsize::new(4).unwrap()).is_err());
        assert_eq!(mock.live_allocation_count(owner), 0);
        mock.fail_alloc.store(false, Ordering::Release);
        let buffer = primary.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        let descriptor = mock.allocation_descriptor(owner, buffer.ptr).unwrap();
        assert!(matches!(
            buffer.copy_from(3, &[1, 2]),
            Err(CudaError::InvalidArgument(_))
        ));
        buffer.close().unwrap();
        assert_eq!(mock.allocation_snapshot(owner, descriptor), None);
        let guard = primary.enter().unwrap();
        let dispatch = primary.0.driver.0.dispatch.as_ref();
        assert_eq!(dispatch.mem_free(buffer.ptr), Mock::INVALID_MEMORY);
        assert_eq!(
            dispatch.memcpy_htod(buffer.ptr, [1].as_ptr().cast(), 1),
            Mock::INVALID_MEMORY
        );
        assert_eq!(
            dispatch.memcpy_htod(u64::MAX, [1].as_ptr().cast(), 1),
            Mock::INVALID_MEMORY
        );
        drop(guard);
        assert_eq!(mock.live_allocation_count(owner), 0);
        drop(buffer);
        drop(primary);
        assert_eq!(mock.registered_primary_owner(owner.identity), None);
    }

    #[test]
    fn mock_primary_memory_is_independent_for_concurrent_owners() {
        use std::sync::{Barrier, mpsc};

        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let (send, receive) = mpsc::channel();
        let mut joins = Vec::new();
        for bytes in [[1, 2], [8, 9]] {
            let driver = driver.clone();
            let barrier = barrier.clone();
            let send = send.clone();
            joins.push(std::thread::spawn(move || {
                let primary = driver
                    .device(DeviceId(0))
                    .unwrap()
                    .retain_primary_context()
                    .unwrap();
                let buffer = primary.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
                buffer.copy_from(1, &bytes).unwrap();
                barrier.wait();
                let mut observed = [0; 2];
                buffer.copy_to(1, &mut observed).unwrap();
                send.send((buffer.ptr, bytes, observed)).unwrap();
            }));
        }
        drop(send);
        let results: Vec<_> = receive.iter().collect();
        for join in joins {
            join.join().unwrap();
        }
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, results[1].0);
        for (_, expected, observed) in results {
            assert_eq!(expected, observed);
        }
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
    fn primary_pool_stats_are_handle_scoped_and_track_transitions() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let pool = primary.allocator();
        let clone = pool.clone();
        let other = primary.allocator();
        let base = pool.stats();
        assert_eq!(base.logical_leased_bytes, 0);
        assert_eq!(base.cached_blocks, 0);
        assert_eq!(base.pool_id, clone.stats().pool_id);
        assert_ne!(base.pool_id, other.stats().pool_id);
        let lease = pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let used = clone.stats();
        assert_eq!(used.logical_leased_bytes, 8);
        assert_eq!(used.peak_in_use_blocks, 1);
        drop(lease);
        let cached = pool.stats();
        assert_eq!(cached.logical_leased_bytes, 0);
        assert_eq!(cached.cached_blocks, 1);
        assert_eq!(cached.cached_bytes, 256);
        assert_eq!(cached.peak_in_use_bytes, 8);
        pool.trim().unwrap();
        let trimmed = pool.stats();
        assert_eq!(trimmed.cached_blocks, 0);
        assert_eq!(trimmed.cached_bytes, 0);
        assert_eq!(trimmed.peak_in_use_bytes, 8);
    }

    #[test]
    fn primary_pool_stats_transition_matrix_uses_the_exact_handle() {
        let mock = Arc::new(Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let pool = primary.allocator();
        let stream = primary.stream().unwrap();
        assert_eq!(pool.stats().logical_leased_bytes, 0);
        let lease = pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let generation = lease.generation;
        let fence = Arc::new(primary.event_fence().unwrap());
        fence.record(&stream).unwrap();
        lease.attach_fence(fence).unwrap();
        drop(lease);
        let deferred = pool.stats();
        assert_eq!(
            (
                deferred.deferred_blocks,
                deferred.deferred_bytes,
                deferred.cached_blocks
            ),
            (1, 256, 0)
        );
        pool.trim().unwrap();
        assert_eq!(pool.stats().deferred_blocks, 1);
        assert_eq!(pool.collect_deferred().unwrap(), 0);
        mock.set_event_ready(true);
        assert_eq!(pool.collect_deferred().unwrap(), 1);
        let cached = pool.stats();
        assert_eq!(
            (
                cached.cached_blocks,
                cached.cached_bytes,
                cached.deferred_blocks
            ),
            (1, 256, 0)
        );
        let reuse = pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        assert!(reuse.generation > generation);
        assert_eq!(pool.stats().logical_leased_bytes, 8);
        drop(reuse);
        pool.trim().unwrap();
        let final_stats = pool.stats();
        assert_eq!(
            (final_stats.logical_leased_bytes, final_stats.cached_bytes),
            (0, 0)
        );
        assert_eq!(final_stats.peak_in_use_bytes, 8);
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
    fn peer_copy_reports_its_own_missing_symbol_before_submission() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let device = driver.device(DeviceId(0)).unwrap();
        let source = device.retain_primary_context().unwrap();
        let destination = device.retain_primary_context().unwrap();
        let peer = source.peer_access_to(&destination).unwrap();
        let src = source
            .allocator()
            .allocate(NonZeroUsize::new(8).unwrap())
            .unwrap();
        let dst = destination
            .allocator()
            .allocate(NonZeroUsize::new(8).unwrap())
            .unwrap();
        let stream = destination.stream().unwrap();
        mock.set_peer_async_supported(false);
        assert!(matches!(
            dst.copy_from_peer_async(0, &peer, &src, 0, 8, &stream),
            Err(CudaError::MissingSymbol("cuMemcpyPeerAsync"))
        ));
        assert!(!mock.calls().contains(&"peer_copy"));
    }

    #[test]
    fn native_peer_resolver_omits_each_symbol_independently() {
        for missing in [
            b"cuDeviceCanAccessPeer\0".as_slice(),
            b"cuCtxEnablePeerAccess\0".as_slice(),
            b"cuCtxDisablePeerAccess\0".as_slice(),
            b"cuMemcpyPeerAsync\0".as_slice(),
        ] {
            let table = NativePeerTable::resolve(|name| {
                if name == missing {
                    None
                } else {
                    Some(std::ptr::dangling_mut::<c_void>())
                }
            });
            assert_eq!(
                table.can_access.is_none(),
                missing == b"cuDeviceCanAccessPeer\0"
            );
            assert_eq!(
                table.enable.is_none(),
                missing == b"cuCtxEnablePeerAccess\0"
            );
            assert_eq!(
                table.disable.is_none(),
                missing == b"cuCtxDisablePeerAccess\0"
            );
            assert_eq!(
                table.copy_async.is_none(),
                missing == b"cuMemcpyPeerAsync\0"
            );
        }
    }

    #[test]
    fn peer_copy_preflight_and_event_failure_never_reuse_uncertain_blocks() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let dev = driver.device(DeviceId(0)).unwrap();
        let a = dev.retain_primary_context().unwrap();
        let b = dev.retain_primary_context().unwrap();
        let peer = a.peer_access_to(&b).unwrap();
        let ap = a.allocator();
        let bp = b.allocator();
        let src = ap.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let dst = bp.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let stream = b.stream().unwrap();
        assert!(matches!(
            dst.copy_from_peer_async(0, &peer, &src, 0, 0, &stream),
            Err(CudaError::InvalidArgument(_))
        ));
        assert!(matches!(
            dst.copy_from_peer_async(1, &peer, &src, 0, 8, &stream),
            Err(CudaError::InvalidArgument(_))
        ));
        mock.set_peer_result(2);
        assert!(matches!(
            dst.copy_from_peer_async(0, &peer, &src, 0, 8, &stream),
            Err(CudaError::Driver { .. })
        ));
        assert_eq!(ap.deferred_blocks(), 0);
        assert_eq!(bp.deferred_blocks(), 0);
        mock.set_peer_result(0);
        mock.set_event_record_result(2);
        assert!(matches!(
            dst.copy_from_peer_async(0, &peer, &src, 0, 8, &stream),
            Err(CudaError::Driver { .. })
        ));
        assert!(mock.calls().contains(&"stream_sync"));
        mock.set_stream_sync_result(2);
        assert!(
            dst.copy_from_peer_async(0, &peer, &src, 0, 8, &stream)
                .is_err()
        );
        drop(src);
        drop(dst);
        assert_eq!(ap.cached_bytes(), 0);
        assert_eq!(bp.cached_bytes(), 0);
    }

    #[test]
    fn peer_access_lifecycle_statuses_and_pending_pressure_are_safe() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let dev = driver.device(DeviceId(0)).unwrap();
        let a = dev.retain_primary_context().unwrap();
        let b = dev.retain_primary_context().unwrap();
        mock.set_peer_enable_result(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
        let peer = a.peer_access_to(&b).unwrap();
        mock.set_peer_disable_result(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
        peer.close().unwrap();
        drop(peer);
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|&&x| x == "peer_disable").count(), 1);
        assert!(calls.contains(&"ctx_push") && calls.contains(&"ctx_pop"));
        mock.set_peer_enable_result(2);
        assert!(matches!(
            a.peer_access_to(&b),
            Err(CudaError::Driver { .. })
        ));
        mock.set_peer_enable_result(0);
        mock.set_peer_disable_result(0);
        let peer = a.peer_access_to(&b).unwrap();
        let ap = a.allocator();
        let bp = b.allocator();
        let src = ap.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let dst = bp.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        let stream = b.stream().unwrap();
        let transfer = dst
            .copy_from_peer_async(0, &peer, &src, 0, 8, &stream)
            .unwrap();
        drop(transfer);
        drop(src);
        drop(dst);
        ap.trim().unwrap();
        bp.trim().unwrap();
        assert_eq!(ap.deferred_blocks(), 1);
        assert_eq!(bp.deferred_blocks(), 1);
        assert_eq!(ap.cached_bytes(), 0);
        assert_eq!(bp.cached_bytes(), 0);
        mock.set_event_ready(true);
        assert_eq!(ap.wait_deferred().unwrap(), 1);
        assert_eq!(bp.wait_deferred().unwrap(), 1);
        assert_eq!(ap.cached_bytes(), 256);
        assert_eq!(bp.cached_bytes(), 256);
    }

    #[test]
    fn concurrent_peer_pairs_keep_owner_and_pool_state_isolated() {
        use std::sync::Barrier;
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let gate = Arc::new(Barrier::new(2));
        let mut joins = Vec::new();
        for _ in 0..2 {
            let driver = driver.clone();
            let gate = gate.clone();
            joins.push(std::thread::spawn(move || {
                let dev = driver.device(DeviceId(0)).unwrap();
                let source = dev.retain_primary_context().unwrap();
                let destination = dev.retain_primary_context().unwrap();
                let source_id = source.identity();
                let destination_id = destination.identity();
                assert_ne!(source_id, destination_id);
                let peer = source.peer_access_to(&destination).unwrap();
                let sp = source.allocator();
                let dp = destination.allocator();
                let src = sp.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
                let dst = dp.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
                let stream = destination.stream().unwrap();
                gate.wait();
                let mut transfer = dst
                    .copy_from_peer_async(0, &peer, &src, 0, 8, &stream)
                    .unwrap();
                transfer.wait().unwrap();
                drop(transfer);
                drop(src);
                drop(dst);
                assert_eq!(sp.wait_deferred().unwrap(), 1);
                assert_eq!(dp.wait_deferred().unwrap(), 1);
                (source_id, destination_id)
            }));
        }
        let pairs = joins
            .into_iter()
            .map(|join| join.join().unwrap())
            .collect::<Vec<_>>();
        assert_ne!(pairs[0].0, pairs[1].0);
        assert_ne!(pairs[0].1, pairs[1].1);
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|&&x| x == "peer_copy").count(), 2);
        assert_eq!(calls.iter().filter(|&&x| x == "peer_disable").count(), 2);
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
    fn graph_capture_rejects_successful_null_driver_outputs_without_raii_cleanup() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let stream = primary.stream().unwrap();
        mock.capture_null_graph.store(true, Ordering::Release);
        assert!(matches!(
            stream.begin_capture().unwrap().finish(),
            Err(CudaError::InvalidArgument("capture returned null graph"))
        ));
        let calls = mock.calls();
        assert!(calls.contains(&"capture_end"));
        assert!(!calls.contains(&"graph_destroy"));
        mock.capture_null_graph.store(false, Ordering::Release);
        let graph = stream.begin_capture().unwrap().finish().unwrap();
        mock.instantiate_null_exec.store(true, Ordering::Release);
        assert!(matches!(
            graph.instantiate(),
            Err(CudaError::InvalidArgument(
                "instantiate returned null graph exec"
            ))
        ));
        let calls = mock.calls();
        assert!(!calls.contains(&"graph_launch"));
        assert!(!calls.contains(&"graph_exec_destroy"));
    }

    #[test]
    fn module_and_function_reject_successful_null_driver_outputs() {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let ptx = CString::new(".version 7.0").unwrap();
        mock.null_module.store(true, Ordering::Release);
        assert!(matches!(
            primary.module_from_ptx(&ptx),
            Err(CudaError::InvalidArgument(
                "module load returned null handle"
            ))
        ));
        assert!(!mock.calls().contains(&"module_unload"));
        mock.null_module.store(false, Ordering::Release);
        let module = primary.module_from_ptx(&ptx).unwrap();
        mock.null_function.store(true, Ordering::Release);
        assert!(matches!(
            module.function(&CString::new("k").unwrap()),
            Err(CudaError::InvalidArgument(
                "function lookup returned null handle"
            ))
        ));
        let calls = mock.calls();
        assert!(!calls.contains(&"launch"));
        assert_eq!(calls.iter().filter(|&&x| x == "module_unload").count(), 0);
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

    #[test]
    fn mock_link_state_raii_cleans_failed_and_retried_links() {
        let mock = Mock::default();
        let input = LinkInput::ptx("empty.ptx", b".version 7.0".to_vec()).unwrap();
        mock.set_link_create_result(2);
        assert!(LinkState::create(&mock).is_err());
        assert_eq!(mock.live_link_state_count(), 0);
        mock.set_link_create_result(CUDA_SUCCESS);

        mock.set_link_add_data_result(2);
        {
            let link = LinkState::create(&mock).unwrap();
            assert!(link.add(&input).is_err());
            assert_eq!(mock.live_link_state_count(), 1);
        }
        assert_eq!(mock.live_link_state_count(), 0);

        mock.set_link_add_data_result(CUDA_SUCCESS);
        mock.set_link_complete_result(2);
        {
            let link = LinkState::create(&mock).unwrap();
            assert!(link.complete().is_err());
        }
        assert_eq!(mock.live_link_state_count(), 0);
        mock.set_link_complete_result(CUDA_SUCCESS);

        mock.set_link_destroy_result(2);
        {
            let link = LinkState::create(&mock).unwrap();
            link.add(&input).unwrap();
        }
        assert_eq!(mock.live_link_state_count(), 0);
        mock.set_link_destroy_result(CUDA_SUCCESS);

        {
            let link = LinkState::create(&mock).unwrap();
            link.add(&input).unwrap();
            assert_eq!(link.complete().unwrap(), 0x77usize as *mut c_void);
        }
        assert_eq!(mock.live_link_state_count(), 0);
        assert!(matches!(
            mock.calls().as_slice(),
            [
                "link_create",
                "link_create",
                "link_add_data",
                "link_destroy",
                "link_create",
                "link_complete",
                "link_destroy",
                "link_create",
                "link_add_data",
                "link_destroy",
                "link_create",
                "link_add_data",
                "link_complete",
                "link_destroy"
            ]
        ));
    }

    #[test]
    fn linked_module_loader_orders_cleanup_and_retries() {
        let mock = Arc::new(Mock::default());
        let context = context(&mock);
        let input = LinkInput::ptx("linked.ptx", b".version 7.0".to_vec()).unwrap();
        let second = LinkInput::ptx("second.ptx", b".target sm_80".to_vec()).unwrap();
        let identity = linked_module_identity(&[input.clone(), second.clone()]).unwrap();
        assert_eq!(
            identity,
            linked_module_identity(&[input.clone(), second.clone()]).unwrap()
        );
        assert_ne!(
            identity,
            linked_module_identity(&[second.clone(), input.clone()]).unwrap()
        );
        assert_ne!(
            identity,
            linked_module_identity(&[
                LinkInput::ptx("linked.ptx", b".version 8.0".to_vec()).unwrap(),
                second.clone(),
            ])
            .unwrap()
        );
        assert_eq!(
            LinkedModuleIdentity::from_cache_key(identity.cache_key()).unwrap(),
            identity
        );
        assert!(LinkedModuleIdentity::from_cache_key("cuda-link-v2:0000000000000000").is_err());
        assert!(LinkedModuleIdentity::from_cache_key("cuda-link-v1:not-a-fingerprint").is_err());
        assert!(LinkInput::ptx("", b".version 7.0".to_vec()).is_err());
        let duplicate = input.clone();
        let calls_before_invalid = mock.calls().len();
        assert!(context.module_from_link_inputs(&[input.clone(), duplicate]).is_err());
        assert_eq!(mock.calls().len(), calls_before_invalid);
        mock.set_link_add_data_result(2);
        assert!(context.module_from_link_inputs(&[input.clone()]).is_err());
        assert_eq!(mock.live_link_state_count(), 0);
        assert!(!mock.calls().contains(&"module_load"));

        mock.set_link_add_data_result(CUDA_SUCCESS);
        mock.set_link_complete_result(2);
        assert!(context.module_from_link_inputs(&[input.clone()]).is_err());
        assert_eq!(mock.live_link_state_count(), 0);
        assert!(!mock.calls().contains(&"module_load"));

        mock.set_link_complete_result(CUDA_SUCCESS);
        mock.set_module_result(2);
        assert!(context.module_from_link_inputs(&[input.clone()]).is_err());
        assert_eq!(mock.live_link_state_count(), 0);

        mock.set_module_result(CUDA_SUCCESS);
        mock.set_link_destroy_result(2);
        assert!(context.module_from_link_inputs(&[input.clone()]).is_err());
        assert_eq!(mock.live_link_state_count(), 0);
        mock.set_link_destroy_result(CUDA_SUCCESS);
        let module = context.module_from_link_inputs(&[input]).unwrap();
        assert_eq!(mock.live_link_state_count(), 0);
        drop(module);
        let calls = mock.calls();
        let ordered = calls
            .iter()
            .filter(|call| {
                matches!(
                    **call,
                    "link_create" | "link_add_data" | "link_complete" | "module_load" | "link_destroy"
                )
            })
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            [
                "link_create",
                "link_add_data",
                "link_destroy",
                "link_create",
                "link_add_data",
                "link_complete",
                "link_destroy",
                "link_create",
                "link_add_data",
                "link_complete",
                "module_load",
                "link_destroy",
                "link_create",
                "link_add_data",
                "link_complete",
                "module_load",
                "link_destroy",
                "link_create",
                "link_add_data",
                "link_complete",
                "module_load",
                "link_destroy"
            ]
        );
        assert!(calls.iter().any(|call| *call == "module_unload"));
    }

    #[test]
    fn linked_library_inputs_have_distinct_ordered_identity_and_driver_kind() {
        let mock = Arc::new(Mock::default());
        let context = context(&mock);
        let ptx = LinkInput::ptx("kernel.ptx", b".version 7.0".to_vec()).unwrap();
        let library = LinkInput::library("math.a", b"immutable-library".to_vec()).unwrap();
        assert_ne!(
            linked_module_identity(&[ptx.clone(), library.clone()]).unwrap(),
            linked_module_identity(&[library.clone(), ptx.clone()]).unwrap()
        );
        assert_ne!(
            linked_module_identity(&[ptx.clone()]).unwrap(),
            linked_module_identity(&[library.clone()]).unwrap()
        );
        let module = context
            .module_from_link_inputs(&[ptx, library])
            .unwrap();
        assert_eq!(mock.link_input_types(), [CU_JIT_INPUT_PTX, CU_JIT_INPUT_LIBRARY]);
        assert_eq!(mock.live_link_state_count(), 0);
        drop(module);
        assert!(mock.calls().windows(5).any(|calls| {
            calls
                == [
                    "link_create",
                    "link_add_data",
                    "link_add_data",
                    "link_complete",
                    "module_load"
                ]
        }));
    }

    #[test]
    fn linked_nvvm_inputs_are_ordered_and_distinct_from_library_bytes() {
        let mock = Arc::new(Mock::default());
        let context = context(&mock);
        let nvvm = LinkInput::nvvm("math.bc", b"bitcode".to_vec()).unwrap();
        let library = LinkInput::library("math.bc", b"bitcode".to_vec()).unwrap();
        assert_ne!(linked_module_identity(&[nvvm.clone()]).unwrap(), linked_module_identity(&[library]).unwrap());
        assert!(LinkInput::nvvm("", b"bitcode".to_vec()).is_err());
        let module = context.module_from_link_inputs(&[nvvm]).unwrap();
        assert_eq!(mock.link_input_types(), [CU_JIT_INPUT_NVVM]);
        drop(module);
    }
}
