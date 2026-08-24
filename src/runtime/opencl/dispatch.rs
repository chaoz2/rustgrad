//! Typed substitution seam between safe resources and an OpenCL ICD.
use super::OpenClError;
use crate::UOp;
use std::{ffi::c_void, sync::Arc};

macro_rules! raw_handle {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        #[doc(hidden)]
        pub struct $name(pub usize);
        impl $name {
            pub(crate) fn from_ptr(value: *mut c_void) -> Self {
                Self(value as usize)
            }
            pub(crate) fn as_ptr(self) -> *mut c_void {
                self.0 as *mut c_void
            }
        }
    };
}
raw_handle!(RawPlatform);
raw_handle!(RawDevice);
raw_handle!(RawContext);
raw_handle!(RawQueue);
raw_handle!(RawBuffer);
raw_handle!(RawProgram);
raw_handle!(RawKernel);
raw_handle!(RawEvent);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct OpenClCapabilities {
    /// Exact `long`/`ulong` storage and arithmetic are available.
    pub int64: bool,
    /// The device advertises a double-precision OpenCL C extension.
    pub fp64: bool,
}

impl OpenClCapabilities {
    pub const CORE_32: Self = Self {
        int64: false,
        fp64: false,
    };

    pub const FULL: Self = Self {
        int64: true,
        fp64: true,
    };

    pub(crate) fn supports(self, required: Self) -> bool {
        (!required.int64 || self.int64) && (!required.fp64 || self.fp64)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub max_work_group_size: usize,
    pub capabilities: OpenClCapabilities,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildInfo {
    pub log: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct BufferCopyRegion {
    pub src_offset: usize,
    pub dst_offset: usize,
    pub bytes: usize,
}

/// Immutable semantic metadata available only to injected dispatches. Native
/// ICD execution uses generated source and ignores this hook.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct KernelSemantics {
    pub buffers: Vec<super::OpenClBufferAbi>,
    pub extent: usize,
    pub program: Arc<UOp>,
}

/// Injectable OpenCL dispatch contract. The native implementation is a thin
/// adapter over exact C ABI functions in `ffi`; tests use a byte-accurate mock.
pub trait Dispatch: Send + Sync + 'static {
    fn platforms(&self) -> Result<Vec<RawPlatform>, OpenClError>;
    fn platform_name(&self, platform: RawPlatform) -> Result<String, OpenClError>;
    fn devices(&self, platform: RawPlatform) -> Result<Vec<RawDevice>, OpenClError>;
    fn device_info(&self, device: RawDevice) -> Result<DeviceInfo, OpenClError>;

    fn context_create(&self, device: RawDevice, owner: u64) -> Result<RawContext, OpenClError>;
    fn context_release(&self, context: RawContext, owner: u64) -> Result<(), OpenClError>;
    fn queue_create(
        &self,
        context: RawContext,
        device: RawDevice,
        owner: u64,
    ) -> Result<RawQueue, OpenClError>;
    fn queue_release(&self, queue: RawQueue, owner: u64) -> Result<(), OpenClError>;
    fn queue_finish(&self, queue: RawQueue, owner: u64) -> Result<(), OpenClError>;

    fn buffer_create(
        &self,
        context: RawContext,
        bytes: usize,
        owner: u64,
    ) -> Result<RawBuffer, OpenClError>;
    fn buffer_release(&self, buffer: RawBuffer, owner: u64) -> Result<(), OpenClError>;
    fn buffer_write(
        &self,
        queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &[u8],
        owner: u64,
    ) -> Result<(), OpenClError>;
    fn buffer_read(
        &self,
        queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &mut [u8],
        owner: u64,
    ) -> Result<(), OpenClError>;
    fn buffer_copy(
        &self,
        queue: RawQueue,
        src: RawBuffer,
        dst: RawBuffer,
        region: BufferCopyRegion,
        owner: u64,
    ) -> Result<RawEvent, OpenClError>;

    fn program_create(
        &self,
        context: RawContext,
        source: &str,
        owner: u64,
    ) -> Result<RawProgram, OpenClError>;
    fn program_build(
        &self,
        program: RawProgram,
        device: RawDevice,
        options: &str,
        owner: u64,
    ) -> Result<(), OpenClError>;
    fn program_build_info(
        &self,
        program: RawProgram,
        device: RawDevice,
        owner: u64,
    ) -> Result<BuildInfo, OpenClError>;
    fn program_release(&self, program: RawProgram, owner: u64) -> Result<(), OpenClError>;
    fn kernel_create(
        &self,
        program: RawProgram,
        entry: &str,
        owner: u64,
    ) -> Result<RawKernel, OpenClError>;
    fn kernel_release(&self, kernel: RawKernel, owner: u64) -> Result<(), OpenClError>;
    fn kernel_arg_buffer(
        &self,
        kernel: RawKernel,
        index: u32,
        buffer: RawBuffer,
        owner: u64,
    ) -> Result<(), OpenClError>;
    fn kernel_arg_u64(
        &self,
        kernel: RawKernel,
        index: u32,
        value: u64,
        owner: u64,
    ) -> Result<(), OpenClError>;
    fn kernel_launch(
        &self,
        queue: RawQueue,
        kernel: RawKernel,
        global: usize,
        local: usize,
        owner: u64,
    ) -> Result<RawEvent, OpenClError>;

    fn event_query(&self, event: RawEvent, owner: u64) -> Result<bool, OpenClError>;
    fn event_wait(&self, event: RawEvent, owner: u64) -> Result<(), OpenClError>;
    fn event_release(&self, event: RawEvent, owner: u64) -> Result<(), OpenClError>;

    #[doc(hidden)]
    fn register_kernel_semantics(
        &self,
        _owner: u64,
        _kernel: RawKernel,
        _semantics: Arc<KernelSemantics>,
    ) -> Result<(), OpenClError> {
        Ok(())
    }
    #[doc(hidden)]
    fn unregister_kernel_semantics(&self, _owner: u64, _kernel: RawKernel) {}
}
