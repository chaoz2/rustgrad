//! Typed substitution boundary between safe resources and Objective-C Metal.
use super::{MetalBufferAbi, MetalError, MetalTransactionAbi};
use crate::{UOp, random::plan::RandomKernelPlan};
use std::sync::Arc;

macro_rules! raw_handle {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub(super) struct $name(pub usize);
    };
}

raw_handle!(RawDevice);
raw_handle!(RawQueue);
raw_handle!(RawBuffer);
raw_handle!(RawLibrary);
raw_handle!(RawPipeline);
raw_handle!(RawCommand);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// Stable device capabilities that participate in Metal source/cache identity.
pub struct MetalCapabilities {
    /// Largest native buffer accepted by the selected device.
    pub max_buffer_length: usize,
    /// Whether CPU and GPU use the same physical memory pool.
    pub unified_memory: bool,
    /// Highest reported Apple or Mac Metal GPU family.
    pub family: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// Deterministic, handle-free information for one discovered Metal device.
pub struct MetalDeviceInfo {
    /// Human-readable system device name.
    pub name: String,
    /// Stable system registry identity used only for ordering and cache keys.
    pub registry_id: u64,
    /// Capabilities observed when the device was discovered.
    pub capabilities: MetalCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CopyRegion {
    pub src_offset: usize,
    pub dst_offset: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LaunchGeometry {
    pub extent: u64,
    pub extent_index: usize,
    pub global: usize,
    pub local: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct KernelSemantics {
    pub buffers: Vec<MetalBufferAbi>,
    pub extent: usize,
    pub program: Arc<KernelSemanticProgram>,
    pub transaction: Option<MetalTransactionAbi>,
}

/// Typed immutable mock payload. Native Metal always executes rendered MSL;
/// the injected dispatch uses this only for deterministic independent tests.
#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum KernelSemanticProgram {
    UOp(Arc<UOp>),
    Random(Arc<RandomKernelPlan>),
}

/// Native and mock dispatch contract. It is private so raw handles cannot
/// escape through any safe public API.
pub(super) trait Dispatch: Send + Sync + 'static {
    fn devices(&self) -> Result<Vec<RawDevice>, MetalError>;
    fn device_info(&self, device: RawDevice) -> Result<MetalDeviceInfo, MetalError>;
    fn device_release(&self, device: RawDevice);

    fn queue_create(&self, device: RawDevice, owner: u64) -> Result<RawQueue, MetalError>;
    fn queue_release(&self, queue: RawQueue, owner: u64);

    fn buffer_create(
        &self,
        device: RawDevice,
        bytes: usize,
        owner: u64,
    ) -> Result<RawBuffer, MetalError>;
    fn buffer_release(&self, buffer: RawBuffer, owner: u64);
    fn buffer_write(
        &self,
        buffer: RawBuffer,
        offset: usize,
        bytes: &[u8],
        owner: u64,
    ) -> Result<(), MetalError>;
    fn buffer_read(
        &self,
        buffer: RawBuffer,
        offset: usize,
        bytes: &mut [u8],
        owner: u64,
    ) -> Result<(), MetalError>;
    fn buffer_copy(
        &self,
        queue: RawQueue,
        src: RawBuffer,
        dst: RawBuffer,
        region: CopyRegion,
        owner: u64,
    ) -> Result<RawCommand, MetalError>;

    fn library_compile(
        &self,
        device: RawDevice,
        source: &str,
        owner: u64,
    ) -> Result<RawLibrary, MetalError>;
    fn library_release(&self, library: RawLibrary, owner: u64);
    fn pipeline_create(
        &self,
        device: RawDevice,
        library: RawLibrary,
        entry: &str,
        owner: u64,
    ) -> Result<(RawPipeline, usize), MetalError>;
    fn pipeline_release(&self, pipeline: RawPipeline, owner: u64);
    fn launch(
        &self,
        queue: RawQueue,
        pipeline: RawPipeline,
        buffers: &[RawBuffer],
        geometry: LaunchGeometry,
        owner: u64,
    ) -> Result<RawCommand, MetalError>;

    fn command_query(&self, command: RawCommand, owner: u64) -> Result<bool, MetalError>;
    fn command_wait(&self, command: RawCommand, owner: u64) -> Result<(), MetalError>;
    fn command_release(&self, command: RawCommand, owner: u64);

    fn register_kernel_semantics(
        &self,
        _owner: u64,
        _pipeline: RawPipeline,
        _semantics: Arc<KernelSemantics>,
    ) -> Result<(), MetalError> {
        Ok(())
    }
    fn unregister_kernel_semantics(&self, _owner: u64, _pipeline: RawPipeline) {}
}
