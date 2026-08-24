//! Private typed substitution boundary for native and semantic-mock WebGPU.
use super::{WebGpuError, WebGpuTransactionAbi, WgslBufferAbi};
use crate::UOp;
use std::sync::Arc;

macro_rules! raw_handle {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub(super) struct $name(pub usize);
    };
}

raw_handle!(RawInstance);
raw_handle!(RawAdapter);
raw_handle!(RawDevice);
raw_handle!(RawQueue);
raw_handle!(RawBuffer);
raw_handle!(RawShader);
raw_handle!(RawPipeline);
raw_handle!(RawCommand);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Stable backend family reported by an adapter.
pub enum WebGpuBackend {
    /// Vulkan backend.
    Vulkan,
    /// Apple Metal backend.
    Metal,
    /// Microsoft Direct3D 12 backend.
    Dx12,
    /// OpenGL or OpenGL ES backend.
    Gl,
    /// Browser-managed WebGPU backend.
    Browser,
    /// Provider-specific backend not in the stable inventory.
    Other(String),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// Limits/features that participate in rendering and pipeline identity.
pub struct WebGpuCapabilities {
    /// Maximum native buffer size in bytes.
    pub max_buffer_size: usize,
    /// Maximum storage bindings visible to one compute stage.
    pub max_storage_buffers_per_shader_stage: u32,
    /// Maximum X dimension of a compute workgroup.
    pub max_compute_workgroup_size_x: u32,
    /// Maximum workgroup count in one dispatch dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Whether timestamp queries are advertised.
    pub timestamp_query: bool,
    /// Whether WGSL `shader-f16` is advertised.
    pub shader_f16: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// Handle-free deterministic adapter information.
pub struct WebGpuAdapterInfo {
    /// Human-readable adapter name.
    pub name: String,
    /// Stable backend family.
    pub backend: WebGpuBackend,
    /// Provider-reported PCI or platform vendor identity.
    pub vendor: u32,
    /// Provider-reported PCI or platform device identity.
    pub device: u32,
    /// Human-readable driver identity included in cache keys.
    pub driver: String,
    /// Checked limits and features.
    pub capabilities: WebGpuCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CopyRegion {
    pub src_offset: usize,
    pub dst_offset: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LaunchGeometry {
    pub extent: u32,
    pub workgroups: u32,
    pub local: u32,
    pub extent_binding: usize,
    pub status_binding: Option<usize>,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct KernelSemantics {
    pub buffers: Vec<WgslBufferAbi>,
    pub extent: usize,
    pub program: Arc<UOp>,
    pub transaction: Option<WebGpuTransactionAbi>,
}

/// Raw-handle operations. This trait stays private so no native identity can
/// cross the safe facade; implementations own callback and ABI safety.
pub(super) trait Dispatch: Send + Sync + 'static {
    fn instance_create(&self) -> Result<RawInstance, WebGpuError>;
    fn instance_release(&self, instance: RawInstance);
    fn adapters(&self, instance: RawInstance) -> Result<Vec<RawAdapter>, WebGpuError>;
    fn adapter_info(&self, adapter: RawAdapter) -> Result<WebGpuAdapterInfo, WebGpuError>;
    fn adapter_release(&self, adapter: RawAdapter);
    fn device_create(&self, adapter: RawAdapter, owner: u64) -> Result<RawDevice, WebGpuError>;
    fn device_release(&self, device: RawDevice, owner: u64);
    fn queue_create(&self, device: RawDevice, owner: u64) -> Result<RawQueue, WebGpuError>;
    fn queue_release(&self, queue: RawQueue, owner: u64);

    fn buffer_create(
        &self,
        device: RawDevice,
        physical_bytes: usize,
        owner: u64,
    ) -> Result<RawBuffer, WebGpuError>;
    fn buffer_release(&self, buffer: RawBuffer, owner: u64);
    fn buffer_write(
        &self,
        queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &[u8],
        owner: u64,
    ) -> Result<(), WebGpuError>;
    fn buffer_read(
        &self,
        buffer: RawBuffer,
        offset: usize,
        bytes: &mut [u8],
        owner: u64,
    ) -> Result<(), WebGpuError>;
    fn buffer_copy(
        &self,
        queue: RawQueue,
        src: RawBuffer,
        dst: RawBuffer,
        region: CopyRegion,
        owner: u64,
    ) -> Result<RawCommand, WebGpuError>;

    fn shader_create(
        &self,
        device: RawDevice,
        source: &str,
        owner: u64,
    ) -> Result<RawShader, WebGpuError>;
    fn shader_release(&self, shader: RawShader, owner: u64);
    fn pipeline_create(
        &self,
        device: RawDevice,
        shader: RawShader,
        entry: &str,
        storage_bindings: usize,
        owner: u64,
    ) -> Result<RawPipeline, WebGpuError>;
    fn pipeline_release(&self, pipeline: RawPipeline, owner: u64);
    fn launch(
        &self,
        queue: RawQueue,
        pipeline: RawPipeline,
        buffers: &[RawBuffer],
        geometry: LaunchGeometry,
        owner: u64,
    ) -> Result<RawCommand, WebGpuError>;
    fn command_query(&self, command: RawCommand, owner: u64) -> Result<bool, WebGpuError>;
    fn command_wait(&self, command: RawCommand, owner: u64) -> Result<(), WebGpuError>;
    fn command_release(&self, command: RawCommand, owner: u64);

    fn register_kernel_semantics(
        &self,
        _owner: u64,
        _pipeline: RawPipeline,
        _semantics: Arc<KernelSemantics>,
    ) -> Result<(), WebGpuError> {
        Ok(())
    }
    fn unregister_kernel_semantics(&self, _owner: u64, _pipeline: RawPipeline) {}
}
