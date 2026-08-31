//! Concrete runtime resources with backend-specific capability boundaries.
pub mod file;
pub mod mapped;
pub mod mapped_mut;
pub mod metal;
pub mod null;
pub mod opencl;
pub(crate) mod scalar_lane;
pub(crate) mod static_schedule;
pub mod webgpu;
