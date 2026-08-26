//! Concrete runtime resources with backend-specific capability boundaries.
pub mod file;
pub mod mapped;
pub mod metal;
pub mod null;
pub mod opencl;
pub mod webgpu;
