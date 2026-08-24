//! Narrow dynamic-library boundary for the evolving WebGPU native C ABI.
//!
//! WebGPU's C headers have incompatible callback/future descriptor revisions
//! across Dawn and `wgpu-native`. Calling an unversioned symbol with the wrong
//! layout is undefined behavior. This milestone therefore probes libraries and
//! required symbols dynamically, then returns a structured ABI error before it
//! creates an instance or registers a callback. The safe injected dispatch and
//! mock remain fully executable; enabling a native dispatch requires pinning a
//! generated header/version and implementing ownership-scoped callback state.
use super::{
    WebGpuError,
    dispatch::{
        CopyRegion, Dispatch, LaunchGeometry, RawAdapter, RawBuffer, RawCommand, RawDevice,
        RawInstance, RawPipeline, RawQueue, RawShader, WebGpuAdapterInfo,
    },
};

#[derive(Debug)]
pub(super) struct NativeDispatch;

impl NativeDispatch {
    pub(super) fn load() -> Result<Self, WebGpuError> {
        Self::load_candidates(default_candidates())
    }

    fn load_candidates(candidates: &[&str]) -> Result<Self, WebGpuError> {
        let mut tried = Vec::new();
        for candidate in candidates {
            tried.push((*candidate).to_string());
            let Ok(library) = DynamicLibrary::open(candidate) else {
                continue;
            };
            for symbol in [
                "wgpuCreateInstance",
                "wgpuDeviceGetQueue",
                "wgpuDeviceCreateBuffer",
                "wgpuDeviceCreateShaderModule",
                "wgpuDeviceCreateComputePipeline",
                "wgpuQueueSubmit",
            ] {
                if !library.has_symbol(symbol) {
                    return Err(WebGpuError::MissingSymbol(symbol));
                }
            }
            for alternatives in [
                (
                    "wgpuInstanceRequestAdapter{,2}",
                    ["wgpuInstanceRequestAdapter", "wgpuInstanceRequestAdapter2"],
                ),
                (
                    "wgpuAdapterRequestDevice{,2}",
                    ["wgpuAdapterRequestDevice", "wgpuAdapterRequestDevice2"],
                ),
            ] {
                if !alternatives
                    .1
                    .iter()
                    .any(|symbol| library.has_symbol(symbol))
                {
                    return Err(WebGpuError::MissingSymbol(alternatives.0));
                }
            }
            let flavor = if library.has_symbol("wgpuGetVersion") {
                "wgpu-native"
            } else if library.has_symbol("wgpuInstanceWaitAny") {
                "Dawn"
            } else {
                "unknown WebGPU provider"
            };
            return Err(WebGpuError::NativeAbiUnsupported {
                detail: format!(
                    "found {flavor}, but no checked-in header/version pins its callback and future descriptor ABI"
                ),
            });
        }
        Err(WebGpuError::LibraryUnavailable { tried })
    }
}

fn default_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "macos")]
    {
        &[
            "libwgpu_native.dylib",
            "libwebgpu_dawn.dylib",
            "webgpu_dawn",
        ]
    }
    #[cfg(target_os = "linux")]
    {
        &[
            "libwgpu_native.so",
            "libwebgpu_dawn.so",
            "libdawn_native.so",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        &["wgpu_native.dll", "webgpu_dawn.dll", "dawn_native.dll"]
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

impl Dispatch for NativeDispatch {
    fn instance_create(&self) -> Result<RawInstance, WebGpuError> {
        Err(native_unavailable())
    }
    fn instance_release(&self, _instance: RawInstance) {}
    fn adapters(&self, _instance: RawInstance) -> Result<Vec<RawAdapter>, WebGpuError> {
        Err(native_unavailable())
    }
    fn adapter_info(&self, _adapter: RawAdapter) -> Result<WebGpuAdapterInfo, WebGpuError> {
        Err(native_unavailable())
    }
    fn adapter_release(&self, _adapter: RawAdapter) {}
    fn device_create(&self, _adapter: RawAdapter, _owner: u64) -> Result<RawDevice, WebGpuError> {
        Err(native_unavailable())
    }
    fn device_release(&self, _device: RawDevice, _owner: u64) {}
    fn queue_create(&self, _device: RawDevice, _owner: u64) -> Result<RawQueue, WebGpuError> {
        Err(native_unavailable())
    }
    fn queue_release(&self, _queue: RawQueue, _owner: u64) {}
    fn buffer_create(
        &self,
        _device: RawDevice,
        _physical_bytes: usize,
        _owner: u64,
    ) -> Result<RawBuffer, WebGpuError> {
        Err(native_unavailable())
    }
    fn buffer_release(&self, _buffer: RawBuffer, _owner: u64) {}
    fn buffer_write(
        &self,
        _queue: RawQueue,
        _buffer: RawBuffer,
        _offset: usize,
        _bytes: &[u8],
        _owner: u64,
    ) -> Result<(), WebGpuError> {
        Err(native_unavailable())
    }
    fn buffer_read(
        &self,
        _buffer: RawBuffer,
        _offset: usize,
        _bytes: &mut [u8],
        _owner: u64,
    ) -> Result<(), WebGpuError> {
        Err(native_unavailable())
    }
    fn buffer_copy(
        &self,
        _queue: RawQueue,
        _src: RawBuffer,
        _dst: RawBuffer,
        _region: CopyRegion,
        _owner: u64,
    ) -> Result<RawCommand, WebGpuError> {
        Err(native_unavailable())
    }
    fn shader_create(
        &self,
        _device: RawDevice,
        _source: &str,
        _owner: u64,
    ) -> Result<RawShader, WebGpuError> {
        Err(native_unavailable())
    }
    fn shader_release(&self, _shader: RawShader, _owner: u64) {}
    fn pipeline_create(
        &self,
        _device: RawDevice,
        _shader: RawShader,
        _entry: &str,
        _storage_bindings: usize,
        _owner: u64,
    ) -> Result<RawPipeline, WebGpuError> {
        Err(native_unavailable())
    }
    fn pipeline_release(&self, _pipeline: RawPipeline, _owner: u64) {}
    fn launch(
        &self,
        _queue: RawQueue,
        _pipeline: RawPipeline,
        _buffers: &[RawBuffer],
        _geometry: LaunchGeometry,
        _owner: u64,
    ) -> Result<RawCommand, WebGpuError> {
        Err(native_unavailable())
    }
    fn command_query(&self, _command: RawCommand, _owner: u64) -> Result<bool, WebGpuError> {
        Err(native_unavailable())
    }
    fn command_wait(&self, _command: RawCommand, _owner: u64) -> Result<(), WebGpuError> {
        Err(native_unavailable())
    }
    fn command_release(&self, _command: RawCommand, _owner: u64) {}
}

fn native_unavailable() -> WebGpuError {
    WebGpuError::NativeAbiUnsupported {
        detail: "native dispatch was not constructed from a pinned C ABI".into(),
    }
}

#[cfg(unix)]
struct DynamicLibrary {
    handle: usize,
}

#[cfg(unix)]
impl DynamicLibrary {
    fn open(path: &str) -> Result<Self, ()> {
        use std::ffi::{CString, c_char, c_int, c_void};
        unsafe extern "C" {
            fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        }
        const RTLD_NOW: c_int = 2;
        const RTLD_LOCAL: c_int = 4;
        let path = CString::new(path).map_err(|_| ())?;
        // SAFETY: `path` is NUL-terminated and lives through this call.
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            Err(())
        } else {
            Ok(Self {
                handle: handle as usize,
            })
        }
    }

    fn has_symbol(&self, name: &str) -> bool {
        use std::ffi::{CString, c_char, c_void};
        unsafe extern "C" {
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        }
        let Ok(name) = CString::new(name) else {
            return false;
        };
        // SAFETY: the handle is live and `name` is NUL-terminated.
        !unsafe { dlsym(self.handle as *mut c_void, name.as_ptr()) }.is_null()
    }
}

#[cfg(unix)]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        use std::ffi::{c_int, c_void};
        unsafe extern "C" {
            fn dlclose(handle: *mut c_void) -> c_int;
        }
        // SAFETY: this handle came from one successful `dlopen` and is closed once.
        let _ = unsafe { dlclose(self.handle as *mut c_void) };
    }
}

#[cfg(windows)]
struct DynamicLibrary {
    handle: usize,
}

#[cfg(windows)]
impl DynamicLibrary {
    fn open(path: &str) -> Result<Self, ()> {
        use std::ffi::{CString, c_char, c_void};
        unsafe extern "system" {
            fn LoadLibraryA(path: *const c_char) -> *mut c_void;
        }
        let path = CString::new(path).map_err(|_| ())?;
        // SAFETY: `path` is NUL-terminated and lives through this call.
        let handle = unsafe { LoadLibraryA(path.as_ptr()) };
        if handle.is_null() {
            Err(())
        } else {
            Ok(Self {
                handle: handle as usize,
            })
        }
    }

    fn has_symbol(&self, name: &str) -> bool {
        use std::ffi::{CString, c_char, c_void};
        unsafe extern "system" {
            fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        }
        let Ok(name) = CString::new(name) else {
            return false;
        };
        // SAFETY: the module is live and `name` is NUL-terminated.
        !unsafe { GetProcAddress(self.handle as *mut c_void, name.as_ptr()) }.is_null()
    }
}

#[cfg(windows)]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        use std::ffi::{c_int, c_void};
        unsafe extern "system" {
            fn FreeLibrary(module: *mut c_void) -> c_int;
        }
        // SAFETY: this module came from one successful `LoadLibraryA` and is freed once.
        let _ = unsafe { FreeLibrary(self.handle as *mut c_void) };
    }
}

#[cfg(not(any(unix, windows)))]
struct DynamicLibrary;

#[cfg(not(any(unix, windows)))]
impl DynamicLibrary {
    fn open(_path: &str) -> Result<Self, ()> {
        Err(())
    }
    fn has_symbol(&self, _name: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_is_structured() {
        let error =
            NativeDispatch::load_candidates(&["rustgrad-webgpu-library-that-does-not-exist"])
                .unwrap_err();
        assert!(matches!(
            error,
            WebGpuError::LibraryUnavailable { tried }
                if tried == ["rustgrad-webgpu-library-that-does-not-exist"]
        ));
    }
}
