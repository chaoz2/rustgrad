//! Minimal documented Objective-C/Metal dynamic boundary.
//!
//! The native implementation uses only runtime selectors and opaque pointers,
//! so compiling RustGrad never requires Apple SDK headers or framework linker
//! flags. All unsafe code in this subsystem is confined to this file.
use super::{
    MetalDeviceInfo, MetalError,
    dispatch::{
        CopyRegion, Dispatch, LaunchGeometry, RawBuffer, RawCommand, RawDevice, RawLibrary,
        RawPipeline, RawQueue,
    },
};

#[cfg(target_os = "macos")]
use super::MetalCapabilities;

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::{
        ffi::{CStr, CString, c_char, c_int, c_void},
        ptr,
    };

    const RTLD_NOW: c_int = 2;
    const RTLD_LOCAL: c_int = 4;
    const MAX_DIAGNOSTIC_BYTES: usize = 64 * 1024;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MtlSize {
        width: usize,
        height: usize,
        depth: usize,
    }

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }

    struct Library {
        handle: usize,
    }

    // Dynamic-library handles may be used from any thread; Metal resource
    // thread confinement is enforced one layer above with `Rc`.
    unsafe impl Send for Library {}
    unsafe impl Sync for Library {}

    impl Library {
        fn open(path: &'static str, framework: &'static str) -> Result<Self, MetalError> {
            let path = CString::new(path).expect("static framework path");
            // SAFETY: `path` is NUL-terminated and remains alive for the call.
            let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
            if handle.is_null() {
                // SAFETY: dlerror returns either null or a process-owned C string.
                let detail = unsafe {
                    let error = dlerror();
                    if error.is_null() {
                        "unknown loader error".into()
                    } else {
                        CStr::from_ptr(error).to_string_lossy().into_owned()
                    }
                };
                return Err(MetalError::FrameworkUnavailable { framework, detail });
            }
            Ok(Self {
                handle: handle as usize,
            })
        }

        fn symbol(&self, name: &'static str) -> Result<usize, MetalError> {
            let name_c = CString::new(name).expect("static symbol");
            // SAFETY: the library is live and the symbol name is NUL-terminated.
            let value = unsafe { dlsym(self.handle as *mut c_void, name_c.as_ptr()) };
            if value.is_null() {
                Err(MetalError::MissingSymbol(name))
            } else {
                Ok(value as usize)
            }
        }

        fn optional_symbol(&self, name: &'static str) -> Option<usize> {
            let name_c = CString::new(name).expect("static symbol");
            // SAFETY: same invariant as `symbol`; null simply means absent.
            let value = unsafe { dlsym(self.handle as *mut c_void, name_c.as_ptr()) };
            (!value.is_null()).then_some(value as usize)
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: `handle` came from a successful dlopen and is closed once.
            let _ = unsafe { dlclose(self.handle as *mut c_void) };
        }
    }

    struct Objc {
        _library: Library,
        msg_send: usize,
        sel_register_name: usize,
        get_class: usize,
    }

    impl Objc {
        fn load() -> Result<Self, MetalError> {
            let library = Library::open("/usr/lib/libobjc.A.dylib", "Objective-C runtime")?;
            Ok(Self {
                msg_send: library.symbol("objc_msgSend")?,
                sel_register_name: library.symbol("sel_registerName")?,
                get_class: library.symbol("objc_getClass")?,
                _library: library,
            })
        }

        fn selector(&self, name: &'static str) -> *mut c_void {
            let name = CString::new(name).expect("static selector");
            // SAFETY: function signature is the documented Objective-C runtime ABI.
            let function: unsafe extern "C" fn(*const c_char) -> *mut c_void =
                unsafe { std::mem::transmute(self.sel_register_name) };
            // SAFETY: `name` is NUL-terminated for the duration of the call.
            unsafe { function(name.as_ptr()) }
        }

        fn class(&self, name: &'static str) -> Result<*mut c_void, MetalError> {
            let name = CString::new(name).expect("static class");
            // SAFETY: function signature is the documented Objective-C runtime ABI.
            let function: unsafe extern "C" fn(*const c_char) -> *mut c_void =
                unsafe { std::mem::transmute(self.get_class) };
            // SAFETY: `name` is NUL-terminated for the duration of the call.
            let class = unsafe { function(name.as_ptr()) };
            if class.is_null() {
                Err(MetalError::Driver {
                    operation: "class lookup",
                    detail: "required Objective-C class is absent".into(),
                })
            } else {
                Ok(class)
            }
        }

        fn retain(&self, object: *mut c_void) -> *mut c_void {
            if object.is_null() {
                return object;
            }
            // SAFETY: all callers pass a live Objective-C object.
            unsafe { self.msg0_obj(object, "retain") }
        }

        fn release(&self, object: *mut c_void) {
            if object.is_null() {
                return;
            }
            // SAFETY: owned objects are released exactly once by their RAII owner.
            unsafe { self.msg0_void(object, "release") };
        }

        unsafe fn msg0_obj(&self, object: *mut c_void, selector: &'static str) -> *mut c_void {
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: Objective-C receiver and selector are valid.
            unsafe { function(object, self.selector(selector)) }
        }

        unsafe fn msg0_void(&self, object: *mut c_void, selector: &'static str) {
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: Objective-C receiver and selector are valid.
            unsafe { function(object, self.selector(selector)) }
        }

        unsafe fn msg0_usize(&self, object: *mut c_void, selector: &'static str) -> usize {
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: Objective-C receiver and selector are valid.
            unsafe { function(object, self.selector(selector)) }
        }

        unsafe fn msg0_u64(&self, object: *mut c_void, selector: &'static str) -> u64 {
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64 =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: Objective-C receiver and selector are valid.
            unsafe { function(object, self.selector(selector)) }
        }

        unsafe fn msg0_bool(&self, object: *mut c_void, selector: &'static str) -> bool {
            // Objective-C BOOL is a signed byte on modern macOS.
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8 =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: Objective-C receiver and selector are valid.
            unsafe { function(object, self.selector(selector)) != 0 }
        }

        unsafe fn msg1_bool_usize(
            &self,
            object: *mut c_void,
            selector: &'static str,
            argument: usize,
        ) -> bool {
            // Objective-C BOOL is a signed byte on modern macOS.
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> i8 =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: Objective-C receiver and selector are valid.
            unsafe { function(object, self.selector(selector), argument) != 0 }
        }

        unsafe fn msg1_obj(
            &self,
            object: *mut c_void,
            selector: &'static str,
            argument: *mut c_void,
        ) -> *mut c_void {
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
            ) -> *mut c_void = unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: receiver, selector, and object argument are valid.
            unsafe { function(object, self.selector(selector), argument) }
        }

        unsafe fn msg1_void_bool(
            &self,
            object: *mut c_void,
            selector: &'static str,
            argument: bool,
        ) {
            // SAFETY: signature matches this selector family.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: receiver and selector are valid.
            unsafe { function(object, self.selector(selector), i8::from(argument)) };
        }

        fn string(&self, text: &str) -> Result<*mut c_void, MetalError> {
            let text = CString::new(text)
                .map_err(|_| MetalError::InvalidArgument("interior NUL in Metal string"))?;
            let class = self.class("NSString")?;
            // SAFETY: signature matches +stringWithUTF8String: and text is live.
            let function: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const c_char,
            ) -> *mut c_void = unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: class, selector, and C string are valid.
            let string =
                unsafe { function(class, self.selector("stringWithUTF8String:"), text.as_ptr()) };
            if string.is_null() {
                Err(MetalError::Driver {
                    operation: "NSString creation",
                    detail: "returned nil".into(),
                })
            } else {
                Ok(self.retain(string))
            }
        }

        fn rust_string(&self, string: *mut c_void) -> Result<String, MetalError> {
            if string.is_null() {
                return Err(MetalError::Utf8);
            }
            // SAFETY: signature matches -UTF8String.
            let function: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const c_char =
                unsafe { std::mem::transmute(self.msg_send) };
            // SAFETY: string is a live NSString.
            let pointer = unsafe { function(string, self.selector("UTF8String")) };
            if pointer.is_null() {
                return Err(MetalError::Utf8);
            }
            // SAFETY: UTF8String is NUL-terminated and lives with `string`.
            let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes();
            let bytes = &bytes[..bytes.len().min(MAX_DIAGNOSTIC_BYTES)];
            String::from_utf8(bytes.to_vec()).map_err(|_| MetalError::Utf8)
        }

        fn error_text(&self, error: *mut c_void) -> String {
            if error.is_null() {
                return "native call returned nil without NSError".into();
            }
            // SAFETY: NSError responds to localizedDescription.
            let description = unsafe { self.msg0_obj(error, "localizedDescription") };
            self.rust_string(description)
                .unwrap_or_else(|_| "native NSError diagnostic is not UTF-8".into())
        }
    }

    pub(crate) struct NativeDispatch {
        objc: Objc,
        _metal: Library,
        _core_graphics: Library,
        create_default_device: usize,
        copy_all_devices: Option<usize>,
    }

    impl NativeDispatch {
        pub(crate) fn load() -> Result<Self, MetalError> {
            let objc = Objc::load()?;
            // Loading CoreGraphics first matches Apple's requirement for default
            // Metal-device creation on systems where it initializes graphics.
            let core_graphics = Library::open(
                "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics",
                "CoreGraphics",
            )?;
            let metal = Library::open("/System/Library/Frameworks/Metal.framework/Metal", "Metal")?;
            Ok(Self {
                create_default_device: metal.symbol("MTLCreateSystemDefaultDevice")?,
                copy_all_devices: metal.optional_symbol("MTLCopyAllDevices"),
                objc,
                _metal: metal,
                _core_graphics: core_graphics,
            })
        }

        fn object(raw: usize, operation: &'static str) -> Result<*mut c_void, MetalError> {
            let object = raw as *mut c_void;
            if object.is_null() {
                Err(MetalError::Driver {
                    operation,
                    detail: "received a nil Objective-C object".into(),
                })
            } else {
                Ok(object)
            }
        }

        fn command_buffer(&self, queue: RawQueue) -> Result<*mut c_void, MetalError> {
            let queue = Self::object(queue.0, "command queue")?;
            // SAFETY: MTLCommandQueue implements -commandBuffer.
            let command = unsafe { self.objc.msg0_obj(queue, "commandBuffer") };
            if command.is_null() {
                Err(MetalError::Driver {
                    operation: "commandBuffer",
                    detail: "returned nil".into(),
                })
            } else {
                Ok(self.objc.retain(command))
            }
        }

        fn commit(&self, command: *mut c_void) {
            // SAFETY: command is a live MTLCommandBuffer.
            unsafe { self.objc.msg0_void(command, "commit") };
        }
    }

    impl Dispatch for NativeDispatch {
        fn devices(&self) -> Result<Vec<RawDevice>, MetalError> {
            if let Some(symbol) = self.copy_all_devices {
                // SAFETY: symbol is MTLCopyAllDevices with the documented ABI.
                let function: unsafe extern "C" fn() -> *mut c_void =
                    unsafe { std::mem::transmute(symbol) };
                // SAFETY: no arguments are required.
                let array = unsafe { function() };
                if !array.is_null() {
                    // SAFETY: NSArray implements -count and -objectAtIndex:.
                    let count = unsafe { self.objc.msg0_usize(array, "count") };
                    let mut devices = Vec::with_capacity(count);
                    for index in 0..count {
                        // SAFETY: index is within NSArray count.
                        let function: unsafe extern "C" fn(
                            *mut c_void,
                            *mut c_void,
                            usize,
                        ) -> *mut c_void = unsafe { std::mem::transmute(self.objc.msg_send) };
                        // SAFETY: array and selector are valid, index is checked.
                        let device =
                            unsafe { function(array, self.objc.selector("objectAtIndex:"), index) };
                        if !device.is_null() {
                            devices.push(RawDevice(self.objc.retain(device) as usize));
                        }
                    }
                    self.objc.release(array);
                    if !devices.is_empty() {
                        return Ok(devices);
                    }
                }
            }
            // SAFETY: symbol is MTLCreateSystemDefaultDevice with documented ABI.
            let function: unsafe extern "C" fn() -> *mut c_void =
                unsafe { std::mem::transmute(self.create_default_device) };
            // SAFETY: no arguments are required.
            let device = unsafe { function() };
            Ok((!device.is_null())
                .then_some(RawDevice(device as usize))
                .into_iter()
                .collect())
        }

        fn device_info(&self, device: RawDevice) -> Result<MetalDeviceInfo, MetalError> {
            let device = Self::object(device.0, "device info")?;
            // SAFETY: MTLDevice implements these selectors on supported macOS.
            let name = unsafe { self.objc.msg0_obj(device, "name") };
            let name = self.objc.rust_string(name)?;
            // SAFETY: registryID/maxBufferLength/hasUnifiedMemory return scalar values.
            let registry_id = unsafe { self.objc.msg0_u64(device, "registryID") };
            // SAFETY: selector return type is NSUInteger.
            let max_buffer_length = unsafe { self.objc.msg0_usize(device, "maxBufferLength") };
            // SAFETY: selector return type is BOOL.
            let unified_memory = unsafe { self.objc.msg0_bool(device, "hasUnifiedMemory") };
            let family = (1..=9)
                .rev()
                .find(|family| {
                    // MTLGPUFamilyApple1...Apple9 are 1001...1009.
                    // SAFETY: MTLDevice implements -supportsFamily: on current macOS.
                    unsafe {
                        self.objc
                            .msg1_bool_usize(device, "supportsFamily:", 1000 + family)
                    }
                })
                .map(|family| format!("Apple{family}"))
                .or_else(|| {
                    (1..=2).rev().find_map(|family| {
                        // MTLGPUFamilyMac1...Mac2 are 2001...2002.
                        // SAFETY: same selector contract as the Apple-family query.
                        unsafe {
                            self.objc
                                .msg1_bool_usize(device, "supportsFamily:", 2000 + family)
                        }
                        .then(|| format!("Mac{family}"))
                    })
                })
                .unwrap_or_else(|| "UnclassifiedMetalFamily".into());
            Ok(MetalDeviceInfo {
                name,
                registry_id,
                capabilities: MetalCapabilities {
                    max_buffer_length,
                    unified_memory,
                    family,
                },
            })
        }

        fn device_release(&self, device: RawDevice) {
            self.objc.release(device.0 as *mut c_void);
        }

        fn queue_create(&self, device: RawDevice, _owner: u64) -> Result<RawQueue, MetalError> {
            let device = Self::object(device.0, "newCommandQueue")?;
            // SAFETY: MTLDevice implements -newCommandQueue and returns owned +1.
            let queue = unsafe { self.objc.msg0_obj(device, "newCommandQueue") };
            if queue.is_null() {
                Err(MetalError::Driver {
                    operation: "newCommandQueue",
                    detail: "returned nil".into(),
                })
            } else {
                Ok(RawQueue(queue as usize))
            }
        }

        fn queue_release(&self, queue: RawQueue, _owner: u64) {
            self.objc.release(queue.0 as *mut c_void);
        }

        fn buffer_create(
            &self,
            device: RawDevice,
            bytes: usize,
            _owner: u64,
        ) -> Result<RawBuffer, MetalError> {
            let device = Self::object(device.0, "newBufferWithLength")?;
            // SAFETY: selector takes NSUInteger length/options and returns +1.
            let function: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                usize,
                usize,
            ) -> *mut c_void = unsafe { std::mem::transmute(self.objc.msg_send) };
            // MTLResourceStorageModeShared is zero on macOS.
            // SAFETY: device and selector are valid.
            let buffer = unsafe {
                function(
                    device,
                    self.objc.selector("newBufferWithLength:options:"),
                    bytes,
                    0,
                )
            };
            if buffer.is_null() {
                Err(MetalError::Driver {
                    operation: "newBufferWithLength",
                    detail: "returned nil (allocation failed)".into(),
                })
            } else {
                Ok(RawBuffer(buffer as usize))
            }
        }

        fn buffer_release(&self, buffer: RawBuffer, _owner: u64) {
            self.objc.release(buffer.0 as *mut c_void);
        }

        fn buffer_write(
            &self,
            buffer: RawBuffer,
            offset: usize,
            bytes: &[u8],
            _owner: u64,
        ) -> Result<(), MetalError> {
            let buffer = Self::object(buffer.0, "buffer contents")?;
            // SAFETY: MTLBuffer implements -contents and safe layer checked bounds.
            let contents = unsafe { self.objc.msg0_obj(buffer, "contents") } as *mut u8;
            if contents.is_null() {
                return Err(MetalError::Driver {
                    operation: "buffer contents",
                    detail: "shared buffer returned null contents".into(),
                });
            }
            // SAFETY: source and checked destination ranges do not overlap.
            unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), contents.add(offset), bytes.len()) };
            Ok(())
        }

        fn buffer_read(
            &self,
            buffer: RawBuffer,
            offset: usize,
            bytes: &mut [u8],
            _owner: u64,
        ) -> Result<(), MetalError> {
            let buffer = Self::object(buffer.0, "buffer contents")?;
            // SAFETY: MTLBuffer implements -contents and safe layer checked bounds.
            let contents = unsafe { self.objc.msg0_obj(buffer, "contents") } as *const u8;
            if contents.is_null() {
                return Err(MetalError::Driver {
                    operation: "buffer contents",
                    detail: "shared buffer returned null contents".into(),
                });
            }
            // SAFETY: checked source and destination ranges do not overlap.
            unsafe {
                ptr::copy_nonoverlapping(contents.add(offset), bytes.as_mut_ptr(), bytes.len())
            };
            Ok(())
        }

        fn buffer_copy(
            &self,
            queue: RawQueue,
            src: RawBuffer,
            dst: RawBuffer,
            region: CopyRegion,
            _owner: u64,
        ) -> Result<RawCommand, MetalError> {
            let command = self.command_buffer(queue)?;
            // SAFETY: command buffer implements -blitCommandEncoder.
            let encoder = unsafe { self.objc.msg0_obj(command, "blitCommandEncoder") };
            if encoder.is_null() {
                self.objc.release(command);
                return Err(MetalError::Driver {
                    operation: "blitCommandEncoder",
                    detail: "returned nil".into(),
                });
            }
            let encoder = self.objc.retain(encoder);
            // SAFETY: selector ABI is (buffer, NSUInteger, buffer, NSUInteger, NSUInteger).
            let function: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                usize,
                *mut c_void,
                usize,
                usize,
            ) = unsafe { std::mem::transmute(self.objc.msg_send) };
            // SAFETY: safe layer validated ownership and regions.
            unsafe {
                function(
                    encoder,
                    self.objc
                        .selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
                    src.0 as *mut c_void,
                    region.src_offset,
                    dst.0 as *mut c_void,
                    region.dst_offset,
                    region.bytes,
                );
                self.objc.msg0_void(encoder, "endEncoding");
            }
            self.objc.release(encoder);
            self.commit(command);
            Ok(RawCommand(command as usize))
        }

        fn library_compile(
            &self,
            device: RawDevice,
            source: &str,
            _owner: u64,
        ) -> Result<RawLibrary, MetalError> {
            let device = Self::object(device.0, "newLibraryWithSource")?;
            let source = self.objc.string(source)?;
            let options_class = self.objc.class("MTLCompileOptions")?;
            // SAFETY: +new returns an owned compile-options instance.
            let options = unsafe { self.objc.msg0_obj(options_class, "new") };
            if options.is_null() {
                self.objc.release(source);
                return Err(MetalError::Driver {
                    operation: "MTLCompileOptions new",
                    detail: "returned nil".into(),
                });
            }
            // SAFETY: MTLCompileOptions implements -setFastMathEnabled:.
            unsafe {
                self.objc
                    .msg1_void_bool(options, "setFastMathEnabled:", false)
            };
            let mut error = ptr::null_mut();
            // SAFETY: selector ABI is (NSString, MTLCompileOptions, NSError**).
            let function: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut *mut c_void,
            ) -> *mut c_void = unsafe { std::mem::transmute(self.objc.msg_send) };
            // SAFETY: all Objective-C objects and error out-pointer are valid.
            let library = unsafe {
                function(
                    device,
                    self.objc.selector("newLibraryWithSource:options:error:"),
                    source,
                    options,
                    &mut error,
                )
            };
            self.objc.release(options);
            self.objc.release(source);
            if library.is_null() {
                Err(MetalError::Build {
                    diagnostic: self.objc.error_text(error),
                })
            } else {
                Ok(RawLibrary(library as usize))
            }
        }

        fn library_release(&self, library: RawLibrary, _owner: u64) {
            self.objc.release(library.0 as *mut c_void);
        }

        fn pipeline_create(
            &self,
            device: RawDevice,
            library: RawLibrary,
            entry: &str,
            _owner: u64,
        ) -> Result<(RawPipeline, usize), MetalError> {
            let library = Self::object(library.0, "newFunctionWithName")?;
            let entry = self.objc.string(entry)?;
            // SAFETY: MTLLibrary implements -newFunctionWithName: and returns +1.
            let function = unsafe { self.objc.msg1_obj(library, "newFunctionWithName:", entry) };
            self.objc.release(entry);
            if function.is_null() {
                return Err(MetalError::Driver {
                    operation: "newFunctionWithName",
                    detail: "entry point was not found".into(),
                });
            }
            let device = Self::object(device.0, "newComputePipelineState")?;
            let mut error = ptr::null_mut();
            // SAFETY: selector ABI is (MTLFunction, NSError**).
            let create: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut *mut c_void,
            ) -> *mut c_void = unsafe { std::mem::transmute(self.objc.msg_send) };
            // SAFETY: device, function, and error pointer are valid.
            let pipeline = unsafe {
                create(
                    device,
                    self.objc
                        .selector("newComputePipelineStateWithFunction:error:"),
                    function,
                    &mut error,
                )
            };
            self.objc.release(function);
            if pipeline.is_null() {
                return Err(MetalError::Driver {
                    operation: "newComputePipelineStateWithFunction",
                    detail: self.objc.error_text(error),
                });
            }
            // SAFETY: pipeline implements -maxTotalThreadsPerThreadgroup.
            let max_threads = unsafe {
                self.objc
                    .msg0_usize(pipeline, "maxTotalThreadsPerThreadgroup")
            };
            Ok((RawPipeline(pipeline as usize), max_threads))
        }

        fn pipeline_release(&self, pipeline: RawPipeline, _owner: u64) {
            self.objc.release(pipeline.0 as *mut c_void);
        }

        fn launch(
            &self,
            queue: RawQueue,
            pipeline: RawPipeline,
            buffers: &[RawBuffer],
            geometry: LaunchGeometry,
            _owner: u64,
        ) -> Result<RawCommand, MetalError> {
            let command = self.command_buffer(queue)?;
            // SAFETY: command buffer implements -computeCommandEncoder.
            let encoder = unsafe { self.objc.msg0_obj(command, "computeCommandEncoder") };
            if encoder.is_null() {
                self.objc.release(command);
                return Err(MetalError::Driver {
                    operation: "computeCommandEncoder",
                    detail: "returned nil".into(),
                });
            }
            let encoder = self.objc.retain(encoder);
            // SAFETY: selector takes one MTLComputePipelineState object.
            let set_pipeline: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
                unsafe { std::mem::transmute(self.objc.msg_send) };
            // SAFETY: encoder and pipeline are live.
            unsafe {
                set_pipeline(
                    encoder,
                    self.objc.selector("setComputePipelineState:"),
                    pipeline.0 as *mut c_void,
                )
            };
            // SAFETY: selector ABI is (MTLBuffer, NSUInteger, NSUInteger).
            let set_buffer: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                usize,
                usize,
            ) = unsafe { std::mem::transmute(self.objc.msg_send) };
            for (index, buffer) in buffers.iter().enumerate() {
                // Transactional kernels reserve the ordinary ABI terminator for
                // the scalar extent, so appended status buffers start one slot
                // later. Standard launches have no buffer after extent_index.
                let abi_index = if index >= geometry.extent_index {
                    index + 1
                } else {
                    index
                };
                // SAFETY: safe layer validated each live buffer and index.
                unsafe {
                    set_buffer(
                        encoder,
                        self.objc.selector("setBuffer:offset:atIndex:"),
                        buffer.0 as *mut c_void,
                        0,
                        abi_index,
                    )
                };
            }
            // SAFETY: selector ABI is (const void*, NSUInteger, NSUInteger) and
            // Metal copies setBytes contents before this call returns.
            let set_bytes: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const c_void,
                usize,
                usize,
            ) = unsafe { std::mem::transmute(self.objc.msg_send) };
            // SAFETY: extent pointer is valid for the synchronous selector call.
            unsafe {
                set_bytes(
                    encoder,
                    self.objc.selector("setBytes:length:atIndex:"),
                    (&geometry.extent as *const u64).cast(),
                    std::mem::size_of::<u64>(),
                    geometry.extent_index,
                )
            };
            // SAFETY: selector takes two MTLSize values by value.
            let dispatch: unsafe extern "C" fn(*mut c_void, *mut c_void, MtlSize, MtlSize) =
                unsafe { std::mem::transmute(self.objc.msg_send) };
            let groups = geometry.global / geometry.local;
            // SAFETY: geometry was completely validated by the safe layer.
            unsafe {
                dispatch(
                    encoder,
                    self.objc
                        .selector("dispatchThreadgroups:threadsPerThreadgroup:"),
                    MtlSize {
                        width: groups,
                        height: 1,
                        depth: 1,
                    },
                    MtlSize {
                        width: geometry.local,
                        height: 1,
                        depth: 1,
                    },
                );
                self.objc.msg0_void(encoder, "endEncoding");
            }
            self.objc.release(encoder);
            self.commit(command);
            Ok(RawCommand(command as usize))
        }

        fn command_query(&self, command: RawCommand, _owner: u64) -> Result<bool, MetalError> {
            let command = Self::object(command.0, "command status")?;
            // MTLCommandBufferStatusCompleted=4 and Error=5.
            // SAFETY: command implements -status returning NSUInteger.
            let status = unsafe { self.objc.msg0_usize(command, "status") };
            Ok(status >= 4)
        }

        fn command_wait(&self, command: RawCommand, _owner: u64) -> Result<(), MetalError> {
            let command = Self::object(command.0, "waitUntilCompleted")?;
            // SAFETY: command implements -waitUntilCompleted.
            unsafe { self.objc.msg0_void(command, "waitUntilCompleted") };
            // SAFETY: command implements -status and -error.
            let status = unsafe { self.objc.msg0_usize(command, "status") };
            if status == 5 {
                // SAFETY: command implements -error.
                let error = unsafe { self.objc.msg0_obj(command, "error") };
                Err(MetalError::Driver {
                    operation: "command completion",
                    detail: self.objc.error_text(error),
                })
            } else if status == 4 {
                Ok(())
            } else {
                Err(MetalError::Driver {
                    operation: "command completion",
                    detail: format!("unexpected terminal status {status}"),
                })
            }
        }

        fn command_release(&self, command: RawCommand, _owner: u64) {
            self.objc.release(command.0 as *mut c_void);
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) use platform::NativeDispatch;

#[cfg(not(target_os = "macos"))]
pub(super) struct NativeDispatch;

#[cfg(not(target_os = "macos"))]
impl NativeDispatch {
    pub(super) fn load() -> Result<Self, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
}

#[cfg(not(target_os = "macos"))]
impl Dispatch for NativeDispatch {
    fn devices(&self) -> Result<Vec<RawDevice>, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn device_info(&self, _: RawDevice) -> Result<MetalDeviceInfo, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn device_release(&self, _: RawDevice) {}
    fn queue_create(&self, _: RawDevice, _: u64) -> Result<RawQueue, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn queue_release(&self, _: RawQueue, _: u64) {}
    fn buffer_create(&self, _: RawDevice, _: usize, _: u64) -> Result<RawBuffer, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn buffer_release(&self, _: RawBuffer, _: u64) {}
    fn buffer_write(&self, _: RawBuffer, _: usize, _: &[u8], _: u64) -> Result<(), MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn buffer_read(&self, _: RawBuffer, _: usize, _: &mut [u8], _: u64) -> Result<(), MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn buffer_copy(
        &self,
        _: RawQueue,
        _: RawBuffer,
        _: RawBuffer,
        _: CopyRegion,
        _: u64,
    ) -> Result<RawCommand, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn library_compile(&self, _: RawDevice, _: &str, _: u64) -> Result<RawLibrary, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn library_release(&self, _: RawLibrary, _: u64) {}
    fn pipeline_create(
        &self,
        _: RawDevice,
        _: RawLibrary,
        _: &str,
        _: u64,
    ) -> Result<(RawPipeline, usize), MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn pipeline_release(&self, _: RawPipeline, _: u64) {}
    fn launch(
        &self,
        _: RawQueue,
        _: RawPipeline,
        _: &[RawBuffer],
        _: LaunchGeometry,
        _: u64,
    ) -> Result<RawCommand, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn command_query(&self, _: RawCommand, _: u64) -> Result<bool, MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn command_wait(&self, _: RawCommand, _: u64) -> Result<(), MetalError> {
        Err(MetalError::PlatformUnsupported)
    }
    fn command_release(&self, _: RawCommand, _: u64) {}
}
