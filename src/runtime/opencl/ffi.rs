//! Audited OpenCL C ABI and dynamic symbol boundary.
//!
//! All function-pointer casts and raw pointer calls are confined to this file.
//! The safe resource layer validates ownership, sizes, and lifetimes before
//! invoking this adapter.
use super::{
    BufferCopyRegion, BuildInfo, DeviceInfo, Dispatch, OpenClError, RawBuffer, RawContext,
    RawDevice, RawEvent, RawKernel, RawPlatform, RawProgram, RawQueue,
};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_ulong, c_void},
    ptr,
};

type ClInt = c_int;
type ClUint = c_uint;
type ClUlong = c_ulong;
type ClBool = ClUint;
type ClBitfield = ClUlong;
type ClDeviceType = ClBitfield;
type ClMemFlags = ClBitfield;
type ClContextProperties = isize;
type ClCommandQueueProperties = ClBitfield;
type ClPlatformId = *mut c_void;
type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClCommandQueue = *mut c_void;
type ClMem = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;
type ClEvent = *mut c_void;

const CL_SUCCESS: ClInt = 0;
const CL_DEVICE_NOT_FOUND: ClInt = -1;
const CL_TRUE: ClBool = 1;
const CL_DEVICE_TYPE_ALL: ClDeviceType = !0;
const CL_PLATFORM_NAME: ClUint = 0x0902;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: ClUint = 0x1004;
const CL_DEVICE_NAME: ClUint = 0x102b;
const CL_MEM_READ_WRITE: ClMemFlags = 1;
const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;
const CL_EVENT_COMMAND_EXECUTION_STATUS: ClUint = 0x1283;
const CL_COMPLETE: ClInt = 0;
const MAX_INFO_BYTES: usize = 64 * 1024;

type ContextNotify = Option<unsafe extern "C" fn(*const c_char, *const c_void, usize, *mut c_void)>;

macro_rules! native_table {
    ($($name:ident: $ty:ty),* $(,)?) => {
        struct NativeTable { $($name: $ty,)* }
    };
}

native_table!(
    get_platform_ids: unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt,
    get_platform_info: unsafe extern "C" fn(ClPlatformId, ClUint, usize, *mut c_void, *mut usize) -> ClInt,
    get_device_ids: unsafe extern "C" fn(ClPlatformId, ClDeviceType, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt,
    get_device_info: unsafe extern "C" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt,
    create_context: unsafe extern "C" fn(*const ClContextProperties, ClUint, *const ClDeviceId, ContextNotify, *mut c_void, *mut ClInt) -> ClContext,
    release_context: unsafe extern "C" fn(ClContext) -> ClInt,
    create_queue: unsafe extern "C" fn(ClContext, ClDeviceId, ClCommandQueueProperties, *mut ClInt) -> ClCommandQueue,
    release_queue: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
    finish: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
    create_buffer: unsafe extern "C" fn(ClContext, ClMemFlags, usize, *mut c_void, *mut ClInt) -> ClMem,
    release_mem: unsafe extern "C" fn(ClMem) -> ClInt,
    enqueue_write: unsafe extern "C" fn(ClCommandQueue, ClMem, ClBool, usize, usize, *const c_void, ClUint, *const ClEvent, *mut ClEvent) -> ClInt,
    enqueue_read: unsafe extern "C" fn(ClCommandQueue, ClMem, ClBool, usize, usize, *mut c_void, ClUint, *const ClEvent, *mut ClEvent) -> ClInt,
    enqueue_copy: unsafe extern "C" fn(ClCommandQueue, ClMem, ClMem, usize, usize, usize, ClUint, *const ClEvent, *mut ClEvent) -> ClInt,
    create_program: unsafe extern "C" fn(ClContext, ClUint, *const *const c_char, *const usize, *mut ClInt) -> ClProgram,
    build_program: unsafe extern "C" fn(ClProgram, ClUint, *const ClDeviceId, *const c_char, Option<unsafe extern "C" fn(ClProgram, *mut c_void)>, *mut c_void) -> ClInt,
    get_program_build_info: unsafe extern "C" fn(ClProgram, ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt,
    release_program: unsafe extern "C" fn(ClProgram) -> ClInt,
    create_kernel: unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel,
    release_kernel: unsafe extern "C" fn(ClKernel) -> ClInt,
    set_kernel_arg: unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt,
    enqueue_ndrange: unsafe extern "C" fn(ClCommandQueue, ClKernel, ClUint, *const usize, *const usize, *const usize, ClUint, *const ClEvent, *mut ClEvent) -> ClInt,
    get_event_info: unsafe extern "C" fn(ClEvent, ClUint, usize, *mut c_void, *mut usize) -> ClInt,
    wait_for_events: unsafe extern "C" fn(ClUint, *const ClEvent) -> ClInt,
    release_event: unsafe extern "C" fn(ClEvent) -> ClInt,
);

pub(super) struct NativeDispatch {
    _library: Library,
    table: NativeTable,
}

impl NativeDispatch {
    pub(super) fn load() -> Result<Self, OpenClError> {
        let library = Library::open()?;
        macro_rules! sym {
            ($rust:ident, $symbol:literal, $ty:ty) => {
                let $rust: $ty = unsafe {
                    // SAFETY: `Library::symbol` returned a non-null address for
                    // the exact OpenCL symbol and this is its published C ABI.
                    std::mem::transmute::<*mut c_void, $ty>(resolve_required(
                        $symbol,
                        concat!($symbol, "\0").as_bytes(),
                        |name| library.symbol(name).ok(),
                    )?)
                };
            };
        }
        sym!(
            get_platform_ids,
            "clGetPlatformIDs",
            unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt
        );
        sym!(
            get_platform_info,
            "clGetPlatformInfo",
            unsafe extern "C" fn(ClPlatformId, ClUint, usize, *mut c_void, *mut usize) -> ClInt
        );
        sym!(
            get_device_ids,
            "clGetDeviceIDs",
            unsafe extern "C" fn(
                ClPlatformId,
                ClDeviceType,
                ClUint,
                *mut ClDeviceId,
                *mut ClUint,
            ) -> ClInt
        );
        sym!(
            get_device_info,
            "clGetDeviceInfo",
            unsafe extern "C" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt
        );
        sym!(
            create_context,
            "clCreateContext",
            unsafe extern "C" fn(
                *const ClContextProperties,
                ClUint,
                *const ClDeviceId,
                ContextNotify,
                *mut c_void,
                *mut ClInt,
            ) -> ClContext
        );
        sym!(
            release_context,
            "clReleaseContext",
            unsafe extern "C" fn(ClContext) -> ClInt
        );
        sym!(
            create_queue,
            "clCreateCommandQueue",
            unsafe extern "C" fn(
                ClContext,
                ClDeviceId,
                ClCommandQueueProperties,
                *mut ClInt,
            ) -> ClCommandQueue
        );
        sym!(
            release_queue,
            "clReleaseCommandQueue",
            unsafe extern "C" fn(ClCommandQueue) -> ClInt
        );
        sym!(
            finish,
            "clFinish",
            unsafe extern "C" fn(ClCommandQueue) -> ClInt
        );
        sym!(
            create_buffer,
            "clCreateBuffer",
            unsafe extern "C" fn(ClContext, ClMemFlags, usize, *mut c_void, *mut ClInt) -> ClMem
        );
        sym!(
            release_mem,
            "clReleaseMemObject",
            unsafe extern "C" fn(ClMem) -> ClInt
        );
        sym!(
            enqueue_write,
            "clEnqueueWriteBuffer",
            unsafe extern "C" fn(
                ClCommandQueue,
                ClMem,
                ClBool,
                usize,
                usize,
                *const c_void,
                ClUint,
                *const ClEvent,
                *mut ClEvent,
            ) -> ClInt
        );
        sym!(
            enqueue_read,
            "clEnqueueReadBuffer",
            unsafe extern "C" fn(
                ClCommandQueue,
                ClMem,
                ClBool,
                usize,
                usize,
                *mut c_void,
                ClUint,
                *const ClEvent,
                *mut ClEvent,
            ) -> ClInt
        );
        sym!(
            enqueue_copy,
            "clEnqueueCopyBuffer",
            unsafe extern "C" fn(
                ClCommandQueue,
                ClMem,
                ClMem,
                usize,
                usize,
                usize,
                ClUint,
                *const ClEvent,
                *mut ClEvent,
            ) -> ClInt
        );
        sym!(
            create_program,
            "clCreateProgramWithSource",
            unsafe extern "C" fn(
                ClContext,
                ClUint,
                *const *const c_char,
                *const usize,
                *mut ClInt,
            ) -> ClProgram
        );
        sym!(
            build_program,
            "clBuildProgram",
            unsafe extern "C" fn(
                ClProgram,
                ClUint,
                *const ClDeviceId,
                *const c_char,
                Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
                *mut c_void,
            ) -> ClInt
        );
        sym!(
            get_program_build_info,
            "clGetProgramBuildInfo",
            unsafe extern "C" fn(
                ClProgram,
                ClDeviceId,
                ClUint,
                usize,
                *mut c_void,
                *mut usize,
            ) -> ClInt
        );
        sym!(
            release_program,
            "clReleaseProgram",
            unsafe extern "C" fn(ClProgram) -> ClInt
        );
        sym!(
            create_kernel,
            "clCreateKernel",
            unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel
        );
        sym!(
            release_kernel,
            "clReleaseKernel",
            unsafe extern "C" fn(ClKernel) -> ClInt
        );
        sym!(
            set_kernel_arg,
            "clSetKernelArg",
            unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt
        );
        sym!(
            enqueue_ndrange,
            "clEnqueueNDRangeKernel",
            unsafe extern "C" fn(
                ClCommandQueue,
                ClKernel,
                ClUint,
                *const usize,
                *const usize,
                *const usize,
                ClUint,
                *const ClEvent,
                *mut ClEvent,
            ) -> ClInt
        );
        sym!(
            get_event_info,
            "clGetEventInfo",
            unsafe extern "C" fn(ClEvent, ClUint, usize, *mut c_void, *mut usize) -> ClInt
        );
        sym!(
            wait_for_events,
            "clWaitForEvents",
            unsafe extern "C" fn(ClUint, *const ClEvent) -> ClInt
        );
        sym!(
            release_event,
            "clReleaseEvent",
            unsafe extern "C" fn(ClEvent) -> ClInt
        );
        Ok(Self {
            _library: library,
            table: NativeTable {
                get_platform_ids,
                get_platform_info,
                get_device_ids,
                get_device_info,
                create_context,
                release_context,
                create_queue,
                release_queue,
                finish,
                create_buffer,
                release_mem,
                enqueue_write,
                enqueue_read,
                enqueue_copy,
                create_program,
                build_program,
                get_program_build_info,
                release_program,
                create_kernel,
                release_kernel,
                set_kernel_arg,
                enqueue_ndrange,
                get_event_info,
                wait_for_events,
                release_event,
            },
        })
    }
}

impl Dispatch for NativeDispatch {
    fn platforms(&self) -> Result<Vec<RawPlatform>, OpenClError> {
        let mut count = 0;
        check("clGetPlatformIDs", unsafe {
            (self.table.get_platform_ids)(0, ptr::null_mut(), &mut count)
        })?;
        let mut values = vec![ptr::null_mut(); count as usize];
        check("clGetPlatformIDs", unsafe {
            (self.table.get_platform_ids)(count, values.as_mut_ptr(), ptr::null_mut())
        })?;
        Ok(values.into_iter().map(RawPlatform::from_ptr).collect())
    }

    fn platform_name(&self, platform: RawPlatform) -> Result<String, OpenClError> {
        info_string("clGetPlatformInfo", |size, out, actual| unsafe {
            (self.table.get_platform_info)(platform.as_ptr(), CL_PLATFORM_NAME, size, out, actual)
        })
    }

    fn devices(&self, platform: RawPlatform) -> Result<Vec<RawDevice>, OpenClError> {
        let mut count = 0;
        let status = unsafe {
            (self.table.get_device_ids)(
                platform.as_ptr(),
                CL_DEVICE_TYPE_ALL,
                0,
                ptr::null_mut(),
                &mut count,
            )
        };
        if status == CL_DEVICE_NOT_FOUND {
            return Ok(Vec::new());
        }
        check("clGetDeviceIDs", status)?;
        let mut values = vec![ptr::null_mut(); count as usize];
        check("clGetDeviceIDs", unsafe {
            (self.table.get_device_ids)(
                platform.as_ptr(),
                CL_DEVICE_TYPE_ALL,
                count,
                values.as_mut_ptr(),
                ptr::null_mut(),
            )
        })?;
        Ok(values.into_iter().map(RawDevice::from_ptr).collect())
    }

    fn device_info(&self, device: RawDevice) -> Result<DeviceInfo, OpenClError> {
        let name = info_string("clGetDeviceInfo", |size, out, actual| unsafe {
            (self.table.get_device_info)(device.as_ptr(), CL_DEVICE_NAME, size, out, actual)
        })?;
        let mut max_work_group_size = 0usize;
        check("clGetDeviceInfo", unsafe {
            (self.table.get_device_info)(
                device.as_ptr(),
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                std::mem::size_of::<usize>(),
                (&mut max_work_group_size as *mut usize).cast(),
                ptr::null_mut(),
            )
        })?;
        Ok(DeviceInfo {
            name,
            max_work_group_size,
        })
    }

    fn context_create(&self, device: RawDevice, _owner: u64) -> Result<RawContext, OpenClError> {
        let mut status = CL_SUCCESS;
        let raw = unsafe {
            (self.table.create_context)(
                ptr::null(),
                1,
                &device.as_ptr(),
                None,
                ptr::null_mut(),
                &mut status,
            )
        };
        check_create("clCreateContext", status, raw).map(RawContext::from_ptr)
    }

    fn context_release(&self, context: RawContext, _owner: u64) -> Result<(), OpenClError> {
        check("clReleaseContext", unsafe {
            (self.table.release_context)(context.as_ptr())
        })
    }

    fn queue_create(
        &self,
        context: RawContext,
        device: RawDevice,
        _owner: u64,
    ) -> Result<RawQueue, OpenClError> {
        let mut status = CL_SUCCESS;
        let raw =
            unsafe { (self.table.create_queue)(context.as_ptr(), device.as_ptr(), 0, &mut status) };
        check_create("clCreateCommandQueue", status, raw).map(RawQueue::from_ptr)
    }

    fn queue_release(&self, queue: RawQueue, _owner: u64) -> Result<(), OpenClError> {
        check("clReleaseCommandQueue", unsafe {
            (self.table.release_queue)(queue.as_ptr())
        })
    }

    fn queue_finish(&self, queue: RawQueue, _owner: u64) -> Result<(), OpenClError> {
        check("clFinish", unsafe { (self.table.finish)(queue.as_ptr()) })
    }

    fn buffer_create(
        &self,
        context: RawContext,
        bytes: usize,
        _owner: u64,
    ) -> Result<RawBuffer, OpenClError> {
        let mut status = CL_SUCCESS;
        let raw = unsafe {
            (self.table.create_buffer)(
                context.as_ptr(),
                CL_MEM_READ_WRITE,
                bytes,
                ptr::null_mut(),
                &mut status,
            )
        };
        check_create("clCreateBuffer", status, raw).map(RawBuffer::from_ptr)
    }

    fn buffer_release(&self, buffer: RawBuffer, _owner: u64) -> Result<(), OpenClError> {
        check("clReleaseMemObject", unsafe {
            (self.table.release_mem)(buffer.as_ptr())
        })
    }

    fn buffer_write(
        &self,
        queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &[u8],
        _owner: u64,
    ) -> Result<(), OpenClError> {
        check("clEnqueueWriteBuffer", unsafe {
            (self.table.enqueue_write)(
                queue.as_ptr(),
                buffer.as_ptr(),
                CL_TRUE,
                offset,
                bytes.len(),
                bytes.as_ptr().cast(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    fn buffer_read(
        &self,
        queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &mut [u8],
        _owner: u64,
    ) -> Result<(), OpenClError> {
        check("clEnqueueReadBuffer", unsafe {
            (self.table.enqueue_read)(
                queue.as_ptr(),
                buffer.as_ptr(),
                CL_TRUE,
                offset,
                bytes.len(),
                bytes.as_mut_ptr().cast(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    fn buffer_copy(
        &self,
        queue: RawQueue,
        src: RawBuffer,
        dst: RawBuffer,
        region: BufferCopyRegion,
        _owner: u64,
    ) -> Result<RawEvent, OpenClError> {
        let mut event = ptr::null_mut();
        check("clEnqueueCopyBuffer", unsafe {
            (self.table.enqueue_copy)(
                queue.as_ptr(),
                src.as_ptr(),
                dst.as_ptr(),
                region.src_offset,
                region.dst_offset,
                region.bytes,
                0,
                ptr::null(),
                &mut event,
            )
        })?;
        non_null("clEnqueueCopyBuffer event", event).map(RawEvent::from_ptr)
    }

    fn program_create(
        &self,
        context: RawContext,
        source: &str,
        _owner: u64,
    ) -> Result<RawProgram, OpenClError> {
        let mut status = CL_SUCCESS;
        let source_ptr = source.as_ptr().cast::<c_char>();
        let length = source.len();
        let raw = unsafe {
            (self.table.create_program)(context.as_ptr(), 1, &source_ptr, &length, &mut status)
        };
        check_create("clCreateProgramWithSource", status, raw).map(RawProgram::from_ptr)
    }

    fn program_build(
        &self,
        program: RawProgram,
        device: RawDevice,
        options: &str,
        _owner: u64,
    ) -> Result<(), OpenClError> {
        let options = CString::new(options)
            .map_err(|_| OpenClError::InvalidArgument("interior NUL in build options"))?;
        check("clBuildProgram", unsafe {
            (self.table.build_program)(
                program.as_ptr(),
                1,
                &device.as_ptr(),
                options.as_ptr(),
                None,
                ptr::null_mut(),
            )
        })
    }

    fn program_build_info(
        &self,
        program: RawProgram,
        device: RawDevice,
        _owner: u64,
    ) -> Result<BuildInfo, OpenClError> {
        let log = info_string("clGetProgramBuildInfo", |size, out, actual| unsafe {
            (self.table.get_program_build_info)(
                program.as_ptr(),
                device.as_ptr(),
                CL_PROGRAM_BUILD_LOG,
                size,
                out,
                actual,
            )
        })?;
        Ok(BuildInfo { log })
    }

    fn program_release(&self, program: RawProgram, _owner: u64) -> Result<(), OpenClError> {
        check("clReleaseProgram", unsafe {
            (self.table.release_program)(program.as_ptr())
        })
    }

    fn kernel_create(
        &self,
        program: RawProgram,
        entry: &str,
        _owner: u64,
    ) -> Result<RawKernel, OpenClError> {
        let entry = CString::new(entry)
            .map_err(|_| OpenClError::InvalidArgument("interior NUL in kernel name"))?;
        let mut status = CL_SUCCESS;
        let raw =
            unsafe { (self.table.create_kernel)(program.as_ptr(), entry.as_ptr(), &mut status) };
        check_create("clCreateKernel", status, raw).map(RawKernel::from_ptr)
    }

    fn kernel_release(&self, kernel: RawKernel, _owner: u64) -> Result<(), OpenClError> {
        check("clReleaseKernel", unsafe {
            (self.table.release_kernel)(kernel.as_ptr())
        })
    }

    fn kernel_arg_buffer(
        &self,
        kernel: RawKernel,
        index: u32,
        buffer: RawBuffer,
        _owner: u64,
    ) -> Result<(), OpenClError> {
        let raw = buffer.as_ptr();
        check("clSetKernelArg", unsafe {
            (self.table.set_kernel_arg)(
                kernel.as_ptr(),
                index,
                std::mem::size_of::<ClMem>(),
                (&raw as *const ClMem).cast(),
            )
        })
    }

    fn kernel_arg_u64(
        &self,
        kernel: RawKernel,
        index: u32,
        value: u64,
        _owner: u64,
    ) -> Result<(), OpenClError> {
        check("clSetKernelArg", unsafe {
            (self.table.set_kernel_arg)(
                kernel.as_ptr(),
                index,
                std::mem::size_of::<u64>(),
                (&value as *const u64).cast(),
            )
        })
    }

    fn kernel_launch(
        &self,
        queue: RawQueue,
        kernel: RawKernel,
        global: usize,
        local: usize,
        _owner: u64,
    ) -> Result<RawEvent, OpenClError> {
        let mut event = ptr::null_mut();
        check("clEnqueueNDRangeKernel", unsafe {
            (self.table.enqueue_ndrange)(
                queue.as_ptr(),
                kernel.as_ptr(),
                1,
                ptr::null(),
                &global,
                &local,
                0,
                ptr::null(),
                &mut event,
            )
        })?;
        non_null("clEnqueueNDRangeKernel event", event).map(RawEvent::from_ptr)
    }

    fn event_query(&self, event: RawEvent, _owner: u64) -> Result<bool, OpenClError> {
        let mut status = 0i32;
        check("clGetEventInfo", unsafe {
            (self.table.get_event_info)(
                event.as_ptr(),
                CL_EVENT_COMMAND_EXECUTION_STATUS,
                std::mem::size_of::<i32>(),
                (&mut status as *mut i32).cast(),
                ptr::null_mut(),
            )
        })?;
        if status < 0 {
            Err(OpenClError::Driver {
                operation: "event execution",
                code: status,
            })
        } else {
            Ok(status == CL_COMPLETE)
        }
    }

    fn event_wait(&self, event: RawEvent, _owner: u64) -> Result<(), OpenClError> {
        let raw = event.as_ptr();
        check("clWaitForEvents", unsafe {
            (self.table.wait_for_events)(1, &raw)
        })
    }

    fn event_release(&self, event: RawEvent, _owner: u64) -> Result<(), OpenClError> {
        check("clReleaseEvent", unsafe {
            (self.table.release_event)(event.as_ptr())
        })
    }
}

fn check(operation: &'static str, status: ClInt) -> Result<(), OpenClError> {
    if status == CL_SUCCESS {
        Ok(())
    } else {
        Err(OpenClError::Driver {
            operation,
            code: status,
        })
    }
}

fn non_null(operation: &'static str, raw: *mut c_void) -> Result<*mut c_void, OpenClError> {
    if raw.is_null() {
        Err(OpenClError::Driver {
            operation,
            code: -1,
        })
    } else {
        Ok(raw)
    }
}

fn check_create(
    operation: &'static str,
    status: ClInt,
    raw: *mut c_void,
) -> Result<*mut c_void, OpenClError> {
    check(operation, status)?;
    non_null(operation, raw)
}

fn info_string(
    operation: &'static str,
    mut call: impl FnMut(usize, *mut c_void, *mut usize) -> ClInt,
) -> Result<String, OpenClError> {
    let mut required = 0usize;
    check(operation, call(0, ptr::null_mut(), &mut required))?;
    if required > MAX_INFO_BYTES {
        return Err(OpenClError::InvalidArgument(
            "OpenCL info exceeds bounded size",
        ));
    }
    let mut bytes = vec![0u8; required];
    if required != 0 {
        check(
            operation,
            call(required, bytes.as_mut_ptr().cast(), ptr::null_mut()),
        )?;
    }
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8(bytes).map_err(|_| OpenClError::Utf8)
}

fn resolve_required(
    name: &'static str,
    symbol: &'static [u8],
    mut resolve: impl FnMut(&'static [u8]) -> Option<*mut c_void>,
) -> Result<*mut c_void, OpenClError> {
    debug_assert_eq!(symbol.last(), Some(&0));
    resolve(symbol).ok_or(OpenClError::MissingSymbol(name))
}

struct Library(*mut c_void);
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    fn open() -> Result<Self, OpenClError> {
        #[cfg(target_os = "macos")]
        let candidates = [
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
            "libOpenCL.dylib",
        ];
        #[cfg(all(unix, not(target_os = "macos")))]
        let candidates = ["libOpenCL.so.1", "libOpenCL.so"];
        #[cfg(windows)]
        let candidates = ["OpenCL.dll"];
        Self::open_candidates(&candidates)
    }

    fn open_candidates(candidates: &[&str]) -> Result<Self, OpenClError> {
        let mut detail = String::new();
        for candidate in candidates {
            let name = CString::new(*candidate).expect("library candidate has no NUL");
            let raw = unsafe { platform::open(name.as_ptr()) };
            if !raw.is_null() {
                return Ok(Self(raw));
            }
            detail = platform::last_error();
        }
        Err(OpenClError::LibraryNotFound {
            tried: candidates.iter().map(|name| (*name).into()).collect(),
            detail,
        })
    }

    fn symbol(&self, name: &'static [u8]) -> Result<*mut c_void, OpenClError> {
        debug_assert_eq!(name.last(), Some(&0));
        let raw = unsafe { platform::symbol(self.0, name.as_ptr().cast()) };
        if raw.is_null() {
            let name = CStr::from_bytes_with_nul(name)
                .expect("static symbol is NUL terminated")
                .to_str()
                .expect("static symbol is ASCII");
            Err(OpenClError::MissingSymbol(name))
        } else {
            Ok(raw)
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe { platform::close(self.0) };
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    const RTLD_NOW: c_int = 2;
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }
    pub(super) unsafe fn open(name: *const c_char) -> *mut c_void {
        unsafe { dlopen(name, RTLD_NOW) }
    }
    pub(super) unsafe fn symbol(handle: *mut c_void, name: *const c_char) -> *mut c_void {
        unsafe { dlsym(handle, name) }
    }
    pub(super) unsafe fn close(handle: *mut c_void) {
        let _ = unsafe { dlclose(handle) };
    }
    pub(super) fn last_error() -> String {
        let raw = unsafe { dlerror() };
        if raw.is_null() {
            "dynamic loader returned no detail".into()
        } else {
            unsafe { CStr::from_ptr(raw) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(handle: *mut c_void) -> i32;
    }
    pub(super) unsafe fn open(name: *const c_char) -> *mut c_void {
        unsafe { LoadLibraryA(name) }
    }
    pub(super) unsafe fn symbol(handle: *mut c_void, name: *const c_char) -> *mut c_void {
        unsafe { GetProcAddress(handle, name) }
    }
    pub(super) unsafe fn close(handle: *mut c_void) {
        let _ = unsafe { FreeLibrary(handle) };
    }
    pub(super) fn last_error() -> String {
        "LoadLibrary/GetProcAddress failed".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_error_retains_candidates() {
        let error = Library::open_candidates(&["rustgrad-opencl-library-that-does-not-exist"])
            .err()
            .expect("impossible library name must fail");
        assert!(matches!(
            error,
            OpenClError::LibraryNotFound { tried, .. }
                if tried == vec!["rustgrad-opencl-library-that-does-not-exist"]
        ));
    }

    #[test]
    fn required_symbol_resolver_reports_exact_omission() {
        assert_eq!(
            resolve_required("clCreateBuffer", b"clCreateBuffer\0", |_| None),
            Err(OpenClError::MissingSymbol("clCreateBuffer"))
        );
        let pointer = std::ptr::dangling_mut::<c_void>();
        assert_eq!(
            resolve_required("clCreateBuffer", b"clCreateBuffer\0", |_| Some(pointer)).unwrap(),
            pointer
        );
    }
}
