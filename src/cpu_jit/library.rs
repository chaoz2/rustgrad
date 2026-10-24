//! OS-owned executable mappings for compiled C11 kernels.

use super::JitError;
use std::{ffi::c_void, path::Path};

pub(super) struct Library(*mut c_void);
// Libraries remain loaded until their last Arc owner is dropped; symbol lookup
// and immutable generated entry points may be used from multiple threads.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    pub(super) fn open(path: &Path) -> Result<Self, JitError> {
        platform::open(path).map(Self).map_err(JitError::Loader)
    }

    /// The caller must supply a NUL-terminated symbol with exactly the ABI of T
    /// and retain the library for every invocation of the returned pointer.
    pub(super) unsafe fn symbol<T: Copy>(&self, name: &[u8]) -> Result<T, JitError> {
        debug_assert_eq!(name.last(), Some(&0));
        // SAFETY: the owning Library retains the live handle; the caller owns
        // the terminated name and authenticates the generated symbol's ABI.
        let pointer = unsafe { platform::symbol(self.0, name) }.map_err(JitError::Loader)?;
        // SAFETY: T is the function pointer type specified by the caller.
        Ok(unsafe { std::mem::transmute_copy(&pointer) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: this handle is owned once and all borrowed entry points have
        // been retired with the final library owner.
        unsafe { platform::close(self.0) };
    }
}

#[cfg(unix)]
mod platform {
    use std::os::unix::ffi::OsStrExt;
    use std::{
        ffi::{CStr, CString, c_char, c_int, c_void},
        path::Path,
    };

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }

    pub(super) fn open(path: &Path) -> Result<*mut c_void, String> {
        let name = CString::new(path.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: name is terminated and RTLD_NOW requests eager resolution.
        let handle = unsafe { dlopen(name.as_ptr(), 2) };
        if handle.is_null() {
            Err(last_error())
        } else {
            Ok(handle)
        }
    }

    pub(super) unsafe fn symbol(handle: *mut c_void, name: &[u8]) -> Result<*mut c_void, String> {
        // SAFETY: the caller retains the handle and supplies a terminated name.
        let pointer = unsafe { dlsym(handle, name.as_ptr().cast()) };
        if pointer.is_null() {
            Err(last_error())
        } else {
            Ok(pointer)
        }
    }

    pub(super) unsafe fn close(handle: *mut c_void) {
        // SAFETY: the caller transfers its uniquely owned loader reference.
        unsafe { dlclose(handle) };
    }

    fn last_error() -> String {
        // SAFETY: dlerror returns a thread-local terminated string or null.
        unsafe {
            let pointer = dlerror();
            if pointer.is_null() {
                "unknown loader error".into()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        }
    }
}

#[cfg(windows)]
mod platform {
    use std::{
        ffi::{c_char, c_void},
        os::windows::ffi::OsStrExt,
        path::Path,
    };

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(path: *const u16) -> *mut c_void;
        fn GetProcAddress(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(handle: *mut c_void) -> i32;
    }

    pub(super) fn open(path: &Path) -> Result<*mut c_void, String> {
        let mut wide = path.as_os_str().encode_wide().collect::<Vec<_>>();
        if wide.contains(&0) {
            return Err("library path contains NUL".into());
        }
        wide.push(0);
        // SAFETY: wide contains a terminated Windows path and lives through
        // the synchronous load. UTF-16 preserves non-ASCII cache paths.
        let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
        if handle.is_null() {
            Err(std::io::Error::last_os_error().to_string())
        } else {
            Ok(handle)
        }
    }

    pub(super) unsafe fn symbol(handle: *mut c_void, name: &[u8]) -> Result<*mut c_void, String> {
        // SAFETY: the caller retains the module and supplies a terminated name.
        let pointer = unsafe { GetProcAddress(handle, name.as_ptr().cast()) };
        if pointer.is_null() {
            Err(std::io::Error::last_os_error().to_string())
        } else {
            Ok(pointer)
        }
    }

    pub(super) unsafe fn close(handle: *mut c_void) {
        // SAFETY: the caller transfers its uniquely owned loader reference.
        unsafe { FreeLibrary(handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_returns_loader_error() {
        let path =
            std::env::temp_dir().join(format!("rustgrad-missing-library-{}", std::process::id()));
        assert!(matches!(Library::open(&path), Err(JitError::Loader(_))));
    }

    #[cfg(windows)]
    #[test]
    fn windows_library_loads_unicode_path_and_resolves_symbols() {
        use std::{fs, process::Command};
        struct Fixture(std::path::PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let fixture = Fixture(
            std::env::temp_dir().join(format!("rustgrad-windows-loader-{}", std::process::id())),
        );
        fs::create_dir_all(&fixture.0).unwrap();
        let source = fixture.0.join("fixture.c");
        let compiled = fixture.0.join("fixture.dll");
        fs::write(
            &source,
            "int rustgrad_loader_fixture(void) { return 42; }\n",
        )
        .unwrap();
        let output = Command::new(super::super::C11_COMPILER_COMMAND)
            .args(super::super::C11_COMPILER_FLAGS)
            .arg("-o")
            .arg(&compiled)
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let unicode = fixture.0.join("kernel-λ.dll");
        fs::copy(compiled, &unicode).unwrap();
        let library = Library::open(&unicode).unwrap();
        // SAFETY: the fixture exports exactly this C signature and the loaded
        // library outlives both lookup and invocation.
        let call: unsafe extern "C" fn() -> i32 =
            unsafe { library.symbol(b"rustgrad_loader_fixture\0").unwrap() };
        assert_eq!(unsafe { call() }, 42);
        assert!(matches!(
            unsafe { library.symbol::<unsafe extern "C" fn()>(b"missing_symbol\0") },
            Err(JitError::Loader(_))
        ));
        drop(library);
        // Removing a loaded DLL is prohibited on Windows; success proves that
        // the owning loader reference was released before fixture cleanup.
        fs::remove_file(unicode).unwrap();
    }
}
