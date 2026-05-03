//! Clang compilation backend for C codegen.
//!
//! By default, compiles C source via `clang -c` stdin→stdout and loads the
//! resulting object via custom ELF parsing + mmap (no temp files, no dlopen).
//!
//! With `dlopen-fallback` feature: compiles via `clang -shared -O2` and loads
//! the resulting shared library via `dlopen` for kernel execution.

// Default: JIT ELF loader (no temp files, no dlopen)
#[cfg(not(feature = "dlopen-fallback"))]
pub use crate::jit_loader::JitKernel as ClangKernel;

// Fallback: dlopen-based loading
#[cfg(feature = "dlopen-fallback")]
mod dlopen_impl {
    use crate::Result;
    use crate::dispatch::KernelCif;

    /// A compiled C kernel loaded as a shared library.
    pub struct ClangKernel {
        _lib: libloading::Library,
        fn_ptr: *const (),
        name: String,
        var_names: Vec<String>,
        cif: KernelCif,
        _tmp_dir: tempfile::TempDir,
    }

    // SAFETY: The function pointer points to read-only compiled code
    // in the loaded shared library. Multiple threads can call it concurrently.
    unsafe impl Send for ClangKernel {}
    unsafe impl Sync for ClangKernel {}

    impl ClangKernel {
        pub fn compile(src: &str, name: &str, var_names: Vec<String>, buf_count: usize) -> Result<Self> {
            use std::io::Write;

            let tmp_dir = tempfile::tempdir().map_err(|e| crate::Error::JitCompilation {
                reason: format!("Failed to create temp directory: {e}"),
            })?;

            let src_path = tmp_dir.path().join(format!("{name}.c"));
            let so_path = tmp_dir.path().join(format!("{name}.so"));

            let mut src_file = std::fs::File::create(&src_path)
                .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to create source file: {e}") })?;
            src_file
                .write_all(src.as_bytes())
                .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to write source file: {e}") })?;
            drop(src_file);

            let output = std::process::Command::new("clang")
                .args([
                    "-shared",
                    "-O2",
                    "-march=native",
                    "-fPIC",
                    "-fno-math-errno",
                    "-lm",
                    "-o",
                    so_path.to_str().unwrap(),
                    src_path.to_str().unwrap(),
                ])
                .output()
                .map_err(|e| crate::Error::JitCompilation {
                    reason: format!("Failed to run clang: {e}. Is clang installed?"),
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(crate::Error::JitCompilation {
                    reason: format!("clang compilation failed:\n{stderr}\nSource:\n{src}"),
                });
            }

            let lib = unsafe {
                libloading::Library::new(&so_path).map_err(|e| crate::Error::JitCompilation {
                    reason: format!("Failed to load shared library: {e}"),
                })?
            };

            let fn_ptr = unsafe {
                let func: libloading::Symbol<unsafe extern "C" fn()> = lib
                    .get(name.as_bytes())
                    .map_err(|e| crate::Error::FunctionNotFound { name: format!("{name}: {e}") })?;
                *func as *const ()
            };

            let cif = KernelCif::new(buf_count + var_names.len());
            tracing::debug!(kernel.name = %name, "Clang kernel compiled and loaded (dlopen)");

            Ok(Self { _lib: lib, fn_ptr, name: name.to_string(), var_names, cif, _tmp_dir: tmp_dir })
        }

        pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
            unsafe { self.cif.dispatch(self.fn_ptr, buffers, vals, None) };
            Ok(())
        }

        pub(crate) fn cif(&self) -> &KernelCif {
            &self.cif
        }

        pub fn var_names(&self) -> &[String] {
            &self.var_names
        }

        pub fn fn_ptr(&self) -> *const () {
            self.fn_ptr
        }

        pub fn name(&self) -> &str {
            &self.name
        }
    }
}

#[cfg(feature = "dlopen-fallback")]
pub use dlopen_impl::ClangKernel;

#[cfg(test)]
#[path = "test/unit/clang.rs"]
mod tests;
