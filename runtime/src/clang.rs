//! Clang compilation and dynamic loading for C codegen backend.
//!
//! Compiles C source via `clang -shared -O2` and loads the resulting
//! shared library via `dlopen` for kernel execution.

use crate::Result;

/// A compiled C kernel loaded as a shared library.
pub struct ClangKernel {
    /// Keep the library alive (prevents dlclose).
    _lib: libloading::Library,
    /// Raw function pointer to the kernel entry point.
    fn_ptr: *const (),
    /// Kernel name for debugging.
    name: String,
    /// Variable names in order (for populating vars array at runtime).
    var_names: Vec<String>,
    /// Keep the temp directory alive so the .so isn't deleted.
    _tmp_dir: tempfile::TempDir,
}

// SAFETY: The function pointer points to read-only compiled code
// in the loaded shared library. Multiple threads can call it concurrently.
unsafe impl Send for ClangKernel {}
unsafe impl Sync for ClangKernel {}

impl ClangKernel {
    /// Compile C source code via clang and load the resulting shared library.
    pub fn compile(src: &str, name: &str, var_names: Vec<String>) -> Result<Self> {
        use std::io::Write;

        // Create temp directory for source and compiled output
        let tmp_dir = tempfile::tempdir()
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to create temp directory: {}", e) })?;

        let src_path = tmp_dir.path().join(format!("{name}.c"));
        let so_path = tmp_dir.path().join(format!("{name}.so"));

        // Write source file
        let mut src_file = std::fs::File::create(&src_path)
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to create source file: {}", e) })?;
        src_file
            .write_all(src.as_bytes())
            .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to write source file: {}", e) })?;
        drop(src_file);

        // Compile with clang
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
                reason: format!("Failed to run clang: {}. Is clang installed?", e),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::Error::JitCompilation {
                reason: format!("clang compilation failed:\n{stderr}\nSource:\n{src}"),
            });
        }

        // Load the shared library
        let lib = unsafe {
            libloading::Library::new(&so_path)
                .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to load shared library: {}", e) })?
        };

        // Look up the kernel function
        let fn_ptr = unsafe {
            let func: libloading::Symbol<unsafe extern "C" fn(*const *mut u8, *const i64)> =
                lib.get(name.as_bytes())
                    .map_err(|e| crate::Error::FunctionNotFound { name: format!("{name}: {e}") })?;
            *func as *const ()
        };

        tracing::debug!(kernel.name = %name, "Clang kernel compiled and loaded");

        Ok(Self { _lib: lib, fn_ptr, name: name.to_string(), var_names, _tmp_dir: tmp_dir })
    }

    /// Execute the kernel with buffer pointers and variable values.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Buffer pointers are valid and properly aligned
    /// - `vals` has the correct length matching `var_names`
    pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
        tracing::debug!(
            kernel.name = %self.name,
            kernel.num_buffers = buffers.len(),
            kernel.num_vals = vals.len(),
            "Executing Clang kernel"
        );

        type KernelFn = unsafe extern "C" fn(*const *mut u8, *const i64);
        unsafe {
            let f: KernelFn = std::mem::transmute(self.fn_ptr);
            f(buffers.as_ptr(), vals.as_ptr());
        }

        Ok(())
    }

    /// Get the variable names in order.
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Get the raw function pointer.
    pub fn fn_ptr(&self) -> *const () {
        self.fn_ptr
    }

    /// Get the kernel name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clang_kernel_noop() {
        let src = r#"
void test_kernel(void** args, long long* vars) {
}
"#;
        let kernel = ClangKernel::compile(src, "test_kernel", vec![]).unwrap();
        assert_eq!(kernel.name(), "test_kernel");
        unsafe {
            kernel.execute_with_vals(&[], &[]).unwrap();
        }
    }

    #[test]
    fn test_clang_kernel_add() {
        let src = r#"
void add_kernel(void** args, long long* vars) {
    float* a = (float*)args[0];
    float* b = (float*)args[1];
    float* out = (float*)args[2];
    out[0] = a[0] + b[0];
}
"#;
        let kernel = ClangKernel::compile(src, "add_kernel", vec![]).unwrap();

        let mut a = [1.0f32];
        let mut b = [2.0f32];
        let mut out = [0.0f32];

        let buffers = vec![a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];

        unsafe {
            kernel.execute_with_vals(&buffers, &[]).unwrap();
        }

        assert_eq!(out[0], 3.0);
    }
}
