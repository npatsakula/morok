//! LLVM JIT compilation via external clang + ELF loader.
//!
//! Compiles LLVM IR text via `clang -x ir -c -O2` stdin→stdout and loads the
//! resulting object via the shared JIT ELF loader. No linked LLVM required.

use crate::Result;
use crate::dispatch::KernelCif;
use tracing::debug;

/// LLVM JIT-compiled kernel using external clang + mmap ELF loader.
pub struct LlvmKernel {
    _mmap: memmap2::MmapMut,
    fn_ptr: *const (),
    entry_point: String,
    name: String,
    var_names: Vec<String>,
    cif: KernelCif,
}

// SAFETY: Function pointer points to read-only compiled code in mmap'd memory.
// Multiple threads can call it concurrently.
unsafe impl Send for LlvmKernel {}
unsafe impl Sync for LlvmKernel {}

impl LlvmKernel {
    /// Compile LLVM IR text to executable code via external clang.
    pub fn compile_ir(
        ir: &str,
        entry_point: impl Into<String>,
        name: impl Into<String>,
        var_names: Vec<String>,
        buf_count: usize,
    ) -> Result<Self> {
        let entry_point = entry_point.into();
        let name = name.into();

        debug!(kernel.name = %name, ir.length = ir.len(), "Compiling LLVM IR via external clang");

        let obj = compile_ir_to_object(ir)?;
        let (fn_ptr, mmap) = crate::jit_loader::jit_load(&obj, &entry_point)?;
        let cif = KernelCif::new(buf_count + var_names.len());

        debug!(kernel.name = %name, "LLVM kernel compiled and loaded");

        Ok(Self { _mmap: mmap, fn_ptr, entry_point, name, var_names, cif })
    }

    /// Compile a RenderedKernel from the codegen crate.
    pub fn compile(kernel: &morok_codegen::RenderedKernel) -> Result<Self> {
        Self::compile_ir(&kernel.code, &kernel.name, &kernel.name, kernel.var_names.clone(), kernel.buffer_args.len())
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

    /// Execute the kernel with buffer pointers and variable values.
    ///
    /// # Safety
    ///
    /// Caller must ensure buffer pointers are valid/aligned and `vals` length
    /// matches `var_names`.
    pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
        debug!(
            kernel.entry_point = %self.entry_point,
            kernel.num_buffers = buffers.len(),
            kernel.num_vals = vals.len(),
            "Executing LLVM kernel"
        );

        unsafe { self.cif.dispatch(self.fn_ptr, buffers, vals, None) };

        Ok(())
    }

    pub(crate) fn cif(&self) -> &KernelCif {
        &self.cif
    }
}

/// Compile LLVM IR text to a relocatable object via `clang -x ir`.
///
/// Uses `--target=<arch>-none-unknown-elf` to produce a relocatable ELF object
/// (same as the C path in jit_loader), so the JIT ELF loader can handle
/// relocations consistently.
fn compile_ir_to_object(ir: &str) -> Result<Vec<u8>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let target = crate::jit_loader::elf_target_triple();

    let mut args = vec![
        "-x",
        "ir",
        "-c",
        "-O2",
        "-march=native",
        "-fPIC",
        "-fno-math-errno",
        "-fno-stack-protector",
        "-funroll-loops",
        "-fvectorize",
        "-fslp-vectorize",
    ];
    args.push(&target);
    args.extend_from_slice(crate::jit_loader::platform_clang_flags());
    args.extend_from_slice(&["-", "-o", "-"]);

    let mut child = Command::new("clang")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| crate::Error::JitCompilation {
            reason: format!("Failed to spawn clang: {e}. Is clang installed?"),
        })?;

    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(ir.as_bytes())
        .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to write IR to clang stdin: {e}") })?;

    let output = child
        .wait_with_output()
        .map_err(|e| crate::Error::JitCompilation { reason: format!("Failed to wait for clang: {e}") })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::Error::JitCompilation { reason: format!("clang IR compilation failed:\n{stderr}") });
    }

    if output.stdout.is_empty() {
        return Err(crate::Error::JitCompilation { reason: "clang produced empty output from IR".to_string() });
    }

    Ok(output.stdout)
}

#[cfg(test)]
#[path = "test/unit/llvm.rs"]
mod tests;
