//! Types for code generation.

use morok_dtype::DType;

// Re-export new unified types from device crate
pub use morok_device::device::{ProgramSpec, Variable};
pub use morok_dtype::DeviceSpec;

/// A rendered kernel ready for compilation and execution.
#[derive(Debug, Clone)]
pub struct RenderedKernel {
    /// The generated code (LLVM IR, CUDA C, etc.)
    pub code: String,

    /// Kernel name (used as entry point and for debugging/caching).
    pub name: String,

    /// Buffer argument information.
    pub buffer_args: Vec<BufferArg>,

    /// Variable names in order (for populating vars array at runtime).
    /// Includes thread_id at the end if threading is enabled.
    pub var_names: Vec<String>,

    /// Global work size (for GPU backends).
    pub global_size: Option<[usize; 3]>,

    /// Local work size (for GPU backends).
    pub local_size: Option<[usize; 3]>,
}

/// Information about a buffer argument to the kernel.
#[derive(Debug, Clone)]
pub struct BufferArg {
    /// Argument index.
    pub index: usize,

    /// Buffer name.
    pub name: String,

    /// Data type.
    pub dtype: DType,

    /// Whether this is an output buffer.
    pub is_output: bool,
}

impl RenderedKernel {
    /// Create a new rendered kernel.
    pub fn new(code: String, name: String) -> Self {
        Self { code, name, buffer_args: Vec::new(), var_names: Vec::new(), global_size: None, local_size: None }
    }

    /// Add a buffer argument.
    pub fn add_buffer_arg(&mut self, arg: BufferArg) {
        self.buffer_args.push(arg);
    }

    /// Set work sizes for GPU execution.
    pub fn set_work_sizes(&mut self, global: [usize; 3], local: [usize; 3]) {
        self.global_size = Some(global);
        self.local_size = Some(local);
    }
}
