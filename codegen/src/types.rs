//! Types for code generation.

use morok_dtype::DType;

/// A rendered kernel ready for compilation and execution.
#[derive(Debug, Clone)]
pub struct RenderedKernel {
    /// The generated code (LLVM IR, CUDA C, etc.)
    pub code: String,

    /// Entry point function name.
    pub entry_point: String,

    /// Kernel name (for debugging/caching).
    pub name: String,

    /// Buffer argument information.
    pub buffer_args: Vec<BufferArg>,

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
    pub fn new(code: String, entry_point: String, name: String) -> Self {
        Self {
            code,
            entry_point,
            name,
            buffer_args: Vec::new(),
            global_size: None,
            local_size: None,
        }
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
