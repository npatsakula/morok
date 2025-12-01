//! Device abstraction following Tinygrad's architecture.
//!
//! This module provides a unified Device abstraction that owns:
//! - **Renderer**: Transforms UOp graphs into source code (ProgramSpec)
//! - **Compiler**: Transforms source code into executable bytes
//! - **Runtime**: Creates executable Programs from compiled bytes
//! - **Allocator**: Manages memory allocation for buffers
//!
//! This design allows multiple backends (LLVM, CUDA, Metal, WebGPU) to coexist
//! and share compiled kernels via the method cache.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use morok_dtype::DeviceSpec;

use crate::allocator::Allocator;
use crate::error::Result;

/// A compiled, executable kernel program.
///
/// This trait abstracts over different backend executors (LLVM JIT, CUDA, Metal, etc.).
/// Each backend implements this to provide unified execution interface.
///
/// Note: This trait does not require Send + Sync because some backends (like LLVM JIT)
/// use non-thread-safe types. Programs are typically executed on the same thread where
/// they were compiled, and caching/sharing is handled at a higher level.
pub trait Program {
    /// Execute the kernel with given buffers and variable values.
    ///
    /// # Arguments
    ///
    /// * `buffers` - Raw pointers to buffer data (input and output buffers)
    /// * `vars` - Variable values (for symbolic shapes/strides)
    /// * `global_size` - Global work size (for GPU backends, None for CPU)
    /// * `local_size` - Local work size (for GPU backends, None for CPU)
    ///
    /// # Safety
    ///
    /// This is unsafe because:
    /// - Buffer pointers must be valid and properly aligned
    /// - Buffer sizes must match what the kernel expects
    /// - Caller must ensure no data races during execution
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vars: &HashMap<String, i64>,
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<()>;

    /// Get the kernel name (for debugging/profiling).
    fn name(&self) -> &str;
}

/// A compiler that transforms source code into executable bytes.
///
/// This trait abstracts over different compilation backends:
/// - LLVM: IR -> object code
/// - CUDA: CUDA C -> PTX/CUBIN
/// - Metal: Metal Shading Language -> metallib
/// - WebGPU: WGSL -> SPIR-V
pub trait Compiler: Send + Sync {
    /// Compile source code into executable bytes.
    ///
    /// # Arguments
    ///
    /// * `src` - The source code (LLVM IR, CUDA C, Metal, WGSL, etc.)
    ///
    /// # Returns
    ///
    /// Compiled bytes ready to be loaded by the runtime.
    /// Format depends on backend:
    /// - LLVM: Object code
    /// - CUDA: PTX or CUBIN
    /// - Metal: metallib binary
    /// - WebGPU: SPIR-V binary
    fn compile(&self, src: &str) -> Result<Vec<u8>>;

    /// Optional cache key for this compiler configuration.
    ///
    /// Used to differentiate compiled artifacts when the same device type
    /// can have multiple compiler configurations (e.g., different optimization levels).
    ///
    /// Returns None if all instances of this compiler produce identical output.
    fn cache_key(&self) -> Option<&str> {
        None
    }
}

/// A renderer that transforms UOp graphs into source code.
///
/// This trait abstracts over different code generation backends:
/// - LLVM IR generator
/// - CUDA C generator
/// - Metal Shading Language generator
/// - WGSL generator
pub trait Renderer: Send + Sync {
    /// Render a UOp graph into source code.
    ///
    /// # Arguments
    ///
    /// * `ast` - The kernel AST (UOp graph rooted at KERNEL op)
    ///
    /// # Returns
    ///
    /// A ProgramSpec containing:
    /// - Generated source code
    /// - Entry point name
    /// - Variable list
    /// - Work sizes (for GPU backends)
    fn render(&self, ast: &Rc<morok_ir::UOp>) -> Result<ProgramSpec>;

    /// Get the device spec for this renderer.
    ///
    /// This is used for cache key construction and device selection.
    fn device(&self) -> &DeviceSpec;
}

/// A factory function that creates executable Programs from compiled bytes.
///
/// This is a function pointer that wraps the backend-specific loader:
/// - LLVM: Load object code via JIT
/// - CUDA: cuModuleLoadData + cuModuleGetFunction
/// - Metal: newLibraryWithData + newFunctionWithName
/// - WebGPU: createShaderModule
pub type RuntimeFactory =
    Arc<dyn Fn(&str, &[u8]) -> Result<Box<dyn Program>> + Send + Sync>;

/// A (Renderer, Compiler) pair for a specific backend.
///
/// Devices can have multiple compiler pairs (e.g., different optimization levels).
pub type CompilerPair = (Arc<dyn Renderer>, Arc<dyn Compiler>);

/// A device that owns renderer, compiler, runtime, and allocator.
///
/// This follows Tinygrad's architecture where a Device is a complete
/// compilation + execution unit for a specific backend.
///
/// # Example
///
/// ```ignore
/// let cpu_device = create_cpu_device()?;
/// let spec = cpu_device.renderer.render(&kernel_ast)?;
/// let compiled_bytes = cpu_device.compiler.compile(&spec.src)?;
/// let program = (cpu_device.runtime)(&spec.name, &compiled_bytes)?;
/// unsafe { program.execute(&buffers, &vars, None, None)?; }
/// ```
pub struct Device {
    /// Device specification
    pub device: DeviceSpec,

    /// Memory allocator for this device
    pub allocator: Arc<dyn Allocator>,

    /// Available (renderer, compiler) pairs for this device
    ///
    /// Most devices have one pair, but some may have multiple
    /// (e.g., different optimization levels or compilation modes).
    pub compilers: Vec<CompilerPair>,

    /// Primary renderer for this device
    ///
    /// This is typically compilers[0].0, stored separately for convenience.
    pub renderer: Arc<dyn Renderer>,

    /// Primary compiler for this device
    ///
    /// This is typically compilers[0].1, stored separately for convenience.
    pub compiler: Arc<dyn Compiler>,

    /// Runtime factory for creating executable programs
    ///
    /// Takes (entry_point, compiled_bytes) and returns a Program.
    pub runtime: RuntimeFactory,
}

impl Device {
    /// Create a new device with a single compiler pair.
    ///
    /// This is a convenience constructor for the common case where
    /// a device has only one renderer/compiler combination.
    pub fn new(
        device: DeviceSpec,
        allocator: Arc<dyn Allocator>,
        renderer: Arc<dyn Renderer>,
        compiler: Arc<dyn Compiler>,
        runtime: RuntimeFactory,
    ) -> Self {
        let compilers = vec![(renderer.clone(), compiler.clone())];
        Self { device, allocator, compilers, renderer, compiler, runtime }
    }

    /// Get the base device key (strips device ID).
    ///
    /// Used for compiled byte cache sharing across device instances.
    /// Examples:
    /// - DeviceSpec::Cpu -> "CPU"
    /// - DeviceSpec::Cuda { device_id: 0 } -> "CUDA"
    /// - DeviceSpec::Cuda { device_id: 1 } -> "CUDA"
    /// - DeviceSpec::Metal { device_id: 0 } -> "Metal"
    ///
    /// This allows compiled CUDA kernels to be reused across CUDA:0 and CUDA:1.
    pub fn base_device_key(&self) -> &'static str {
        match &self.device {
            DeviceSpec::Cpu => "CPU",
            DeviceSpec::Cuda { .. } => "CUDA",
            DeviceSpec::Metal { .. } => "Metal",
            DeviceSpec::WebGpu => "WebGPU",
        }
    }
}

/// Program specification containing source code and metadata.
///
/// This is returned by Renderer::render() and consumed by Compiler::compile().
/// It bridges the gap between UOp graphs and compiled executables.
#[derive(Debug, Clone)]
pub struct ProgramSpec {
    /// Kernel name (for debugging/profiling)
    pub name: String,

    /// Generated source code (LLVM IR, CUDA C, Metal, WGSL, etc.)
    pub src: String,

    /// Device specification
    pub device: DeviceSpec,

    /// Original AST (for cache key construction via hash consing)
    pub ast: Rc<morok_ir::UOp>,

    /// Global work size (for GPU backends, None for CPU)
    pub global_size: Option<[usize; 3]>,

    /// Local work size (for GPU backends, None for CPU)
    pub local_size: Option<[usize; 3]>,

    /// Variable list (for symbolic shapes/strides)
    pub vars: Vec<Variable>,
}

impl ProgramSpec {
    /// Create a new program specification.
    pub fn new(name: String, src: String, device: DeviceSpec, ast: Rc<morok_ir::UOp>) -> Self {
        Self {
            name,
            src,
            device,
            ast,
            global_size: None,
            local_size: None,
            vars: Vec::new(),
        }
    }

    /// Add a variable to the program.
    pub fn add_var(&mut self, var: Variable) {
        self.vars.push(var);
    }

    /// Set work sizes for GPU execution.
    pub fn set_work_sizes(&mut self, global: [usize; 3], local: [usize; 3]) {
        self.global_size = Some(global);
        self.local_size = Some(local);
    }
}

/// A variable in the kernel (for symbolic shapes/strides).
///
/// Variables represent symbolic values that are bound at kernel execution time.
/// Examples:
/// - Shape dimensions that vary per input
/// - Stride values computed from shapes
/// - Loop bounds determined by input sizes
#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name (must be unique within the kernel)
    pub name: String,

    /// Minimum value (for range validation)
    pub min: i64,

    /// Maximum value (for range validation)
    pub max: i64,
}

impl Variable {
    /// Create a new variable.
    pub fn new(name: String, min: i64, max: i64) -> Self {
        Self { name, min, max }
    }
}
