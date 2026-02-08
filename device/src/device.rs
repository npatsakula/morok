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

use std::sync::Arc;

use morok_dtype::DeviceSpec;
use morok_ir::UOp;

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
///
/// # Tinygrad Alignment
///
/// This trait follows Tinygrad's `Program` interface where variable values are
/// passed as a positional tuple/array (`vals`) rather than a named HashMap.
/// The order matches `var_names` in `CompiledSpec`.
pub trait Program {
    /// Execute the kernel with given buffers and variable values.
    ///
    /// # Arguments
    ///
    /// * `buffers` - Raw pointers to buffer data (input and output buffers)
    /// * `vals` - Variable values in positional order (matches `var_names` in CompiledSpec)
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
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<()>;

    /// Get the kernel name (for debugging/profiling).
    fn name(&self) -> &str;
}

/// Compilation result carrying source (JIT) or bytes (AOT).
///
/// Different backends need different information:
/// - LLVM JIT: needs source code to compile during runtime
/// - CUDA: needs PTX/CUBIN bytes to load
/// - Metal: needs metallib bytes to load
///
/// This design allows the RuntimeFactory to access whatever it needs
/// without requiring separate code paths for JIT vs AOT backends.
#[derive(Debug, Clone)]
pub struct CompiledSpec {
    /// Entry point function name
    pub name: String,

    /// Source code (for JIT backends like LLVM)
    /// Set to Some(...) for LLVM JIT, None for AOT backends
    pub src: Option<String>,

    /// Compiled bytes (for AOT backends like CUDA/Metal)
    /// Empty for LLVM JIT, populated for AOT backends
    pub bytes: Vec<u8>,

    /// Original AST for cache key construction via hash consing
    pub ast: Arc<UOp>,

    /// Variable names in order for populating vars array at runtime.
    /// Includes thread_id at the end if threading is enabled.
    pub var_names: Vec<String>,

    /// Global work size for dispatch (GPU backends, CPU threading)
    /// For CPU threading: [thread_count, 1, 1]
    pub global_size: Option<[usize; 3]>,

    /// Local work size for dispatch (GPU backends)
    pub local_size: Option<[usize; 3]>,
}

impl CompiledSpec {
    /// Create a new CompiledSpec for JIT backends (source-based).
    pub fn from_source(name: String, src: String, ast: Arc<UOp>) -> Self {
        Self {
            name,
            src: Some(src),
            bytes: Vec::new(),
            ast,
            var_names: Vec::new(),
            global_size: None,
            local_size: None,
        }
    }

    /// Create a new CompiledSpec for AOT backends (bytecode-based).
    pub fn from_bytes(name: String, bytes: Vec<u8>, ast: Arc<UOp>) -> Self {
        Self { name, src: None, bytes, ast, var_names: Vec::new(), global_size: None, local_size: None }
    }

    /// Create a new CompiledSpec with work sizes for JIT backends.
    pub fn from_source_with_sizes(
        name: String,
        src: String,
        ast: Arc<UOp>,
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Self {
        Self { name, src: Some(src), bytes: Vec::new(), ast, var_names: Vec::new(), global_size, local_size }
    }
}

/// A compiler that transforms source code into a compiled specification.
///
/// This trait abstracts over different compilation backends:
/// - LLVM: IR validation (JIT compiles at runtime)
/// - CUDA: CUDA C -> PTX/CUBIN
/// - Metal: Metal Shading Language -> metallib
/// - WebGPU: WGSL -> SPIR-V
pub trait Compiler: Send + Sync {
    /// Compile a program specification into executable form.
    ///
    /// # Arguments
    ///
    /// * `spec` - The program specification containing source code and metadata
    ///
    /// # Returns
    ///
    /// A CompiledSpec containing:
    /// - For JIT backends (LLVM): source code in `src` field, empty `bytes`
    /// - For AOT backends (CUDA/Metal): compiled bytes in `bytes` field, no `src`
    ///
    /// # Examples
    ///
    /// JIT backend (LLVM):
    /// ```ignore
    /// let compiled = compiler.compile(&spec)?;
    /// assert!(compiled.src.is_some());
    /// assert!(compiled.bytes.is_empty());
    /// ```
    ///
    /// AOT backend (CUDA):
    /// ```ignore
    /// let compiled = compiler.compile(&spec)?;
    /// assert!(compiled.src.is_none());
    /// assert!(!compiled.bytes.is_empty());
    /// ```
    fn compile(&self, spec: &ProgramSpec) -> Result<CompiledSpec>;

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
    fn render(&self, ast: &Arc<UOp>) -> Result<ProgramSpec>;

    /// Get the device spec for this renderer.
    ///
    /// This is used for cache key construction and device selection.
    fn device(&self) -> &DeviceSpec;

    /// Returns decomposition patterns for operations this backend doesn't support.
    ///
    /// This is used by the realization pass to decompose complex operations
    /// into simpler primitives before rendering.
    ///
    /// # Default Implementation
    ///
    /// Returns `None`, meaning no decomposition is needed (backend supports all ops).
    /// Backends that don't support certain operations (e.g., transcendentals)
    /// should override this to return appropriate patterns.
    fn decompositor(&self) -> Option<morok_ir::pattern::TypedPatternMatcher<()>> {
        None
    }
}

/// A factory function that creates executable Programs from a compiled specification.
///
/// This is a function pointer that wraps the backend-specific loader:
/// - LLVM: Extract source from CompiledSpec and JIT compile
/// - CUDA: Extract bytes from CompiledSpec and call cuModuleLoadData + cuModuleGetFunction
/// - Metal: Extract bytes from CompiledSpec and call newLibraryWithData + newFunctionWithName
/// - WebGPU: Extract bytes from CompiledSpec and call createShaderModule
///
/// The CompiledSpec contains either source (for JIT) or bytes (for AOT),
/// allowing each backend to access what it needs.
pub type RuntimeFactory = Arc<dyn Fn(&CompiledSpec) -> Result<Box<dyn Program>> + Send + Sync>;

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
/// let compiled = cpu_device.compiler.compile(&spec)?;
/// let program = (cpu_device.runtime)(&compiled)?;
/// unsafe { program.execute(&buffers, &vals, None, None)?; }
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
        self.device.base_type()
    }
}

/// Program specification containing source code and metadata.
///
/// This is returned by Renderer::render() and consumed by Compiler::compile().
/// It bridges the gap between UOp graphs and compiled executables.
///
/// # Tinygrad Alignment
///
/// Buffer metadata (`globals`, `outs`, `ins`) matches Tinygrad's Program class:
/// - `globals`: Buffer indices from DefineGlobal ops
/// - `outs`: Output buffer indices (written by STORE ops)
/// - `ins`: Input buffer indices (read by LOAD ops)
#[derive(Debug, Clone)]
pub struct ProgramSpec {
    /// Kernel name (for debugging/profiling)
    pub name: String,

    /// Generated source code (LLVM IR, CUDA C, Metal, WGSL, etc.)
    pub src: String,

    /// Device specification
    pub device: DeviceSpec,

    /// Original AST (for cache key construction via hash consing)
    pub ast: Arc<UOp>,

    /// Global work size (for GPU backends, None for CPU)
    pub global_size: Option<[usize; 3]>,

    /// Local work size (for GPU backends, None for CPU)
    pub local_size: Option<[usize; 3]>,

    /// Variable list (for symbolic shapes/strides)
    pub vars: Vec<Variable>,

    /// Variable names in order for populating vars array at runtime.
    /// Includes thread_id at the end if threading is enabled.
    pub var_names: Vec<String>,

    /// Global buffer indices (from DefineGlobal argument values).
    /// Matches Tinygrad's `globals` field.
    pub globals: Vec<usize>,

    /// Output buffer indices (written by STORE ops).
    /// Matches Tinygrad's `outs` field.
    pub outs: Vec<usize>,

    /// Input buffer indices (read by LOAD ops, excluding outputs).
    /// Matches Tinygrad's `ins` field.
    pub ins: Vec<usize>,
}

impl ProgramSpec {
    /// Create a new program specification.
    pub fn new(name: String, src: String, device: DeviceSpec, ast: Arc<UOp>) -> Self {
        Self {
            name,
            src,
            device,
            ast,
            global_size: None,
            local_size: None,
            vars: Vec::new(),
            var_names: Vec::new(),
            globals: Vec::new(),
            outs: Vec::new(),
            ins: Vec::new(),
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

    /// Set variable names for populating vars array at runtime.
    pub fn set_var_names(&mut self, var_names: Vec<String>) {
        self.var_names = var_names;
    }

    /// Set buffer metadata (globals, outs, ins).
    pub fn set_buffer_metadata(&mut self, globals: Vec<usize>, outs: Vec<usize>, ins: Vec<usize>) {
        self.globals = globals;
        self.outs = outs;
        self.ins = ins;
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
