//! Backend renderer capabilities and tensor core configurations.
//!
//! This module defines the interface between the optimizer and backend code generators.
//! It describes what optimizations a backend supports (local memory, threading, etc.)
//! and provides tensor core configurations for hardware-accelerated matrix multiplication.

use morok_dtype::DType;
use smallvec::SmallVec;

/// Tensor core optimization operation.
///
/// Represents a single transformation step when applying tensor cores.
/// Each operation splits a dimension and assigns it to a new axis type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcOpt {
    /// Upcast (vectorize) the specified dimension (0=N, 1=M, 2=K).
    Upcast(usize),
    /// Move the specified dimension to local memory (0=N, 1=M, 2=K).
    Local(usize),
}

impl TcOpt {
    /// Get the dimension index (0=N, 1=M, 2=K).
    pub const fn dim(&self) -> usize {
        match self {
            Self::Upcast(dim) | Self::Local(dim) => *dim,
        }
    }

    /// Returns true if this is an upcast operation.
    pub const fn is_upcast(&self) -> bool {
        matches!(self, Self::Upcast(_))
    }

    /// Returns true if this is a local operation.
    pub const fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }
}

impl std::fmt::Display for TcOpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upcast(dim) => write!(f, "u{}", dim),
            Self::Local(dim) => write!(f, "l{}", dim),
        }
    }
}

/// Swizzle axis specifier for tensor core data layout transformations.
///
/// Describes axis references in swizzle patterns that remap data layouts
/// for optimal tensor core memory access. Unlike TcOpt (operations),
/// SwizzleAxis describes axis identities in the remapping pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwizzleAxis {
    /// Upcast axis with index (0, 1, 2, ...).
    Upcast(usize),
    /// Local axis with index (0, 1, 2, ...).
    Local(usize),
    /// Reduce axis with index (0, 1, 2, ...).
    Reduce(usize),
}

impl std::fmt::Display for SwizzleAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upcast(idx) => write!(f, "u{}", idx),
            Self::Local(idx) => write!(f, "l{}", idx),
            Self::Reduce(idx) => write!(f, "r{}", idx),
        }
    }
}

/// Backend renderer capabilities.
///
/// Describes what features and optimizations a particular backend supports.
/// Used by the optimizer to determine valid transformations and enforce device limits.
#[derive(Debug, Clone)]
pub struct Renderer {
    /// Backend device identifier (e.g., "CUDA", "Metal", "CPU").
    pub device: String,

    /// Whether the backend supports local/shared memory (GPU workgroups).
    pub has_local: bool,

    /// Whether the backend supports shared memory across threads in a workgroup.
    pub has_shared: bool,

    /// Whether the backend supports CPU-style threading (not GPU threads).
    pub has_threads: bool,

    /// Maximum shared memory size in bytes.
    ///
    /// Used to validate GROUP/GROUPTOP optimizations that allocate shared memory.
    /// Typical values: 48KB-96KB for modern GPUs.
    pub shared_max: usize,

    /// Maximum global work dimensions [x, y, z].
    ///
    /// Maximum size for each global thread dimension.
    /// Used to validate thread count in THREAD optimization.
    /// None if unlimited or not applicable.
    pub global_max: Option<Vec<usize>>,

    /// Maximum local work group size.
    ///
    /// Maximum number of threads in a workgroup (product of local dimensions).
    /// Typical values: 256-1024 for GPUs.
    pub local_max: Option<usize>,

    /// Maximum vectorization width (upcast limit).
    ///
    /// Maximum number of elements that can be processed as a vector.
    /// Typical values: 8-16 for SIMD, 4 for GPU float4.
    pub upcast_max: usize,

    /// Maximum number of buffers/arguments per kernel.
    ///
    /// Some backends have limits on kernel arguments.
    /// Metal: 31, WebGPU: 8, CUDA: typically unlimited.
    pub buffer_max: Option<usize>,

    /// Available tensor core configurations.
    ///
    /// Hardware-accelerated matrix multiplication units with specific size constraints.
    /// Empty if tensor cores not available.
    pub tensor_cores: Vec<TensorCore>,
}

impl Renderer {
    /// Create a CPU renderer configuration.
    pub fn cpu() -> Self {
        let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
        Self {
            device: "CPU".to_string(),
            has_local: false,
            has_shared: false,
            has_threads: true,
            shared_max: 0,
            global_max: Some(vec![cores]), // Actual available CPU cores
            local_max: None,
            upcast_max: 16, // AVX512 can do 16-wide float
            buffer_max: None,
            tensor_cores: vec![],
        }
    }

    /// Create a CUDA GPU renderer configuration (SM80/Ampere by default).
    ///
    /// For specific architectures, use `cuda_sm75()`, `cuda_sm80()`, or `cuda_sm89()`.
    pub fn cuda() -> Self {
        Self::cuda_sm80(false) // Default to SM80 (A100) without TF32
    }

    /// Create a CUDA GPU renderer for SM75 (Turing - RTX 20xx, T4).
    pub fn cuda_sm75() -> Self {
        Self {
            device: "CUDA_SM75".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 49152,
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::sm75_tensor_cores(),
        }
    }

    /// Create a CUDA GPU renderer for SM80 (Ampere - A100, RTX 30xx).
    pub fn cuda_sm80(allow_tf32: bool) -> Self {
        Self {
            device: "CUDA_SM80".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 49152,
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::sm80_tensor_cores(allow_tf32),
        }
    }

    /// Create a CUDA GPU renderer for SM89 (Hopper - H100).
    pub fn cuda_sm89(allow_tf32: bool) -> Self {
        Self {
            device: "CUDA_SM89".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 49152,
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::sm89_tensor_cores(allow_tf32),
        }
    }

    /// Create a Metal GPU renderer configuration (Apple M1/M2/M3).
    pub fn metal() -> Self {
        Self {
            device: "Metal".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 32768, // 32KB for Metal
            global_max: None,
            local_max: Some(1024),
            upcast_max: 4,        // float4 for Metal
            buffer_max: Some(31), // Metal has 31 buffer argument limit
            tensor_cores: TensorCore::metal_tensor_cores(),
        }
    }

    /// Create an Apple AMX renderer configuration (M1/M2/M3 matrix coprocessor).
    pub fn apple_amx() -> Self {
        Self {
            device: "AppleAMX".to_string(),
            has_local: false, // AMX doesn't use traditional local memory
            has_shared: false,
            has_threads: true, // CPU-style threading
            shared_max: 0,
            global_max: Some(vec![256]),
            local_max: None,
            upcast_max: 16,
            buffer_max: None,
            tensor_cores: TensorCore::amx_tensor_cores(),
        }
    }

    /// Create an AMD RDNA3 GPU renderer (RX 7000 series).
    pub fn amd_rdna3() -> Self {
        Self {
            device: "AMD_RDNA3".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536, // 64KB for RDNA3
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::rdna3_tensor_cores(),
        }
    }

    /// Create an AMD RDNA4 GPU renderer.
    pub fn amd_rdna4() -> Self {
        Self {
            device: "AMD_RDNA4".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536,
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::rdna4_tensor_cores(),
        }
    }

    /// Create an AMD CDNA3 GPU renderer (MI300 series).
    pub fn amd_cdna3() -> Self {
        Self {
            device: "AMD_CDNA3".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536, // 64KB for CDNA
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::cdna3_tensor_cores(),
        }
    }

    /// Create an AMD CDNA4 GPU renderer.
    pub fn amd_cdna4() -> Self {
        Self {
            device: "AMD_CDNA4".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536,
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(1024),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::cdna4_tensor_cores(),
        }
    }

    /// Create an Intel Xe GPU renderer.
    pub fn intel_xe() -> Self {
        Self {
            device: "IntelXe".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 65536, // 64KB for Xe
            global_max: Some(vec![2147483647, 65535, 65535]),
            local_max: Some(512),
            upcast_max: 8,
            buffer_max: None,
            tensor_cores: TensorCore::intel_tensor_cores(),
        }
    }

    /// Create a WebGPU renderer configuration.
    pub fn webgpu() -> Self {
        Self {
            device: "WebGPU".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            shared_max: 16384, // 16KB typical for WebGPU
            global_max: Some(vec![65535, 65535, 65535]),
            local_max: Some(256),
            upcast_max: 4,
            buffer_max: Some(8), // WebGPU has 8 buffer limit in some implementations
            tensor_cores: vec![],
        }
    }
}

/// Tensor core configuration for hardware-accelerated matrix multiplication.
///
/// Describes a specific matrix multiplication unit with fixed dimensions and data types.
/// Based on NVIDIA's WMMA (Warp Matrix Multiply-Accumulate) API and similar accelerators.
///
/// # Matrix Dimensions
///
/// Tensor cores perform: `C[M,N] += A[M,K] × B[K,N]`
/// - `dims.0` (N): Number of output columns
/// - `dims.1` (M): Number of output rows
/// - `dims.2` (K): Reduction dimension size
///
/// # Example
///
/// NVIDIA Tensor Core 16x16x16:
/// - Processes 16×16 output tile
/// - Accumulates across 16 K elements
/// - Uses 32 threads (warp size)
/// - Each thread handles multiple elements via opts
#[derive(Debug, Clone)]
pub struct TensorCore {
    /// Matrix dimensions (N, M, K).
    pub dims: (usize, usize, usize),

    /// Number of threads required (typically warp size: 32 for CUDA, 64 for AMD).
    pub threads: usize,

    /// Elements per thread in each dimension (N, M, K).
    ///
    /// Describes how the matrix is distributed across threads.
    /// Example: (2, 2, 4) means each thread handles 2×2 output elements
    /// and processes 4 K elements.
    pub elements_per_thread: (usize, usize, usize),

    /// Input matrix data type (A and B matrices).
    pub dtype_in: DType,

    /// Output/accumulator data type (C matrix).
    pub dtype_out: DType,

    /// Optimization sequence for tensor core application.
    ///
    /// A sequence of operations to transform ranges. Each operation splits
    /// a dimension (N, M, or K) and assigns it to a new axis type.
    ///
    /// Example: `[Upcast(0), Local(0), Local(0), Local(1), Local(1), Local(1), Upcast(1)]`
    /// - Upcast N once
    /// - Local split N twice
    /// - Local split M three times
    /// - Upcast M once
    ///
    /// Uses SmallVec to avoid heap allocation for typical tensor cores (≤8 ops).
    pub opts: SmallVec<[TcOpt; 8]>,

    /// Swizzle patterns for input permutation.
    ///
    /// Describes how to permute input matrices to match hardware layout.
    /// Format: ((A_local, A_upcast, A_reduce), (B_local, B_upcast, B_reduce))
    ///
    /// Each tuple contains axis references that describe the permutation pattern
    /// for optimal memory access. The first tuple is for matrix A, second for B.
    ///
    /// Uses SmallVec to avoid heap allocation for typical swizzles (≤8 axes per vec).
    #[allow(clippy::type_complexity)]
    pub swizzle: (
        (SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>),
        (SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>, SmallVec<[SwizzleAxis; 8]>),
    ),
}

// ============================================================================
// TENSOR CORE CONFIGURATION (Static Const Data)
// ============================================================================

/// Static tensor core configuration for const definitions.
///
/// Uses static slices instead of SmallVec for const-compatibility.
/// Use `build()` to convert to runtime `TensorCore`.
pub struct TcConfig {
    dims: (usize, usize, usize),
    threads: usize,
    ept: (usize, usize, usize),
    opts: &'static [TcOpt],
    swizzle_a: (&'static [SwizzleAxis], &'static [SwizzleAxis], &'static [SwizzleAxis]),
    swizzle_b: (&'static [SwizzleAxis], &'static [SwizzleAxis], &'static [SwizzleAxis]),
}

impl TcConfig {
    /// Build a TensorCore from static config with specified dtypes.
    pub fn build(&self, dtype_in: DType, dtype_out: DType) -> TensorCore {
        TensorCore {
            dims: self.dims,
            threads: self.threads,
            elements_per_thread: self.ept,
            dtype_in,
            dtype_out,
            opts: self.opts.iter().copied().collect(),
            swizzle: (
                (
                    self.swizzle_a.0.iter().copied().collect(),
                    self.swizzle_a.1.iter().copied().collect(),
                    self.swizzle_a.2.iter().copied().collect(),
                ),
                (
                    self.swizzle_b.0.iter().copied().collect(),
                    self.swizzle_b.1.iter().copied().collect(),
                    self.swizzle_b.2.iter().copied().collect(),
                ),
            ),
        }
    }
}

// Aliases for brevity in const definitions
use SwizzleAxis::{Local as SL, Reduce as R, Upcast as SU};
use TcOpt::{Local as L, Upcast as U};

// NVIDIA CUDA Tensor Cores
pub const CUDA_81616: TcConfig = TcConfig {
    dims: (8, 16, 16),
    threads: 32,
    ept: (8, 4, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(1), R(2), SL(2), SL(3), SL(4)], &[SU(1), R(3)], &[SL(0), SL(1), SU(0), R(0)]),
    swizzle_b: (&[R(1), R(2), SU(0), SL(0), SL(1)], &[R(0), R(3)], &[SL(2), SL(3), SL(4), SU(1)]),
};

pub const CUDA_81632: TcConfig = TcConfig {
    dims: (8, 16, 32),
    threads: 32,
    ept: (16, 8, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(2), R(3), SL(2), SL(3), SL(4)], &[SU(1), R(4)], &[SL(0), SL(1), SU(0), R(0), R(1)]),
    swizzle_b: (&[R(2), R(3), SU(0), SL(0), SL(1)], &[R(1), R(4)], &[SL(2), SL(3), SL(4), SU(1), R(0)]),
};

pub const CUDA_8168: TcConfig = TcConfig {
    dims: (8, 16, 8),
    threads: 32,
    ept: (4, 2, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(1), R(2), SL(2), SL(3), SL(4)], &[R(0), SU(1)], &[SL(0), SL(1), SU(0)]),
    swizzle_b: (&[R(1), R(2), SU(0), SL(0), SL(1)], &[SU(1), R(0)], &[SL(2), SL(3), SL(4)]),
};

pub const CUDA_8168_TF32: TcConfig = TcConfig {
    dims: (8, 16, 8),
    threads: 32,
    ept: (4, 2, 4),
    opts: &[U(0), L(0), L(0), L(1), L(1), L(1), U(1)],
    swizzle_a: (&[R(0), R(1), SL(2), SL(3), SL(4)], &[SU(1), R(2)], &[SL(0), SL(1), SU(0)]),
    swizzle_b: (&[R(0), R(1), SU(0), SL(0), SL(1)], &[SU(1), R(2)], &[SL(2), SL(3), SL(4)]),
};

// AMD Tensor Cores
pub const AMD_RDNA3: TcConfig = TcConfig {
    dims: (16, 16, 16),
    threads: 32,
    ept: (16, 16, 8),
    opts: &[L(0), L(0), L(0), L(0), L(1), U(1), U(1), U(1)],
    swizzle_a: (&[SL(4), SU(0), SU(1), SU(2), SL(0)], &[R(1), R(2), R(3)], &[SL(1), SL(2), SL(3), R(0)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), SL(4)], &[R(1), R(2), R(3)], &[SU(0), SU(1), SU(2), R(0)]),
};

pub const AMD_RDNA4: TcConfig = TcConfig {
    dims: (16, 16, 16),
    threads: 32,
    ept: (8, 8, 8),
    opts: &[L(0), L(0), L(0), L(0), U(1), U(1), U(1), L(1)],
    swizzle_a: (&[SU(0), SU(1), SU(2), SL(4), R(2)], &[R(0), R(1), R(3)], &[SL(0), SL(1), SL(2), SL(3)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), R(2)], &[R(0), R(1), R(3)], &[SL(4), SU(0), SU(1), SU(2)]),
};

pub const AMD_CDNA_161616: TcConfig = TcConfig {
    dims: (16, 16, 16),
    threads: 64,
    ept: (4, 4, 4),
    opts: &[L(0), L(0), L(0), L(0), U(1), U(1), L(1), L(1)],
    swizzle_a: (&[SU(0), SU(1), SL(4), SL(5), R(2), R(3)], &[R(0), R(1)], &[SL(0), SL(1), SL(2), SL(3)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), R(2), R(3)], &[R(0), R(1)], &[SL(4), SL(5), SU(0), SU(1)]),
};

pub const AMD_CDNA_161632: TcConfig = TcConfig {
    dims: (16, 16, 32),
    threads: 64,
    ept: (8, 8, 4),
    opts: &[L(0), L(0), L(0), L(0), U(1), U(1), L(1), L(1)],
    swizzle_a: (&[SU(0), SU(1), SL(4), SL(5), R(3), R(4)], &[R(0), R(1)], &[SL(0), SL(1), SL(2), SL(3), R(2)]),
    swizzle_b: (&[SL(0), SL(1), SL(2), SL(3), R(3), R(4)], &[R(0), R(1)], &[SL(4), SL(5), SU(0), SU(1), R(2)]),
};

// Apple Metal Tensor Cores
pub const METAL_888: TcConfig = TcConfig {
    dims: (8, 8, 8),
    threads: 32,
    ept: (2, 2, 2),
    opts: &[U(0), L(0), L(1), L(1), L(0), L(1)],
    swizzle_a: (&[R(1), SL(1), SL(2), R(2), SL(4)], &[R(0)], &[SU(0), SL(0), SL(3)]),
    swizzle_b: (&[SL(0), R(0), R(1), SL(3), R(2)], &[SU(0)], &[SL(1), SL(2), SL(4)]),
};

// Apple AMX
pub const APPLE_AMX: TcConfig = TcConfig {
    dims: (64, 64, 1),
    threads: 1,
    ept: (64, 64, 4096),
    opts: &[U(0), U(0), U(0), U(0), U(1), U(1), U(1), U(1)],
    swizzle_a: (&[], &[SU(0), SU(1), SU(2), SU(3), SU(4), SU(5), SU(6), SU(7)], &[]),
    swizzle_b: (&[], &[SU(4), SU(5), SU(6), SU(7), SU(0), SU(1), SU(2), SU(3)], &[]),
};

// Intel Xe Tensor Cores
pub const INTEL_XE_8816: TcConfig = TcConfig {
    dims: (8, 8, 16),
    threads: 8,
    ept: (16, 16, 8),
    opts: &[L(0), L(0), L(0), U(1), U(1), U(1)],
    swizzle_a: (&[R(1), R(2), R(3)], &[SU(0), SU(1), SU(2)], &[SL(0), SL(1), SL(2), R(0)]),
    swizzle_b: (&[SL(0), SL(1), SL(2)], &[R(1), R(2), R(3)], &[SU(0), SU(1), SU(2), R(0)]),
};

impl TensorCore {
    // ===== Helper Methods =====

    /// Get the axes for reduction unrolling.
    ///
    /// Returns pairs of (dimension_index, unroll_amount) for the K dimension.
    /// Used during TC application to unroll the reduction dimension.
    pub fn get_reduce_axes(&self) -> Vec<(usize, usize)> {
        // Typically unroll K dimension by 2 twice (2×2=4 total unroll)
        // This is based on Tinygrad's tensor core implementation
        vec![(0, 2), (1, 2)]
    }

    /// Get the upcast axes configuration for WMMA construction.
    ///
    /// Returns axes configuration for CONTRACT operations.
    /// Format: (A_axes, B_axes, output_axes)
    pub fn upcast_axes(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        // This is simplified - actual implementation depends on opts sequence
        // For 16x16x16 WMMA: each has specific upcast patterns
        (vec![0, 1], vec![0, 1], vec![0, 1])
    }

    // ===== Hardware-Specific Collections =====

    /// Get all tensor cores for NVIDIA SM75 architecture (Turing).
    pub fn sm75_tensor_cores() -> Vec<TensorCore> {
        vec![CUDA_8168.build(DType::Float16, DType::Float32), CUDA_8168.build(DType::Float16, DType::Float16)]
    }

    /// Get all tensor cores for NVIDIA SM80 architecture (Ampere).
    pub fn sm80_tensor_cores(allow_tf32: bool) -> Vec<TensorCore> {
        let mut tcs = vec![
            CUDA_81616.build(DType::Float16, DType::Float32),
            CUDA_81616.build(DType::BFloat16, DType::Float32),
            CUDA_81616.build(DType::Float16, DType::Float16),
            CUDA_8168.build(DType::Float16, DType::Float32),
            CUDA_8168.build(DType::Float16, DType::Float16),
        ];
        if allow_tf32 {
            tcs.push(CUDA_8168_TF32.build(DType::Float32, DType::Float32));
        }
        tcs
    }

    /// Get all tensor cores for NVIDIA SM89 architecture (Hopper).
    pub fn sm89_tensor_cores(allow_tf32: bool) -> Vec<TensorCore> {
        let mut tcs = Self::sm80_tensor_cores(allow_tf32);
        tcs.push(CUDA_81632.build(DType::FP8E4M3, DType::Float32));
        tcs.push(CUDA_81632.build(DType::FP8E5M2, DType::Float32));
        tcs
    }

    /// Get all tensor cores for AMD RDNA3 architecture (RX 7000 series).
    pub fn rdna3_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_RDNA3.build(DType::Float16, DType::Float32),
            AMD_RDNA3.build(DType::Float16, DType::Float16),
            AMD_RDNA3.build(DType::BFloat16, DType::Float32),
        ]
    }

    /// Get all tensor cores for AMD RDNA4 architecture.
    pub fn rdna4_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_RDNA4.build(DType::Float16, DType::Float32),
            AMD_RDNA4.build(DType::Float16, DType::Float16),
            AMD_RDNA4.build(DType::BFloat16, DType::Float32),
            AMD_RDNA4.build(DType::BFloat16, DType::BFloat16),
        ]
    }

    /// Get all tensor cores for AMD CDNA3 architecture (MI300).
    pub fn cdna3_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_CDNA_161632.build(DType::FP8E5M2, DType::Float32),
            AMD_CDNA_161632.build(DType::FP8E4M3, DType::Float32),
            AMD_CDNA_161616.build(DType::Float16, DType::Float32),
            AMD_CDNA_161616.build(DType::BFloat16, DType::Float32),
        ]
    }

    /// Get all tensor cores for AMD CDNA4 architecture.
    pub fn cdna4_tensor_cores() -> Vec<TensorCore> {
        vec![
            AMD_CDNA_161632.build(DType::FP8E5M2, DType::Float32),
            AMD_CDNA_161632.build(DType::FP8E4M3, DType::Float32),
            AMD_CDNA_161632.build(DType::Float16, DType::Float32),
            AMD_CDNA_161632.build(DType::BFloat16, DType::Float32),
            AMD_CDNA_161616.build(DType::Float16, DType::Float32),
            AMD_CDNA_161616.build(DType::BFloat16, DType::Float32),
        ]
    }

    /// Get all tensor cores for Apple Metal (M1/M2/M3).
    pub fn metal_tensor_cores() -> Vec<TensorCore> {
        vec![
            METAL_888.build(DType::Float32, DType::Float32),
            METAL_888.build(DType::Float16, DType::Float32),
            METAL_888.build(DType::Float16, DType::Float16),
            METAL_888.build(DType::BFloat16, DType::Float32),
            METAL_888.build(DType::BFloat16, DType::BFloat16),
        ]
    }

    /// Get all tensor cores for Apple AMX (M1/M2/M3 matrix accelerators).
    pub fn amx_tensor_cores() -> Vec<TensorCore> {
        vec![APPLE_AMX.build(DType::Float32, DType::Float32)]
    }

    /// Get all tensor cores for Intel Xe architecture.
    pub fn intel_tensor_cores() -> Vec<TensorCore> {
        vec![INTEL_XE_8816.build(DType::Float16, DType::Float32)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_cpu() {
        let r = Renderer::cpu();
        assert_eq!(r.device, "CPU");
        assert!(!r.has_local);
        assert!(r.has_threads);
        assert_eq!(r.tensor_cores.len(), 0);
    }

    #[test]
    fn test_renderer_cuda() {
        let r = Renderer::cuda();
        assert_eq!(r.device, "CUDA_SM80"); // Default is SM80/Ampere
        assert!(r.has_local);
        assert!(r.has_shared);
        assert!(!r.has_threads);
        assert!(r.shared_max > 0);
        assert!(!r.tensor_cores.is_empty());
    }

    #[test]
    fn test_tensor_core_cuda() {
        let tc = CUDA_81616.build(DType::Float16, DType::Float32);
        assert_eq!(tc.dims, (8, 16, 16));
        assert_eq!(tc.threads, 32);
        assert_eq!(tc.dtype_in, DType::Float16);
        assert_eq!(tc.dtype_out, DType::Float32);
        assert!(!tc.opts.is_empty());
    }
}
