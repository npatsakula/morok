pub mod cast;
pub mod ext;

#[cfg(any(test, feature = "proptest"))]
pub mod test;

/// Device specification parsed from a device string.
///
/// This enum represents different compute devices that can execute kernels.
/// It's used throughout the compilation pipeline for device selection and
/// kernel caching.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceSpec {
    /// CPU device (single-threaded or multi-threaded execution)
    Cpu,
    /// CUDA GPU device with specific device ID
    Cuda { device_id: usize },
    /// Metal GPU device (Apple Silicon) with specific device ID
    Metal { device_id: usize },
    /// WebGPU device (browser or native WebGPU)
    WebGpu,
}

impl DeviceSpec {
    /// Canonicalize the device spec to a standard string representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use morok_dtype::DeviceSpec;
    ///
    /// assert_eq!(DeviceSpec::Cpu.canonicalize(), "CPU");
    /// assert_eq!(DeviceSpec::Cuda { device_id: 0 }.canonicalize(), "CUDA:0");
    /// assert_eq!(DeviceSpec::Cuda { device_id: 1 }.canonicalize(), "CUDA:1");
    /// ```
    pub fn canonicalize(&self) -> String {
        match self {
            DeviceSpec::Cpu => "CPU".to_string(),
            DeviceSpec::Cuda { device_id } => format!("CUDA:{}", device_id),
            DeviceSpec::Metal { device_id } => format!("Metal:{}", device_id),
            DeviceSpec::WebGpu => "WebGPU".to_string(),
        }
    }

    /// Get maximum buffer count for this device.
    ///
    /// Returns None if the device has no buffer limit (effectively unlimited).
    ///
    /// Known limits:
    /// - Metal: 31 buffers (Apple Silicon hardware limit)
    /// - WebGPU: 8 buffers (WebGPU specification limit)
    /// - CPU/CUDA: None (no practical limit)
    pub fn max_buffers(&self) -> Option<usize> {
        match self {
            DeviceSpec::Cpu => None,
            DeviceSpec::Cuda { .. } => Some(128),
            DeviceSpec::Metal { .. } => Some(31),
            DeviceSpec::WebGpu => Some(8),
        }
    }

    /// Get the base device type string (strips device ID).
    ///
    /// Used for device factory lookup and cache key construction.
    /// Unlike `canonicalize()`, this returns a static string without device ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use morok_dtype::DeviceSpec;
    ///
    /// assert_eq!(DeviceSpec::Cpu.base_type(), "CPU");
    /// assert_eq!(DeviceSpec::Cuda { device_id: 0 }.base_type(), "CUDA");
    /// assert_eq!(DeviceSpec::Cuda { device_id: 1 }.base_type(), "CUDA");
    /// ```
    pub fn base_type(&self) -> &'static str {
        match self {
            DeviceSpec::Cpu => "CPU",
            DeviceSpec::Cuda { .. } => "CUDA",
            DeviceSpec::Metal { .. } => "METAL",
            DeviceSpec::WebGpu => "WEBGPU",
        }
    }
}

/// Address space for pointer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AddrSpace {
    /// Global/device memory.
    Global,
    /// Local/shared memory.
    Local,
    /// Register memory.
    Reg,
}

/// Image type kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageKind {
    /// Half precision image.
    Half,
    /// Float precision image.
    Float,
}

/// Scalar data types (base numeric types).
#[derive(Debug, Hash, PartialOrd, Ord)]
#[derive(strum::EnumCount, strum::EnumIter, strum::VariantArray, strum::FromRepr)]
#[derive(enumset::EnumSetType)]
#[cfg_attr(feature = "proptest", derive(proptest_derive::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[enumset(repr = "u32")]
pub enum ScalarDType {
    Bool = 0,

    // Interleaved signed/unsigned for correct LUB priority (lower = more specific)
    Int8 = 1,
    UInt8 = 2,
    Int16 = 3,
    UInt16 = 4,
    Int32 = 5,
    UInt32 = 6,
    Int64 = 7,
    UInt64 = 8,

    FP8E4M3 = 9,
    FP8E5M2 = 10,
    Float16 = 11,
    BFloat16 = 12,
    Float32 = 13,
    Float64 = 14,

    /// Void type for metadata operations (no data).
    Void = 15,

    /// Index type for array indexing and loop iteration.
    Index = 16,
}

/// Data type including scalars, vectors, pointers, and images.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// Scalar type (single value).
    Scalar(ScalarDType),

    /// Vector type (SIMD).
    Vector { scalar: ScalarDType, count: usize },

    /// Pointer type.
    Ptr { base: Box<DType>, addrspace: AddrSpace, size: Option<usize> },

    /// Image type (for texture operations).
    Image { kind: ImageKind, shape: Vec<usize> },
}

impl ScalarDType {
    pub const fn bytes(&self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::UInt8 => 1,
            Self::UInt16 => 2,
            Self::UInt32 => 4,
            Self::UInt64 => 8,
            Self::FP8E4M3 => 1,
            Self::FP8E5M2 => 1,
            Self::Float16 => 2,
            Self::BFloat16 => 2,
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Void => 0,
            Self::Index => 8, // Treat as 64-bit index
        }
    }

    pub const fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }

    pub const fn is_signed(&self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64)
    }

    pub const fn is_unsigned(&self) -> bool {
        matches!(self, Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64)
    }

    pub const fn is_int(&self) -> bool {
        self.is_signed() || self.is_unsigned() || matches!(self, Self::Index)
    }

    pub const fn is_float(&self) -> bool {
        matches!(self, Self::Float16 | Self::Float32 | Self::Float64)
    }

    pub const fn c_style(&self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "signed char",
            Self::Int16 => "short",
            Self::Int32 => "int",
            Self::Int64 => "long",
            Self::UInt8 => "unsigned char",
            Self::UInt16 => "unsigned short",
            Self::UInt32 => "unsigned int",
            Self::UInt64 => "unsigned long",
            Self::FP8E4M3 => "float8_e4m3",
            Self::FP8E5M2 => "float8_e5m2",
            Self::Float16 => "half",
            Self::Float32 => "float",
            Self::Float64 => "double",
            Self::BFloat16 => "__bf16",
            Self::Void => "void",
            Self::Index => "size_t",
        }
    }

    /// Create a vector DType from this scalar type.
    pub const fn vec(self, count: usize) -> DType {
        DType::Vector { scalar: self, count }
    }
}

impl From<ScalarDType> for DType {
    fn from(scalar: ScalarDType) -> Self {
        Self::Scalar(scalar)
    }
}

impl DType {
    // =========================================================================
    // Type Constructors
    // =========================================================================

    /// Create a vector type from this dtype.
    pub fn vec(&self, count: usize) -> Self {
        if count == 1 {
            return self.clone();
        }

        match self {
            Self::Scalar(s) if !matches!(s, ScalarDType::Void) => Self::Vector { scalar: *s, count },
            Self::Vector { .. } => panic!("Cannot vectorize an already vectorized type"),
            _ => self.clone(),
        }
    }

    /// Create a pointer type from this dtype.
    pub fn ptr(self, size: Option<usize>, addrspace: AddrSpace) -> Self {
        match self {
            Self::Ptr { .. } => panic!("Cannot make a pointer from a pointer"),
            _ => Self::Ptr { base: Box::new(self), addrspace, size },
        }
    }

    pub fn scalar(&self) -> Option<ScalarDType> {
        match self {
            Self::Scalar(s) => Some(*s),
            _ => None,
        }
    }

    /// Check if this is a vector type.
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector { .. })
    }

    /// Get the base scalar type (works for both scalars and vectors).
    pub fn base(&self) -> ScalarDType {
        match self {
            Self::Scalar(s) => *s,
            Self::Vector { scalar, .. } => *scalar,
            Self::Ptr { base, .. } => base.base(),
            Self::Image { .. } => ScalarDType::Float32, // Images use float32 by default
        }
    }

    /// Create a new dtype with a different base scalar type, preserving vector count.
    ///
    /// Useful for type conversions like bool→uint8 where the structure is preserved.
    pub fn with_base(&self, new_base: ScalarDType) -> Self {
        let count = self.vcount();
        if count > 1 { Self::Scalar(new_base).vec(count) } else { Self::Scalar(new_base) }
    }

    /// Get the vector count (1 for scalars).
    pub fn count(&self) -> usize {
        match self {
            Self::Vector { count, .. } => *count,
            _ => 1,
        }
    }

    /// Get effective vectorization count (for pointers to vectors).
    pub fn vcount(&self) -> usize {
        match self {
            Self::Vector { count, .. } => *count,
            Self::Ptr { base, .. } => base.count(),
            _ => 1,
        }
    }

    // =========================================================================
    // Type Properties
    // =========================================================================

    pub fn bytes(&self) -> usize {
        match self {
            Self::Scalar(s) => s.bytes(),
            Self::Vector { scalar, count } => scalar.bytes() * count,
            Self::Ptr { .. } => 8,   // Pointers are 64-bit
            Self::Image { .. } => 8, // Image handles are pointers
        }
    }

    pub fn is_bool(&self) -> bool {
        // Use base() to handle both Scalar and Vector types
        self.base() == ScalarDType::Bool
    }

    pub fn is_signed(&self) -> bool {
        // Use base() to handle both Scalar and Vector types
        self.base().is_signed()
    }

    pub fn is_unsigned(&self) -> bool {
        // Use base() to handle both Scalar and Vector types
        self.base().is_unsigned()
    }

    pub fn is_int(&self) -> bool {
        // Use base() to handle both Scalar and Vector types
        self.base().is_int()
    }

    pub fn is_float(&self) -> bool {
        // Use base() to handle both Scalar and Vector types
        self.base().is_float()
    }

    pub fn c_style(&self) -> String {
        match self {
            Self::Scalar(s) => s.c_style().to_string(),
            Self::Vector { scalar, count } => format!("{}[{}]", scalar.c_style(), count),
            Self::Ptr { base, addrspace, .. } => {
                let addr_str = match addrspace {
                    AddrSpace::Global => "__global",
                    AddrSpace::Local => "__local",
                    AddrSpace::Reg => "__register",
                };
                format!("{} {}*", addr_str, base.c_style())
            }
            Self::Image { kind, .. } => match kind {
                ImageKind::Half => "image2d_t".to_string(),
                ImageKind::Float => "image2d_t".to_string(),
            },
        }
    }
}

// Convenient constructors for common scalar types
impl DType {
    pub const fn bool_() -> Self {
        Self::Scalar(ScalarDType::Bool)
    }
    pub const fn int8() -> Self {
        Self::Scalar(ScalarDType::Int8)
    }
    pub const fn int16() -> Self {
        Self::Scalar(ScalarDType::Int16)
    }
    pub const fn int32() -> Self {
        Self::Scalar(ScalarDType::Int32)
    }
    pub const fn int64() -> Self {
        Self::Scalar(ScalarDType::Int64)
    }
    pub const fn uint8() -> Self {
        Self::Scalar(ScalarDType::UInt8)
    }
    pub const fn uint16() -> Self {
        Self::Scalar(ScalarDType::UInt16)
    }
    pub const fn uint32() -> Self {
        Self::Scalar(ScalarDType::UInt32)
    }
    pub const fn uint64() -> Self {
        Self::Scalar(ScalarDType::UInt64)
    }
    pub const fn float16() -> Self {
        Self::Scalar(ScalarDType::Float16)
    }
    pub const fn bfloat16() -> Self {
        Self::Scalar(ScalarDType::BFloat16)
    }
    pub const fn float32() -> Self {
        Self::Scalar(ScalarDType::Float32)
    }
    pub const fn float64() -> Self {
        Self::Scalar(ScalarDType::Float64)
    }
    pub const fn void_() -> Self {
        Self::Scalar(ScalarDType::Void)
    }
    pub const fn index() -> Self {
        Self::Scalar(ScalarDType::Index)
    }
}

// Legacy aliases for compatibility
#[allow(non_upper_case_globals)]
impl DType {
    pub const Bool: Self = Self::Scalar(ScalarDType::Bool);
    pub const Int8: Self = Self::Scalar(ScalarDType::Int8);
    pub const Int16: Self = Self::Scalar(ScalarDType::Int16);
    pub const Int32: Self = Self::Scalar(ScalarDType::Int32);
    pub const Int64: Self = Self::Scalar(ScalarDType::Int64);
    pub const UInt8: Self = Self::Scalar(ScalarDType::UInt8);
    pub const UInt16: Self = Self::Scalar(ScalarDType::UInt16);
    pub const UInt32: Self = Self::Scalar(ScalarDType::UInt32);
    pub const UInt64: Self = Self::Scalar(ScalarDType::UInt64);
    pub const FP8E4M3: Self = Self::Scalar(ScalarDType::FP8E4M3);
    pub const FP8E5M2: Self = Self::Scalar(ScalarDType::FP8E5M2);
    pub const Float16: Self = Self::Scalar(ScalarDType::Float16);
    pub const BFloat16: Self = Self::Scalar(ScalarDType::BFloat16);
    pub const Float32: Self = Self::Scalar(ScalarDType::Float32);
    pub const Float64: Self = Self::Scalar(ScalarDType::Float64);
    pub const Void: Self = Self::Scalar(ScalarDType::Void);
    pub const Index: Self = Self::Scalar(ScalarDType::Index);
}

/// Trait for types that have an associated DType.
///
/// This trait is used for type-safe tensor data extraction (e.g., to_ndarray<T>()).
pub trait HasDType: Clone + Default {
    const DTYPE: DType;
}

impl HasDType for f32 {
    const DTYPE: DType = DType::Float32;
}

impl HasDType for f64 {
    const DTYPE: DType = DType::Float64;
}

impl HasDType for i8 {
    const DTYPE: DType = DType::Int8;
}

impl HasDType for i16 {
    const DTYPE: DType = DType::Int16;
}

impl HasDType for i32 {
    const DTYPE: DType = DType::Int32;
}

impl HasDType for i64 {
    const DTYPE: DType = DType::Int64;
}

impl HasDType for u8 {
    const DTYPE: DType = DType::UInt8;
}

impl HasDType for u16 {
    const DTYPE: DType = DType::UInt16;
}

impl HasDType for u32 {
    const DTYPE: DType = DType::UInt32;
}

impl HasDType for u64 {
    const DTYPE: DType = DType::UInt64;
}

impl HasDType for bool {
    const DTYPE: DType = DType::Bool;
}
