//! Hardware-specific operations: WMMA, vectorize, kernel.
//!
//! This module contains hardware-specific operations:
//! - Tensor cores: wmma
//! - Vectorization: vectorize, gep, contract, unroll, cat, ptrcat
//! - Multi-device: mstack, mselect
//! - Kernels: kernel

use std::sync::Arc;

use bon::bon;
use morok_dtype::DType;
use smallvec::SmallVec;
use snafu::ensure;

use crate::Result;
use crate::error::{
    BroadcastRequiresScalarSnafu, ContractCountMismatchSnafu, GepIndexOutOfBoundsSnafu, GepRequiresVectorSnafu,
    UnrollCountMismatchSnafu, VectorizeDTypeMismatchSnafu, VectorizeEmptySnafu,
};
use crate::op::Op;
use crate::types::WmmaMetadata;
use crate::uop::UOp;

#[bon]
impl UOp {
    // =========================================================================
    // Tensor Core Operations
    // =========================================================================

    /// Warp Matrix Multiply-Accumulate for tensor cores.
    ///
    /// Computes D = A × B + C using hardware matrix units.
    /// `metadata` specifies dimensions, dtypes, and upcast axes for vectorization.
    pub fn wmma(a: Arc<Self>, b: Arc<Self>, c: Arc<Self>, metadata: WmmaMetadata) -> Arc<Self> {
        let base_dtype = metadata.dtype_out.clone();

        // Calculate vector size from upcast axes (product of all axis sizes)
        let vec_size = metadata.upcast_axes.iter().map(|(_, size)| size).product::<usize>();

        let dtype = if vec_size > 1 { base_dtype.vec(vec_size) } else { base_dtype };

        Self::new(Op::Wmma { a, b, c, metadata }, dtype)
    }

    // =========================================================================
    // Vectorization Operations
    // =========================================================================

    /// Create vector from scalar elements (fallible version with validation).
    ///
    /// # Errors
    /// - `VectorizeRequiresMultiple` if elements is empty
    /// - `VectorizeDTypeMismatch` if elements have different scalar dtypes
    pub fn try_vectorize(elements: SmallVec<[Arc<Self>; 4]>) -> Result<Arc<Self>> {
        ensure!(!elements.is_empty(), VectorizeEmptySnafu);

        // Use full dtype (not scalar_dtype) to preserve Ptr type for pointer vectors.
        // This matches Tinygrad's broadcast: `UOp(Ops.VECTORIZE, self.dtype.vec(count), ...)`
        // For Ptr types: Ptr{vcount:1}.vec(N) → Ptr{vcount:N} (vector of pointers)
        // For Scalar types: Scalar(Float32).vec(N) → Vector{Float32, N}
        let expected_dtype = elements[0].dtype();
        for elem in elements.iter().skip(1) {
            let actual = elem.dtype();
            ensure!(expected_dtype == actual, VectorizeDTypeMismatchSnafu { expected: expected_dtype, actual });
        }

        let vec_dtype = expected_dtype.vec(elements.len());
        Ok(Self::new(Op::Vectorize { elements }, vec_dtype))
    }

    /// Create vector from scalar elements (panics on violation).
    pub fn vectorize(elements: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::try_vectorize(elements).expect("vectorize precondition violated")
    }

    /// Broadcast a scalar value to a vector by replication (fallible version).
    ///
    /// Creates a VECTORIZE operation with `count` copies of the source.
    /// If `count == 1`, returns the source unchanged.
    ///
    /// # Errors
    /// - `BroadcastRequiresScalar` if source has vcount > 1
    pub fn try_broadcast(self: &Arc<Self>, count: usize) -> Result<Arc<Self>> {
        ensure!(self.dtype().vcount() == 1, BroadcastRequiresScalarSnafu { dtype: self.dtype() });

        if count == 1 {
            return Ok(self.clone());
        }
        let elements: SmallVec<[Arc<Self>; 4]> = (0..count).map(|_| self.clone()).collect();
        Ok(Self::vectorize(elements))
    }

    /// Broadcast a scalar value to a vector by replication.
    ///
    /// Creates a VECTORIZE operation with `count` copies of the source.
    /// If `count == 1`, returns the source unchanged.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vector = scalar.broadcast(4);
    /// ```
    pub fn broadcast(self: &Arc<Self>, count: usize) -> Arc<Self> {
        if count == 1 {
            return self.clone();
        }
        let elements: SmallVec<[Arc<Self>; 4]> = (0..count).map(|_| self.clone()).collect();
        Self::vectorize(elements)
    }

    /// Extract element(s) from vector (fallible version with validation).
    ///
    /// # Errors
    /// - `GepRequiresVector` if source has vcount <= 1
    /// - `GepIndexOutOfBounds` if any index >= source vcount
    pub fn try_gep(self: &Arc<Self>, indices: Vec<usize>) -> Result<Arc<Self>> {
        let vector_dtype = self.dtype();
        let vcount = vector_dtype.vcount();

        ensure!(vcount > 1, GepRequiresVectorSnafu { dtype: vector_dtype.clone() });

        for &index in &indices {
            ensure!(index < vcount, GepIndexOutOfBoundsSnafu { index, vcount });
        }

        let dtype = if indices.len() == 1 {
            DType::Scalar(vector_dtype.base())
        } else {
            DType::Scalar(vector_dtype.base()).vec(indices.len())
        };

        Ok(Self::new(Op::Gep { vector: self.clone(), indices }, dtype))
    }

    /// Extract element(s) from vector (Get Element Pointer).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let elem = vector.gep(vec![0]);      // Extract single element
    /// let sub = vector.gep(vec![0, 2]);    // Extract multiple elements
    /// ```
    pub fn gep(self: &Arc<Self>, indices: Vec<usize>) -> Arc<Self> {
        let vector_dtype = self.dtype();
        let dtype = if indices.len() == 1 {
            DType::Scalar(vector_dtype.base())
        } else {
            DType::Scalar(vector_dtype.base()).vec(indices.len())
        };
        Self::new(Op::Gep { vector: self.clone(), indices }, dtype)
    }

    /// Contract unrolled values back into vectorized form (fallible version).
    ///
    /// # Errors
    /// - `ContractCountMismatch` if dtype.vcount != product of axis sizes
    pub fn try_contract(self: &Arc<Self>, upcast_ranges: Vec<(usize, usize)>) -> Result<Arc<Self>> {
        let base_dtype = self.dtype();
        let dtype_count = base_dtype.vcount();
        let axis_product: usize = upcast_ranges.iter().map(|(_, size)| size).product();

        // Only validate if dtype is not void (STORE ops have void dtype)
        if base_dtype != DType::Void {
            ensure!(dtype_count == axis_product, ContractCountMismatchSnafu { dtype_count, axis_product });
        }

        let dtype = if axis_product > 1 { base_dtype.vec(axis_product) } else { base_dtype };

        Ok(Self::new(Op::Contract { src: self.clone(), upcast_ranges }, dtype))
    }

    /// Contract unrolled values back into vectorized form.
    ///
    /// Pairs with UNROLL: UNROLL expands loops for optimization,
    /// CONTRACT combines the results. Used in WMMA and vectorization passes.
    pub fn contract(self: &Arc<Self>, upcast_ranges: Vec<(usize, usize)>) -> Arc<Self> {
        let base_dtype = self.dtype();
        let vec_size = upcast_ranges.iter().map(|(_, size)| size).product::<usize>();
        let dtype = if vec_size > 1 { base_dtype.vec(vec_size) } else { base_dtype };
        Self::new(Op::Contract { src: self.clone(), upcast_ranges }, dtype)
    }

    /// Expand a value across unrolled loop iterations (fallible version).
    ///
    /// # Errors
    /// - `UnrollCountMismatch` if src.dtype.vcount != product of axis sizes
    pub fn try_unroll(self: &Arc<Self>, unroll_axes: Vec<(usize, usize)>) -> Result<Arc<Self>> {
        let dtype = self.dtype();
        let dtype_count = dtype.vcount();
        let axis_product: usize = unroll_axes.iter().map(|(_, size)| size).product();

        // Only validate if we have axes to unroll
        if !unroll_axes.is_empty() {
            ensure!(dtype_count == axis_product, UnrollCountMismatchSnafu { dtype_count, axis_product });
        }

        Ok(Self::new(Op::Unroll { src: self.clone(), unroll_axes }, dtype))
    }

    /// Expand a value across unrolled loop iterations.
    ///
    /// Creates multiple versions of the computation for each unroll axis.
    /// Pairs with CONTRACT which combines results back together.
    pub fn unroll(self: &Arc<Self>, unroll_axes: Vec<(usize, usize)>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Unroll { src: self.clone(), unroll_axes }, dtype)
    }

    /// Create UNROLL with explicit dtype (for do_contract pattern).
    ///
    /// Used when UNROLL dtype should differ from source dtype,
    /// specifically when CONTRACT collapses UNROLL via GEP and
    /// we need to preserve the per-iteration element type.
    ///
    /// Based on Tinygrad's pattern where partial contraction creates
    /// UNROLL with remaining axes but CONTRACT's dtype.
    pub fn unroll_with_dtype(self: &Arc<Self>, unroll_axes: Vec<(usize, usize)>, dtype: DType) -> Arc<Self> {
        Self::new(Op::Unroll { src: self.clone(), unroll_axes }, dtype)
    }

    /// Create a CAT operation (concatenate vectors).
    ///
    /// # Example
    /// ```ignore
    /// // Infer dtype (sum of vcounts)
    /// UOp::cat().sources(vec![a, b]).call()
    ///
    /// // Explicit dtype
    /// UOp::cat().sources(vec![a, b]).dtype(vec8_dtype).call()
    /// ```
    #[builder]
    pub fn cat(sources: Vec<Arc<Self>>, dtype: Option<DType>) -> Arc<Self> {
        assert!(!sources.is_empty(), "CAT requires at least one source");
        let dtype = dtype.unwrap_or_else(|| {
            let total_count: usize = sources.iter().map(|s| s.dtype().vcount()).sum();
            DType::Scalar(sources[0].dtype.base()).vec(total_count)
        });
        Self::new(Op::Cat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Create a PTRCAT operation (concatenate pointers).
    ///
    /// # Example
    /// ```ignore
    /// UOp::ptrcat().sources(vec![a, b]).dtype(ptr_dtype).call()
    /// ```
    #[builder]
    pub fn ptrcat(sources: Vec<Arc<Self>>, dtype: Option<DType>) -> Arc<Self> {
        assert!(!sources.is_empty(), "PTRCAT requires at least one source");
        let dtype = dtype.unwrap_or_else(|| {
            // Compute vcount from total source pointer vcount, matching CAT's approach.
            let total_vcount: usize = sources.iter().map(|s| s.dtype().vcount()).sum();
            let base = &sources[0].dtype;
            match base {
                DType::Ptr { base, addrspace, size, .. } => {
                    DType::Ptr { base: base.clone(), addrspace: *addrspace, size: *size, vcount: total_vcount }
                }
                _ => base.clone(),
            }
        });
        Self::new(Op::PtrCat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    // =========================================================================
    // Multi-Device Operations
    // =========================================================================

    /// Stack multiple buffers (multi-device tensors).
    ///
    /// MStack combines buffers from multiple devices into a single logical tensor.
    /// Used for distributed/multi-GPU tensor operations.
    pub fn mstack(buffers: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        let dtype = buffers.first().map(|b| b.dtype()).unwrap_or(DType::Void);
        Self::new(Op::MStack { buffers }, dtype)
    }

    /// Select buffer by device index (multi-device access).
    ///
    /// MSelect retrieves a specific device's buffer from a multi-device tensor.
    pub fn mselect(self: &Arc<Self>, device_index: usize) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::MSelect { buffer: self.clone(), device_index }, dtype)
    }

    // =========================================================================
    // Kernel Operations
    // =========================================================================

    /// Kernel wrapper.
    ///
    /// Creates a KERNEL operation with the given sources (kernel arguments) and AST (computation).
    ///
    /// # Arguments
    ///
    /// * `sources` - Kernel arguments (buffers and variables)
    /// * `ast` - The computation graph (usually SINK, COPY, or BUFFER_VIEW)
    pub fn kernel(sources: SmallVec<[Arc<Self>; 4]>, ast: Arc<Self>) -> Arc<Self> {
        Self::new(Op::Kernel { sources, ast }, DType::Void)
    }
}
