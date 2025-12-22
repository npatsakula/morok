//! Hardware-specific operations: WMMA, vectorize, kernel.
//!
//! This module contains hardware-specific operations:
//! - Tensor cores: wmma
//! - Vectorization: vectorize, gep, contract, unroll, cat, ptrcat
//! - Multi-device: mstack, mselect
//! - Kernels: kernel

use std::sync::Arc;

use morok_dtype::DType;
use smallvec::SmallVec;

use crate::op::Op;
use crate::types::WmmaMetadata;
use crate::uop::UOp;

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

    /// Create vector from scalar elements.
    pub fn vectorize(elements: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        let base_dtype = if let Some(first) = elements.first() {
            first.dtype()
        } else {
            DType::Float32 // Default for empty vectors
        };
        let vec_dtype = base_dtype.vec(elements.len());
        Self::new(Op::Vectorize { elements }, vec_dtype)
    }

    /// Broadcast a scalar value to a vector by replication.
    ///
    /// Based on Tinygrad's UOp.broadcast() (ops.py:379-382).
    /// Creates a VECTORIZE operation with `count` copies of the source.
    ///
    /// If `count == 1`, returns the source unchanged (optimization).
    ///
    /// # Arguments
    ///
    /// * `src` - The scalar value to broadcast
    /// * `count` - The target vector width
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scalar = UOp::const_(DType::Float32, 5.0);
    /// let vector = UOp::broadcast(scalar, 4);
    /// // Creates VECTORIZE(float32.vec(4), (scalar, scalar, scalar, scalar))
    /// ```
    pub fn broadcast(src: Arc<Self>, count: usize) -> Arc<Self> {
        if count == 1 {
            return src;
        }
        let elements: SmallVec<[Arc<Self>; 4]> = (0..count).map(|_| src.clone()).collect();
        Self::vectorize(elements)
    }

    /// Get element pointer (extract element(s) from vector).
    pub fn gep(vector: Arc<Self>, indices: Vec<usize>) -> Arc<Self> {
        let vector_dtype = vector.dtype();

        // Extract scalar if single element, keep vector if multiple
        let dtype = if indices.len() == 1 {
            // Extract single element -> scalar
            match vector_dtype.scalar() {
                Some(s) => DType::Scalar(s),
                None => vector_dtype.clone(),
            }
        } else {
            // Extract multiple elements -> vector
            match vector_dtype.scalar() {
                Some(s) => DType::Scalar(s).vec(indices.len()),
                None => vector_dtype.clone(),
            }
        };

        Self::new(Op::Gep { vector, indices }, dtype)
    }

    /// Contract unrolled values back into vectorized form.
    ///
    /// Pairs with UNROLL: UNROLL expands loops for optimization,
    /// CONTRACT combines the results. Used in WMMA and vectorization passes.
    pub fn contract(src: Arc<Self>, upcast_ranges: Vec<(usize, usize)>) -> Arc<Self> {
        let base_dtype = src.dtype();

        // Calculate vector size from upcast ranges (product of all range sizes)
        let vec_size = upcast_ranges.iter().map(|(_, size)| size).product::<usize>();

        let dtype = if vec_size > 1 { base_dtype.vec(vec_size) } else { base_dtype };

        Self::new(Op::Contract { src, upcast_ranges }, dtype)
    }

    /// Expand a value across unrolled loop iterations.
    ///
    /// Creates multiple versions of the computation for each unroll axis.
    /// Pairs with CONTRACT which combines results back together.
    pub fn unroll(src: Arc<Self>, unroll_axes: Vec<(usize, usize)>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Unroll { src, unroll_axes }, dtype)
    }

    /// Create a CAT operation (concatenate vectors).
    ///
    /// Combines multiple scalar or vector values into a single larger vector.
    /// This is an expander-level operation used during kernel optimization.
    ///
    /// Like VECTORIZE but sources can be vectors themselves.
    /// Output dtype vcount = sum of all input vcounts.
    pub fn cat(sources: Vec<Arc<Self>>) -> Arc<Self> {
        assert!(!sources.is_empty(), "CAT requires at least one source");
        let total_count: usize = sources.iter().map(|s| s.dtype().vcount()).sum();
        let dtype = if let Some(scalar) = sources[0].dtype.scalar() {
            DType::Scalar(scalar).vec(total_count)
        } else {
            sources[0].dtype.clone()
        };
        Self::new(Op::Cat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Create a PTRCAT operation (concatenate pointers).
    ///
    /// Combines multiple pointer indices into a vectorized pointer.
    /// This is an expander-level operation used in devectorizer for grouping memory accesses.
    pub fn ptrcat(sources: Vec<Arc<Self>>) -> Arc<Self> {
        assert!(!sources.is_empty(), "PTRCAT requires at least one source");
        let dtype = sources[0].dtype.clone();
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
    pub fn mselect(buffer: Arc<Self>, device_index: usize) -> Arc<Self> {
        let dtype = buffer.dtype();
        Self::new(Op::MSelect { buffer, device_index }, dtype)
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
