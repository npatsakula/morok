//! Hardware-specific operations: WMMA, vectorize, kernel.
//!
//! This module contains hardware-specific operations:
//! - Tensor cores: wmma
//! - Vectorization: vectorize, gep, contract, unroll, cat, ptrcat
//! - Multi-device: mstack, mselect
//! - Kernels: kernel

use std::rc::Rc;

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
    pub fn wmma(a: Rc<Self>, b: Rc<Self>, c: Rc<Self>, metadata: WmmaMetadata) -> Rc<Self> {
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
    pub fn vectorize(elements: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        let base_dtype = if let Some(first) = elements.first() {
            first.dtype()
        } else {
            DType::Float32 // Default for empty vectors
        };
        let vec_dtype = base_dtype.vec(elements.len());
        Self::new(Op::Vectorize { elements }, vec_dtype)
    }

    /// Get element pointer (extract element(s) from vector).
    pub fn gep(vector: Rc<Self>, indices: Vec<usize>) -> Rc<Self> {
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
    pub fn contract(src: Rc<Self>, upcast_ranges: Vec<(usize, usize)>) -> Rc<Self> {
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
    pub fn unroll(src: Rc<Self>, unroll_axes: Vec<(usize, usize)>) -> Rc<Self> {
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
    pub fn cat(sources: Vec<Rc<Self>>) -> Rc<Self> {
        assert!(!sources.is_empty(), "CAT requires at least one source");
        let dtype = sources[0].dtype.clone();
        Self::new(Op::Cat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Create a PTRCAT operation (concatenate pointers).
    ///
    /// Combines multiple pointer indices into a vectorized pointer.
    /// This is an expander-level operation used in devectorizer for grouping memory accesses.
    pub fn ptrcat(sources: Vec<Rc<Self>>) -> Rc<Self> {
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
    pub fn mstack(buffers: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        let dtype = buffers.first().map(|b| b.dtype()).unwrap_or(DType::Void);
        Self::new(Op::MStack { buffers }, dtype)
    }

    /// Select buffer by device index (multi-device access).
    ///
    /// MSelect retrieves a specific device's buffer from a multi-device tensor.
    pub fn mselect(buffer: Rc<Self>, device_index: usize) -> Rc<Self> {
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
    pub fn kernel(sources: SmallVec<[Rc<Self>; 4]>, ast: Rc<Self>) -> Rc<Self> {
        Self::new(Op::Kernel { sources, ast }, DType::Void)
    }
}
