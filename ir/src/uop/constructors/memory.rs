//! Memory operations: load, store, index, copy, bufferize.
//!
//! This module contains operations for memory access:
//! - Indexing: index, index_gated, pointer_index, slice
//! - Memory access: load, store (gate is on INDEX, not LOAD/STORE)
//! - Device operations: copy, copy_to_device
//! - Bufferization: bufferize, bufferize_global, bufferize_local
//! - Memory definitions: define_global, define_local, define_reg

use std::sync::Arc;

use bon::bon;
use morok_dtype::DType;
use morok_dtype::DeviceSpec;
use smallvec::SmallVec;
use snafu::ensure;

use crate::Result;
use crate::error::IndexTypeMismatchSnafu;
use crate::indexing::IndexSpec;
use crate::op::Op;
use crate::types::{AddrSpace, BufferizeOpts};
use crate::uop::UOp;

#[bon]
impl UOp {
    // =========================================================================
    // Indexing Operations
    // =========================================================================

    /// Create a buffer index operation for multi-dimensional access.
    ///
    /// All indices must have Index dtype.
    ///
    /// # Dtype behavior (matches Tinygrad's `buf.index(idx, ptr=False, dtype=None)`)
    /// - If `dtype` is provided: use it directly (for STORE targets, use Ptr dtype)
    /// - Otherwise: derive element dtype from buffer (default, for LOAD sources)
    ///
    /// # Examples
    /// ```ignore
    /// // Element dtype (default) - for LOAD
    /// UOp::index().buffer(buf).indices(vec![idx]).call()?
    ///
    /// // Explicit Ptr dtype - for STORE
    /// let ptr_dtype = DType::Float32.ptr(Some(size), AddrSpace::Global);
    /// UOp::index().buffer(buf).indices(vec![idx]).dtype(ptr_dtype).call()?
    ///
    /// // With gate
    /// UOp::index().buffer(buf).indices(vec![idx]).gate(gate_uop).call()?
    /// ```
    #[builder]
    pub fn index<I: Into<SmallVec<[Arc<Self>; 4]>>>(
        buffer: Arc<Self>,
        indices: I,
        gate: Option<Arc<Self>>,
        dtype: Option<DType>,
    ) -> Result<Arc<Self>> {
        let indices = indices.into();
        // Validate that all indices have Index dtype
        for idx in &indices {
            let idx_dtype = idx.dtype();
            ensure!(idx_dtype == DType::Index, IndexTypeMismatchSnafu { actual: idx_dtype });
        }

        // Use provided dtype or derive element type from buffer
        // (Tinygrad: `self.dtype.base` when ptr=False, or explicit dtype)
        let result_dtype = dtype.unwrap_or_else(|| match buffer.dtype() {
            DType::Ptr { base, .. } => base.as_ref().clone(),
            other => other,
        });

        Ok(Self::new(Op::Index { buffer, indices, gate }, result_dtype))
    }

    /// Create a pointer index operation (pointer arithmetic).
    ///
    /// Performs pointer + offset arithmetic for address calculation in kernels.
    /// Both self (ptr) and offset should have Index dtype.
    pub fn pointer_index(self: &Arc<Self>, offset: Arc<Self>) -> Result<Arc<Self>> {
        let ptr_dtype = self.dtype();
        let offset_dtype = offset.dtype();
        ensure!(ptr_dtype == DType::Index, IndexTypeMismatchSnafu { actual: ptr_dtype });
        ensure!(offset_dtype == DType::Index, IndexTypeMismatchSnafu { actual: offset_dtype });
        Ok(Self::new(Op::PointerIndex { ptr: self.clone(), offset }, DType::Index))
    }

    /// Multi-dimensional slicing with IndexSpec.
    ///
    /// **Note**: Range and NewAxis specs are not fully implemented;
    /// currently only Single indices are properly supported.
    pub fn slice(buffer: Arc<Self>, specs: Vec<IndexSpec>) -> Result<Arc<Self>> {
        let mut indices = Vec::new();

        for spec in specs {
            match spec {
                IndexSpec::Single(idx) => {
                    // Single index - just use it directly
                    indices.push(idx);
                }
                IndexSpec::Range { start, end: _, step: _ } => {
                    // Range indexing - for now, just use start as a simple index
                    // TODO: Proper range expansion requires loop IR and range operations
                    indices.push(start);
                }
                IndexSpec::Full => {
                    // Full slice - skip (means "all elements")
                    // TODO: Proper handling requires understanding dimension size
                }
                IndexSpec::NewAxis => {
                    // NewAxis - adds dimension
                    // TODO: Requires reshape operation
                }
            }
        }

        if indices.is_empty() {
            // No actual indexing, just return buffer
            Ok(buffer)
        } else {
            Self::index().buffer(buffer).indices(indices).call()
        }
    }

    /// Gated slicing - conditional access with gate.
    pub fn slice_gated(buffer: Arc<Self>, specs: Vec<IndexSpec>, gate: Arc<Self>) -> Result<Arc<Self>> {
        let mut indices = Vec::new();

        for spec in specs {
            match spec {
                IndexSpec::Single(idx) => indices.push(idx),
                IndexSpec::Range { start, .. } => indices.push(start),
                IndexSpec::Full | IndexSpec::NewAxis => {}
            }
        }

        if indices.is_empty() { Ok(buffer) } else { Self::index().buffer(buffer).indices(indices).gate(gate).call() }
    }

    // =========================================================================
    // Index Helpers
    // =========================================================================

    /// Wrap index with validity condition.
    ///
    /// This is the Rust equivalent of Tinygrad's `idx.valid(cond)`.
    /// Creates WHERE(cond, self, Invalid) to mark conditional index validity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Create a conditionally valid index
    /// let valid_idx = idx.valid(cond);
    /// // Equivalent to: WHERE(cond, idx, INVALID)
    /// ```
    pub fn valid(self: &Arc<Self>, cond: Arc<Self>) -> Arc<Self> {
        UOp::try_where(cond, self.clone(), UOp::invalid_marker()).expect("valid: WHERE construction failed")
    }

    // =========================================================================
    // Memory Access Operations
    // =========================================================================

    /// Create a LOAD operation.
    ///
    /// # Example
    /// ```ignore
    /// // Infer dtype from buffer
    /// UOp::load().buffer(buf).index(idx).call()
    ///
    /// // Explicit dtype for vector loads
    /// UOp::load().buffer(buf).index(idx).dtype(vec4_dtype).call()
    ///
    /// // With alt value for gated loads
    /// UOp::load().buffer(buf).index(idx).alt(zero).call()
    /// ```
    #[builder]
    pub fn load(buffer: Arc<Self>, index: Arc<Self>, dtype: Option<DType>, alt: Option<Arc<Self>>) -> Arc<Self> {
        let dtype = dtype.unwrap_or_else(|| match &buffer.dtype {
            DType::Ptr { base, .. } => (**base).clone(),
            other => other.clone(),
        });
        Self::new(Op::Load { buffer, index, alt }, dtype)
    }

    /// Create a STORE operation without ranges.
    ///
    /// Stores a value at self (INDEX location).
    /// The buffer is accessed indirectly through the INDEX node.
    /// For stores with ranges (e.g., output upcasting), use `store_with_ranges`.
    ///
    /// For gated stores, use an INDEX with a gate (INDEX has optional gate field).
    pub fn store(self: &Arc<Self>, value: Arc<Self>) -> Arc<Self> {
        self.store_with_ranges(value, SmallVec::new())
    }

    /// Create a STORE operation with ranges.
    ///
    /// Stores a value at self (INDEX location), with explicit ranges
    /// that define the scope of the store operation. This matches Tinygrad's
    /// architecture where STORE sources are `(index, value, *ranges)`.
    ///
    /// Ranges are used for output upcasting: Range(Upcast) becomes UNROLL
    /// during expansion, which `fix_store_unroll` contracts via CONTRACT.
    ///
    /// For gated stores, use an INDEX with a gate (INDEX has optional gate field).
    pub fn store_with_ranges(self: &Arc<Self>, value: Arc<Self>, ranges: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::Store { index: self.clone(), value, ranges }, DType::Void)
    }

    // =========================================================================
    // Device Operations
    // =========================================================================

    /// Copy to a different device.
    pub fn copy_to_device(self: &Arc<Self>, device: DeviceSpec) -> Arc<Self> {
        let dev = Self::device(device);
        Self::new(Op::Copy { src: self.clone(), device: dev }, self.dtype.clone())
    }

    /// Create a COPY operation with explicit device UOp.
    ///
    /// Unlike `copy_to_device` which takes a `DeviceSpec`, this takes
    /// a device UOp directly (useful when you already have one).
    pub fn copy(self: &Arc<Self>, device: Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype.clone();
        Self::new(Op::Copy { src: self.clone(), device }, dtype)
    }

    // =========================================================================
    // Bufferization Operations
    // =========================================================================

    /// Create a BUFFERIZE operation.
    ///
    /// Marks a computation to be materialized into a buffer.
    /// The computation is evaluated over the given ranges and stored.
    pub fn bufferize(compute: Arc<Self>, ranges: Vec<Arc<Self>>, opts: BufferizeOpts) -> Arc<Self> {
        let dtype = compute.dtype.clone();
        Self::new(Op::Bufferize { compute, ranges: SmallVec::from_vec(ranges), opts }, dtype)
    }

    /// Create a BUFFERIZE operation with Global address space.
    ///
    /// This is the most common pattern - bufferize to global memory.
    pub fn bufferize_global(compute: Arc<Self>, ranges: Vec<Arc<Self>>) -> Arc<Self> {
        Self::bufferize(compute, ranges, BufferizeOpts { device: None, addrspace: AddrSpace::Global })
    }

    /// Create a BUFFERIZE operation with Local address space.
    ///
    /// For shared/local memory bufferization.
    pub fn bufferize_local(compute: Arc<Self>, ranges: Vec<Arc<Self>>) -> Arc<Self> {
        Self::bufferize(compute, ranges, BufferizeOpts { device: None, addrspace: AddrSpace::Local })
    }

    // =========================================================================
    // Memory Definition Operations
    // =========================================================================

    /// Create a DEFINE_GLOBAL operation.
    ///
    /// Defines a global memory allocation with the given ID.
    pub fn define_global(id: usize, dtype: DType) -> Arc<Self> {
        Self::new(Op::DefineGlobal(id), dtype)
    }

    /// Create a DEFINE_LOCAL operation.
    ///
    /// Defines a local (shared) memory allocation with the given ID.
    pub fn define_local(id: usize, dtype: DType) -> Arc<Self> {
        Self::new(Op::DefineLocal(id), dtype)
    }

    /// Define register memory (void pointer - type determined by usage).
    pub fn define_reg(size: usize) -> Arc<Self> {
        use morok_dtype::AddrSpace;
        let ptr_dtype = DType::Void.ptr(Some(size), AddrSpace::Reg);
        Self::new(Op::DefineReg { size }, ptr_dtype)
    }

    /// Define register memory with explicit element type.
    ///
    /// Creates a typed register accumulator for use in reductions.
    /// The element_dtype specifies the type of each element (e.g., Float32 for a float accumulator).
    pub fn define_reg_typed(size: usize, element_dtype: DType) -> Arc<Self> {
        use morok_dtype::AddrSpace;
        let ptr_dtype =
            DType::Ptr { base: Box::new(element_dtype), addrspace: AddrSpace::Reg, size: Some(size), vcount: 1 };
        Self::new(Op::DefineReg { size }, ptr_dtype)
    }
}
