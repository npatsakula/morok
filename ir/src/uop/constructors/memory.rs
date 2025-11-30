//! Memory operations: load, store, index, copy, bufferize.
//!
//! This module contains operations for memory access:
//! - Indexing: index, index_gated, pointer_index, slice
//! - Memory access: load, load_gated, store, store_gated
//! - Device operations: copy, copy_to_device
//! - Bufferization: bufferize, bufferize_global, bufferize_local
//! - Memory definitions: define_global, define_local, define_reg

use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_dtype::DType;
use smallvec::SmallVec;
use snafu::ensure;

use crate::Result;
use crate::error::IndexTypeMismatchSnafu;
use crate::indexing::IndexSpec;
use crate::op::Op;
use crate::types::{AddrSpace, BufferizeOpts};
use crate::uop::UOp;

impl UOp {
    // =========================================================================
    // Indexing Operations
    // =========================================================================

    /// Create a buffer index operation for multi-dimensional access.
    ///
    /// All indices must have Index dtype. Returns element at specified position.
    pub fn index(buffer: Rc<Self>, indices: Vec<Rc<Self>>) -> Result<Rc<Self>> {
        // Validate that all indices have Index dtype
        for idx in &indices {
            let idx_dtype = idx.dtype();
            ensure!(idx_dtype == DType::Index, IndexTypeMismatchSnafu { actual: idx_dtype });
        }

        let dtype = buffer.dtype.clone();
        let indices = SmallVec::from_vec(indices);
        Ok(Self::new(Op::Index { buffer, indices, gate: None }, dtype))
    }

    /// Create a gated index operation.
    pub fn index_gated(buffer: Rc<Self>, indices: Vec<Rc<Self>>, gate: Rc<Self>) -> Result<Rc<Self>> {
        // Validate that all indices have Index dtype
        for idx in &indices {
            let idx_dtype = idx.dtype();
            ensure!(idx_dtype == DType::Index, IndexTypeMismatchSnafu { actual: idx_dtype });
        }

        let dtype = buffer.dtype.clone();
        let indices = SmallVec::from_vec(indices);
        Ok(Self::new(Op::Index { buffer, indices, gate: Some(gate) }, dtype))
    }

    /// Create a pointer index operation (pointer arithmetic).
    ///
    /// Performs pointer + offset arithmetic for address calculation in kernels.
    /// Both ptr and offset should have Index dtype.
    pub fn pointer_index(ptr: Rc<Self>, offset: Rc<Self>) -> Result<Rc<Self>> {
        let ptr_dtype = ptr.dtype();
        let offset_dtype = offset.dtype();
        ensure!(ptr_dtype == DType::Index, IndexTypeMismatchSnafu { actual: ptr_dtype });
        ensure!(offset_dtype == DType::Index, IndexTypeMismatchSnafu { actual: offset_dtype });
        Ok(Self::new(Op::PointerIndex { ptr, offset }, DType::Index))
    }

    /// Multi-dimensional slicing with IndexSpec.
    ///
    /// **Note**: Range and NewAxis specs are not fully implemented;
    /// currently only Single indices are properly supported.
    pub fn slice(buffer: Rc<Self>, specs: Vec<IndexSpec>) -> Result<Rc<Self>> {
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
            Self::index(buffer, indices)
        }
    }

    /// Gated slicing - conditional access with gate.
    pub fn slice_gated(buffer: Rc<Self>, specs: Vec<IndexSpec>, gate: Rc<Self>) -> Result<Rc<Self>> {
        let mut indices = Vec::new();

        for spec in specs {
            match spec {
                IndexSpec::Single(idx) => indices.push(idx),
                IndexSpec::Range { start, .. } => indices.push(start),
                IndexSpec::Full | IndexSpec::NewAxis => {}
            }
        }

        if indices.is_empty() { Ok(buffer) } else { Self::index_gated(buffer, indices, gate) }
    }

    // =========================================================================
    // Memory Access Operations
    // =========================================================================

    /// Create a LOAD operation.
    ///
    /// Loads a value from a buffer at the given index.
    pub fn load(buffer: Rc<Self>, index: Rc<Self>) -> Rc<Self> {
        let dtype = buffer.dtype.clone();
        Self::new(Op::Load { buffer, index }, dtype)
    }

    /// Create a gated LOAD operation.
    ///
    /// Loads a value from a buffer at the given index, conditionally based on gate.
    /// If gate is false, the load may be skipped or return undefined.
    pub fn load_gated(buffer: Rc<Self>, index: Rc<Self>, gate: Rc<Self>) -> Rc<Self> {
        let dtype = buffer.dtype.clone();
        Self::new(Op::LoadGated { buffer, index, gate }, dtype)
    }

    /// Create a STORE operation.
    ///
    /// Stores a value to a buffer at the given index.
    pub fn store(buffer: Rc<Self>, index: Rc<Self>, value: Rc<Self>) -> Rc<Self> {
        Self::new(Op::Store { buffer, index, value }, DType::Void)
    }

    /// Create a gated STORE operation.
    ///
    /// Stores a value to a buffer at the given index, conditionally based on gate.
    /// If gate is false, the store may be skipped.
    pub fn store_gated(buffer: Rc<Self>, index: Rc<Self>, value: Rc<Self>, gate: Rc<Self>) -> Rc<Self> {
        Self::new(Op::StoreGated { buffer, index, value, gate }, DType::Void)
    }

    // =========================================================================
    // Device Operations
    // =========================================================================

    /// Copy to a different device.
    pub fn copy_to_device(self: &Rc<Self>, device: DeviceSpec) -> Rc<Self> {
        let dev = Self::device(device);
        Self::new(Op::Copy { src: self.clone(), device: dev }, self.dtype.clone())
    }

    /// Create a COPY operation with explicit device UOp.
    ///
    /// Unlike `copy_to_device` which takes a `DeviceSpec`, this takes
    /// a device UOp directly (useful when you already have one).
    pub fn copy(src: Rc<Self>, device: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype.clone();
        Self::new(Op::Copy { src, device }, dtype)
    }

    // =========================================================================
    // Bufferization Operations
    // =========================================================================

    /// Create a BUFFERIZE operation.
    ///
    /// Marks a computation to be materialized into a buffer.
    /// The computation is evaluated over the given ranges and stored.
    pub fn bufferize(compute: Rc<Self>, ranges: Vec<Rc<Self>>, opts: BufferizeOpts) -> Rc<Self> {
        let dtype = compute.dtype.clone();
        Self::new(Op::Bufferize { compute, ranges: SmallVec::from_vec(ranges), opts }, dtype)
    }

    /// Create a BUFFERIZE operation with Global address space.
    ///
    /// This is the most common pattern - bufferize to global memory.
    pub fn bufferize_global(compute: Rc<Self>, ranges: Vec<Rc<Self>>) -> Rc<Self> {
        Self::bufferize(compute, ranges, BufferizeOpts { device: None, addrspace: AddrSpace::Global })
    }

    /// Create a BUFFERIZE operation with Local address space.
    ///
    /// For shared/local memory bufferization.
    pub fn bufferize_local(compute: Rc<Self>, ranges: Vec<Rc<Self>>) -> Rc<Self> {
        Self::bufferize(compute, ranges, BufferizeOpts { device: None, addrspace: AddrSpace::Local })
    }

    // =========================================================================
    // Memory Definition Operations
    // =========================================================================

    /// Create a DEFINE_GLOBAL operation.
    ///
    /// Defines a global memory allocation with the given ID.
    pub fn define_global(id: usize, dtype: DType) -> Rc<Self> {
        Self::new(Op::DefineGlobal(id), dtype)
    }

    /// Create a DEFINE_LOCAL operation.
    ///
    /// Defines a local (shared) memory allocation with the given ID.
    pub fn define_local(id: usize, dtype: DType) -> Rc<Self> {
        Self::new(Op::DefineLocal(id), dtype)
    }

    /// Define register memory.
    pub fn define_reg(size: usize) -> Rc<Self> {
        use morok_dtype::AddrSpace;
        let ptr_dtype = DType::Void.ptr(Some(size), AddrSpace::Reg);
        Self::new(Op::DefineReg { size }, ptr_dtype)
    }
}
