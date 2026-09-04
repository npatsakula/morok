//! Memory operations: load, store, index, copy, stage.
//!
//! This module contains operations for memory access:
//! - Indexing: index, getaddr, slice
//! - Memory access: load, store (gate is on INDEX, not LOAD/STORE)
//! - Device operations: copy, copy_to_device
//! - Bufferization: stage, stage_global, stage_local
//! - Memory definitions: define_local, define_reg

use std::sync::Arc;

use bon::bon;
use smallvec::SmallVec;
use snafu::ensure;
use svod_dtype::DType;
use svod_dtype::DeviceSpec;

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
    /// The dtype is inferred from the buffer. Shape is carried independently by
    /// the buffer and index sources, never by widening the access dtype.
    ///
    /// # Examples
    /// ```ignore
    /// // Inferred dtype
    /// UOp::index().buffer(buf).indices(vec![idx]).call()?
    ///
    /// // With validity
    /// UOp::index().buffer(buf).indices(vec![idx.valid(gate_uop)]).call()?
    /// ```
    #[builder]
    pub fn index<I: Into<SmallVec<[Arc<Self>; 4]>>>(
        buffer: Arc<Self>,
        indices: I,
        dtype: Option<DType>,
    ) -> Result<Arc<Self>> {
        let indices = indices.into();

        // STACK is a shaped value, so a constant scalar index selects a lane
        // directly rather than constructing a memory INDEX.
        if let Op::Stack { sources } = buffer.op()
            && indices.len() == 1
            && let Op::Const(value) = indices[0].op()
            && let Some(index) = match value.0 {
                crate::ConstValue::Int(index) if index >= 0 => Some(index as usize),
                crate::ConstValue::UInt(index) => usize::try_from(index).ok(),
                _ => None,
            }
            && let Some(source) = sources.get(index)
        {
            return Ok(source.clone());
        }

        // Tinygrad accepts every integer dtype, including weak integers.
        for idx in &indices {
            if Self::is_invalid_marker(idx) {
                continue;
            }
            ensure!(idx.dtype().is_int(), IndexTypeMismatchSnafu { actual: idx.dtype() });
        }

        let op = Op::Index { buffer, indices };
        let inferred = crate::dtype_from_op(&op).expect("INDEX has an inferred dtype");
        let result_dtype = dtype.unwrap_or_else(|| inferred.clone());
        // Tinygrad's INDEX carries exactly the buffer's base dtype (`uop/ops.py:574`);
        // only a weak request may collapse onto it.
        ensure!(
            result_dtype == inferred || (result_dtype.is_weak() && result_dtype.weak_dtype() == inferred.weak_dtype()),
            crate::error::DTypeMismatchSnafu { lhs: inferred, rhs: result_dtype.clone() }
        );
        Ok(Self::new(op, result_dtype))
    }

    /// Index the leading axis at one or more constant positions.
    ///
    /// Multiple positions are represented by one shaped STACK index, exactly as
    /// Tinygrad's `INDEX(value, STACK(CONST...))`.
    pub fn index_axes(self: &Arc<Self>, positions: Vec<usize>) -> Arc<Self> {
        assert!(!positions.is_empty(), "INDEX requires at least one position");
        let index = if positions.len() == 1 {
            Self::index_const(positions[0] as i64)
        } else {
            Self::stack(positions.into_iter().map(|position| Self::index_const(position as i64)).collect())
        };
        Self::index().buffer(self.clone()).indices(vec![index]).call().expect("constant INDEX must be valid")
    }

    /// Lower a storage object to its 64-bit address on `device`.
    ///
    /// Matches Tinygrad's canonical `getaddr`: unsupported values pass through,
    /// while BUFFER/PARAM and their supported storage wrappers produce GETADDR.
    pub fn getaddr(self: &Arc<Self>, device: Option<DeviceSpec>) -> Arc<Self> {
        let mut base = self;
        while let Op::After { passthrough, .. } = base.op() {
            base = passthrough;
        }
        if !matches!(
            base.op(),
            Op::Buffer { .. }
                | Op::Param { .. }
                | Op::Slice { .. }
                | Op::ProgramBinary { .. }
                | Op::MStack { .. }
                | Op::MSelect { .. }
        ) {
            return self.clone();
        }
        let device = device.or_else(|| self.device_spec()).expect("GETADDR requires an explicit or source device");
        Self::new(Op::GetAddr { src: self.clone(), device }, DType::UInt64)
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
    /// // Infer dtype from the address
    /// UOp::load().index(idx).call()
    ///
    /// // With alt value for gated loads
    /// UOp::load().index(idx).alt(zero).gate(gate).call()
    /// ```
    #[builder]
    pub fn load(index: Arc<Self>, dtype: Option<DType>, alt: Option<Arc<Self>>, gate: Option<Arc<Self>>) -> Arc<Self> {
        let inferred = index.dtype();
        let dtype = dtype.unwrap_or_else(|| inferred.clone());
        assert_eq!(dtype, inferred, "LOAD dtype must match INDEX element dtype");
        assert_eq!(alt.is_some(), gate.is_some(), "LOAD requires either index only or index, alt, and gate");
        Self::new(Op::Load { index, alt, gate }, dtype)
    }

    /// Create a STORE operation.
    ///
    /// Stores a value at self (INDEX location).
    /// The buffer is accessed indirectly through the INDEX node.
    ///
    /// For gated stores, use an INDEX with a gate (INDEX has optional gate field).
    pub fn store(self: &Arc<Self>, value: Arc<Self>) -> Arc<Self> {
        Self::new(Op::Store { index: self.clone(), value, gate: None }, DType::Void)
    }

    /// Store a value conditionally at this address.
    pub fn store_gated(self: &Arc<Self>, value: Arc<Self>, gate: Arc<Self>) -> Arc<Self> {
        Self::new(Op::Store { index: self.clone(), value, gate: Some(gate) }, DType::Void)
    }

    // =========================================================================
    // Device Operations
    // =========================================================================

    /// Copy to a different device.
    pub fn copy_to_device(self: &Arc<Self>, device: DeviceSpec) -> Arc<Self> {
        Self::new(Op::Copy { src: self.clone(), device }, self.dtype.clone())
    }

    /// Create a COPY operation with an explicit target device.
    pub fn copy(self: &Arc<Self>, device: DeviceSpec) -> Arc<Self> {
        self.copy_to_device(device)
    }

    // =========================================================================
    // Bufferization Operations
    // =========================================================================

    /// Create a STAGE operation.
    ///
    /// Marks a computation to be materialized into a buffer.
    /// The computation is evaluated over the given ranges and stored.
    pub fn stage(compute: Arc<Self>, ranges: Vec<Arc<Self>>, opts: impl Into<Box<BufferizeOpts>>) -> Arc<Self> {
        let dtype = compute.dtype.clone();
        Self::new(Op::Stage { compute, ranges: SmallVec::from_vec(ranges), opts: opts.into() }, dtype)
    }

    /// Create a STAGE operation with Global address space.
    ///
    /// This is the most common pattern - stage to global memory.
    pub fn stage_global(compute: Arc<Self>, ranges: Vec<Arc<Self>>) -> Arc<Self> {
        Self::stage(
            compute,
            ranges,
            BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Global, removable: true },
        )
    }

    /// Create a STAGE operation with Local address space.
    ///
    /// For shared/local memory bufferization.
    pub fn stage_local(compute: Arc<Self>, ranges: Vec<Arc<Self>>) -> Arc<Self> {
        Self::stage(compute, ranges, BufferizeOpts::local())
    }
}
