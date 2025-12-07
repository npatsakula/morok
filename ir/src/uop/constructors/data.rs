//! Data creation: constants, buffers, device specifications.
//!
//! This module contains constructors for creating data primitives:
//! - Constants (scalar, native, index)
//! - Buffers (new, view)
//! - Device specifications
//! - No-op and cast operations

use std::sync::Arc;

use morok_dtype::DType;
use morok_dtype::DeviceSpec;
use morok_dtype::ext::HasDType;

use crate::IntoUOp;
use crate::op::Op;
use crate::types::{ConstValue, ConstValueHash};
use crate::uop::core::UOp;
use crate::uop::hash_consing::next_unique_id;

impl UOp {
    // =========================================================================
    // Constants
    // =========================================================================

    /// Create a constant UOp with explicit dtype and value.
    ///
    /// Use `native_const` for type-inferred constants from Rust values.
    pub fn const_(dtype: DType, value: ConstValue) -> Arc<Self> {
        Self::new(Op::Const(ConstValueHash(value)), dtype)
    }

    /// Create a constant UOp from a Rust native value with automatic dtype inference.
    pub fn native_const<T: HasDType + IntoUOp>(value: T) -> Arc<Self> {
        value.into_uop(T::DTYPE)
    }

    /// Create an index constant.
    pub fn index_const(value: i64) -> Arc<Self> {
        Self::const_(DType::Index, ConstValue::Int(value))
    }

    /// Create a vector constant from multiple values.
    ///
    /// Dtype is inferred from the first value; all values must be same type.
    pub fn vconst(values: Vec<ConstValue>) -> Arc<Self> {
        let scalar_dtype = match values.first() {
            Some(ConstValue::Int(_)) => DType::Int64,
            Some(ConstValue::UInt(_)) => DType::UInt64,
            Some(ConstValue::Float(_)) => DType::Float64,
            Some(ConstValue::Bool(_)) => DType::Bool,
            None => DType::Float32,
        };
        let vec_dtype = scalar_dtype.vec(values.len());
        Self::new(Op::VConst { values }, vec_dtype)
    }

    // =========================================================================
    // Buffers
    // =========================================================================

    /// Create a unique buffer identifier.
    pub fn buffer_id(num: Option<usize>) -> Arc<Self> {
        let id = num.unwrap_or_else(next_unique_id);
        Self::new(Op::Unique(id), DType::Void)
    }

    /// Create a new buffer.
    ///
    /// Equivalent to: `UOp(Ops.BUFFER, dtype, (unique(), device(device_spec)), size)`
    pub fn new_buffer(device: DeviceSpec, size: usize, dtype: DType) -> Arc<Self> {
        let unique = Self::buffer_id(None);
        let dev = Self::device(device);
        Self::new(Op::Buffer { unique, device: dev, size }, dtype)
    }

    /// Create a buffer view.
    pub fn buffer_view(buffer: Arc<Self>, size: usize, offset: usize) -> Arc<Self> {
        let dtype = buffer.dtype.clone();
        Self::new(Op::BufferView { buffer, size, offset }, dtype)
    }

    // =========================================================================
    // Device
    // =========================================================================

    /// Create a device specification.
    pub fn device(device: DeviceSpec) -> Arc<Self> {
        Self::new(Op::Device(device), DType::Void)
    }

    // =========================================================================
    // Type Operations
    // =========================================================================

    /// Create a no-op.
    pub fn noop() -> Arc<Self> {
        Self::new(Op::Noop, DType::Void)
    }

    /// Create a cast operation.
    pub fn cast(src: Arc<Self>, dtype: DType) -> Arc<Self> {
        Self::new(Op::Cast { src, dtype: dtype.clone() }, dtype)
    }

    /// Bitcast: reinterpret bits as different type.
    pub fn bitcast(src: Arc<Self>, dtype: DType) -> Arc<Self> {
        Self::new(Op::BitCast { src, dtype: dtype.clone() }, dtype)
    }
}
