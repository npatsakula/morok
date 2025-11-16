//! UOp constructor methods and operation macros.
//!
//! This module contains all the convenience constructors for creating UOps,
//! as well as macros for generating operation methods.

use std::rc::Rc;

use smallvec::SmallVec;
use snafu::ensure;

use crate::error::{Error, IndexTypeMismatchSnafu, Result};
use crate::op::Op;
use crate::types::*;
use crate::uop::core::UOp;
use crate::uop::hash_consing::next_unique_id;
use morok_device::DeviceSpec;
use morok_dtype::DType;

impl UOp {
    /// Create a constant UOp.
    pub fn const_(dtype: DType, value: ConstValue) -> Rc<Self> {
        Self::new(Op::Const(ConstValueHash(value)), dtype)
    }

    /// Create a unique identifier.
    pub fn unique(num: Option<usize>) -> Rc<Self> {
        let id = num.unwrap_or_else(next_unique_id);
        Self::new(Op::Unique(id), DType::Void)
    }

    /// Create a device specification.
    pub fn device(device: DeviceSpec) -> Rc<Self> {
        Self::new(Op::Device(device), DType::Void)
    }

    /// Create a no-op.
    pub fn noop() -> Rc<Self> {
        Self::new(Op::Noop, DType::Void)
    }

    /// Create a cast operation.
    pub fn cast(src: Rc<Self>, dtype: DType) -> Rc<Self> {
        Self::new(Op::Cast { src, dtype: dtype.clone() }, dtype)
    }

    /// Create a new buffer.
    ///
    /// Equivalent to: `UOp(Ops.BUFFER, dtype, (unique(), device(device_spec)), size)`
    pub fn new_buffer(device: DeviceSpec, size: usize, dtype: DType) -> Rc<Self> {
        let unique = Self::unique(None);
        let dev = Self::device(device);
        Self::new(Op::Buffer { unique, device: dev, size }, dtype)
    }

    /// Create a buffer view.
    pub fn buffer_view(buffer: Rc<Self>, size: usize, offset: usize) -> Rc<Self> {
        let dtype = buffer.dtype.clone();
        Self::new(Op::BufferView { buffer, size, offset }, dtype)
    }

    /// Create an index operation.
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

    /// Copy to a different device.
    pub fn copy_to_device(self: &Rc<Self>, device: DeviceSpec) -> Rc<Self> {
        let dev = Self::device(device);
        Self::new(Op::Copy { src: self.clone(), device: dev }, self.dtype.clone())
    }

    /// Create a sink operation (graph termination).
    ///
    /// Sink marks outputs that must be evaluated. All sources are dependencies.
    pub fn sink(sources: Vec<Rc<Self>>) -> Rc<Self> {
        Self::new(Op::Sink { sources: SmallVec::from_vec(sources) }, DType::Void)
    }

    /// Create a group operation (merging/organizing related ops).
    ///
    /// Group is a NOOP that helps organize related operations together.
    /// It passes through the first source while ensuring all sources are dependencies.
    pub fn group(sources: Vec<Rc<Self>>) -> Rc<Self> {
        let dtype = if sources.is_empty() { DType::Void } else { sources[0].dtype.clone() };
        Self::new(Op::Group { sources: SmallVec::from_vec(sources) }, dtype)
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

    /// Create a CAT operation (concatenate vectors).
    ///
    /// Combines multiple scalar or vector values into a single larger vector.
    /// This is an expander-level operation used during kernel optimization.
    ///
    /// Like VECTORIZE but sources can be vectors themselves.
    /// Output dtype vcount = sum of all input vcounts.
    ///
    /// Example: CAT(vec4, vec2) → vec6
    ///
    /// Note: This operation should be lowered before final codegen.
    pub fn cat(sources: Vec<Rc<Self>>) -> Rc<Self> {
        assert!(!sources.is_empty(), "CAT requires at least one source");
        let dtype = sources[0].dtype.clone();
        Self::new(Op::Cat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Create a PTRCAT operation (concatenate pointers).
    ///
    /// Combines multiple pointer indices into a vectorized pointer.
    /// This is an expander-level operation used in devectorizer for grouping memory accesses.
    ///
    /// Output dtype vcount = sum of all source base counts.
    ///
    /// Note: This operation should be lowered to multiple LOAD/STORE before final codegen.
    pub fn ptrcat(sources: Vec<Rc<Self>>) -> Rc<Self> {
        assert!(!sources.is_empty(), "PTRCAT requires at least one source");
        let dtype = sources[0].dtype.clone();
        Self::new(Op::PtrCat { sources: SmallVec::from_vec(sources) }, dtype)
    }

    /// Create a BUFFERIZE operation.
    ///
    /// Marks a computation to be materialized into a buffer.
    /// The computation is evaluated over the given ranges and stored.
    pub fn bufferize(compute: Rc<Self>, ranges: Vec<Rc<Self>>, opts: BufferizeOpts) -> Rc<Self> {
        let dtype = compute.dtype.clone();
        Self::new(Op::Bufferize { compute, ranges: SmallVec::from_vec(ranges), opts }, dtype)
    }

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
}

// Macro-generated helper methods for arithmetic operations
macro_rules! unary_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(arg: &Rc<Self>) -> Rc<Self> {
                    let dtype = arg.dtype.clone();
                    Self::new(Op::Unary(UnaryOp::$op, arg.clone()), dtype)
                }
            )*
        }
    }
}

macro_rules! cmp_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(lhs: &Rc<Self>, rhs: &Rc<Self>) -> Result<Rc<Self>> {
                    // Use type promotion to validate types and find common type
                    let (lhs, rhs, _) = Self::promote_and_cast(lhs.clone(), rhs.clone())?;
                    Ok(Self::new(Op::Binary(BinaryOp::$op, lhs, rhs), DType::Bool))
                }
            )*
        }
    }
}

// Macro for transcendental functions that require float dtype
macro_rules! transcendental_ops {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl UOp {
            $(
                pub fn $name(arg: &Rc<Self>) -> Result<Rc<Self>> {
                    let dtype = arg.dtype();
                    if !dtype.is_float() {
                        return Err(Error::InvalidDTypeForOp {
                            operation: stringify!($name),
                            dtype: dtype.clone()
                        });
                    }
                    Ok(Self::new(Op::Unary(UnaryOp::$op, arg.clone()), dtype))
                }
            )*
        }
    }
}

// Negation works on any numeric type
unary_ops! {
    neg => Neg,
}

// Transcendental functions require float types
transcendental_ops! {
    sqrt => Sqrt,
    exp2 => Exp2,
    log2 => Log2,
}

cmp_ops! {
    cmplt => Lt,
    cmpeq => Eq,
    cmpne => Ne,
}
