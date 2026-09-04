//! Hardware-specific operations: WMMA, lane packing, callable kernels/programs.
//!
//! This module contains hardware-specific operations:
//! - Tensor cores: wmma
//! - Vectorization: stack and index helpers
//! - Multi-device: mstack, mselect
//! - Callable/program IR: call, program

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use snafu::{OptionExt, ensure};
use svod_dtype::DType;

use crate::Result;
use crate::error::{BroadcastRequiresScalarSnafu, GetTupleIndexOutOfBoundsSnafu, GetTupleNotATupleSnafu};
use crate::op::Op;
use crate::ops;
use crate::types::{CallInfo, WmmaMetadata};
use crate::uop::UOp;

impl UOp {
    // =========================================================================
    // Tensor Core Operations
    // =========================================================================

    /// Warp Matrix Multiply-Accumulate for tensor cores.
    ///
    /// Computes D = A × B + C using hardware matrix units.
    /// `metadata` specifies dimensions, dtypes, and upcast axes for vectorization.
    pub fn wmma(a: Arc<Self>, b: Arc<Self>, c: Arc<Self>, metadata: impl Into<Box<WmmaMetadata>>) -> Arc<Self> {
        let dtype = c.dtype();
        Self::new(Op::Wmma(ops::Wmma { a, b, c, metadata: metadata.into() }), dtype)
    }

    // =========================================================================
    // Vectorization Operations
    // =========================================================================

    /// Broadcast a scalar value along a new leading axis (fallible version).
    ///
    /// Creates a STACK operation with `count` copies of the source.
    /// If `count == 1`, returns the source unchanged.
    ///
    /// # Errors
    /// - `BroadcastRequiresScalar` if source has a vector dtype
    pub fn try_broadcast(self: &Arc<Self>, count: usize) -> Result<Arc<Self>> {
        ensure!(self.dtype().vcount() == 1, BroadcastRequiresScalarSnafu { dtype: self.dtype() });

        if count == 1 {
            return Ok(self.clone());
        }
        let elements: SmallVec<[Arc<Self>; 4]> = (0..count).map(|_| self.clone()).collect();
        Ok(Self::stack(elements))
    }

    /// Broadcast a scalar value along a new leading axis.
    ///
    /// Creates a STACK operation with `count` copies of the source.
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
        Self::stack(elements)
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
        Self::new(Op::MStack(ops::MStack { buffers }), dtype)
    }

    /// Select buffer by device index (multi-device access).
    ///
    /// MSelect retrieves a specific device's buffer from a multi-device tensor.
    pub fn mselect(self: &Arc<Self>, device_index: usize) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::MSelect(ops::MSelect { buffer: self.clone(), device_index }), dtype)
    }

    // =========================================================================
    // Callable Operations
    // =========================================================================

    /// Callable wrapper around a body UOp and runtime arguments.
    ///
    /// CALL dtype is always void per tinygrad's spec.
    pub fn call(self: &Arc<Self>, args: SmallVec<[Arc<Self>; 4]>, info: impl Into<Box<CallInfo>>) -> Arc<Self> {
        Self::new(Op::Call(ops::Call { body: self.clone(), args, info: info.into() }), DType::Void)
    }

    /// Typed instruction-style CALL. Its result is scalar and its body remains opaque.
    pub fn call_typed(
        self: &Arc<Self>,
        args: SmallVec<[Arc<Self>; 4]>,
        info: CallInfo,
        return_dtype: DType,
    ) -> Arc<Self> {
        Self::new(Op::Call(ops::Call { body: self.clone(), args, info: info.into() }), return_dtype)
    }

    /// FUNCTION wrapper around a value-producing body UOp and runtime arguments.
    ///
    /// FUNCTION dtype is always void per tinygrad's spec, and its body is
    /// always a TUPLE; non-Tuple bodies are auto-wrapped.
    /// For opaque bodies (SINK / PROGRAM / COPY / SLICE / CUSTOM_FUNCTION) prefer
    /// `.call()` instead — those mirror tinygrad's `_OPAQUE_CALL_BODIES` set.
    pub fn function(self: &Arc<Self>, args: SmallVec<[Arc<Self>; 4]>, info: impl Into<Box<CallInfo>>) -> Arc<Self> {
        let body = if matches!(self.op(), Op::Tuple(..)) { self.clone() } else { self.maketuple() };
        Self::new(Op::Function(ops::Function { body, args, info: info.into() }), DType::Void)
    }

    /// Fallible FUNCTION constructor with positional formal/actual validation.
    pub fn try_function(self: &Arc<Self>, args: SmallVec<[Arc<Self>; 4]>, info: CallInfo) -> Result<Arc<Self>> {
        let body = if matches!(self.op(), Op::Tuple(..)) { self.clone() } else { self.maketuple() };
        crate::shape::function_param_substitutions(&body, &args)?;
        Ok(Self::new(Op::Function(ops::Function { body, args, info: info.into() }), DType::Void))
    }

    /// Construct a TUPLE from value-producing UOps. dtype is always void.
    /// Mirrors tinygrad `Ops.TUPLE`.
    pub fn tuple(srcs: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::Tuple(ops::Tuple { src: srcs }), DType::Void)
    }

    /// Wrap `self` in a single-element TUPLE. Mirrors tinygrad `UOp.maketuple(self)`.
    pub fn maketuple(self: &Arc<Self>) -> Arc<Self> {
        Self::tuple(smallvec![self.clone()])
    }

    /// Extract element `index` from a TUPLE (or a FUNCTION whose body is a TUPLE).
    /// dtype matches the inner element. Mirrors tinygrad `Ops.GETTUPLE`.
    ///
    /// # Errors
    /// - `GetTupleNotATuple` if `self` is neither a TUPLE nor a FUNCTION whose body is a TUPLE
    /// - `GetTupleIndexOutOfBounds` if `index` is out of bounds for the tuple
    pub fn try_gettuple(self: &Arc<Self>, index: usize) -> Result<Arc<Self>> {
        let inner_tuple_src: &SmallVec<[Arc<UOp>; 4]> = match self.op() {
            Op::Tuple(ops::Tuple { src }) => src,
            Op::Function(ops::Function { body, .. }) => match body.op() {
                Op::Tuple(ops::Tuple { src }) => src,
                _ => return GetTupleNotATupleSnafu { op: "FUNCTION body (expected TUPLE)" }.fail(),
            },
            _ => return GetTupleNotATupleSnafu { op: "non-TUPLE/non-FUNCTION source" }.fail(),
        };
        let elem_dtype = inner_tuple_src
            .get(index)
            .context(GetTupleIndexOutOfBoundsSnafu { index, len: inner_tuple_src.len(), kind: "tuple" })?
            .dtype();
        Ok(Self::new(Op::GetTuple(ops::GetTuple { src: self.clone(), index }), elem_dtype))
    }

    /// Extract element `index` from a TUPLE (or a FUNCTION whose body is a TUPLE).
    ///
    /// Panicking wrapper around [`Self::try_gettuple`]; use the fallible variant
    /// when the source structure or index is not guaranteed by construction.
    pub fn gettuple(self: &Arc<Self>, index: usize) -> Arc<Self> {
        self.try_gettuple(index).expect("gettuple precondition violated")
    }

    /// PROGRAM wrapper with optional progressive pipeline stages.
    pub fn program(
        sink: Arc<Self>,
        info: impl Into<Box<crate::ProgramInfo>>,
        linear: Option<Arc<Self>>,
        source: Option<Arc<Self>>,
        binary: Option<Arc<Self>>,
    ) -> Arc<Self> {
        Self::new(Op::Program(ops::Program { sink, info: info.into(), linear, source, binary }), DType::Void)
    }

    /// LINEAR stage payload.
    pub fn linear(ops: SmallVec<[Arc<Self>; 8]>) -> Arc<Self> {
        Self::new(Op::Linear(ops::Linear { ops }), DType::Void)
    }

    /// SOURCE stage payload.
    pub fn source(code: String) -> Arc<Self> {
        Self::new(Op::Source(ops::Source { code, identity: None }), DType::Void)
    }

    /// SOURCE stage payload bound to an executable PROGRAM identity.
    pub fn source_with_identity(code: String, identity: crate::SourceStageIdentity) -> Arc<Self> {
        Self::new(Op::Source(ops::Source { code, identity: Some(identity.into()) }), DType::Void)
    }

    /// BINARY stage payload.
    pub fn binary(bytes: Vec<u8>) -> Arc<Self> {
        Self::new(Op::ProgramBinary(ops::ProgramBinary { bytes, identity: None }), DType::UInt8)
    }

    /// BINARY stage payload bound to its exact SOURCE and compiler identity.
    pub fn binary_with_identity(bytes: Vec<u8>, identity: crate::BinaryStageIdentity) -> Arc<Self> {
        Self::new(Op::ProgramBinary(ops::ProgramBinary { bytes, identity: Some(identity.into()) }), DType::UInt8)
    }

    /// Construct a target instruction. INS has no inferred dtype because an
    /// instruction may define a value of any target type or be void.
    pub fn ins(sources: impl IntoIterator<Item = Arc<Self>>, dtype: DType, arg: crate::InsArg) -> Arc<Self> {
        Self::new(Op::Ins(ops::Ins { sources: sources.into_iter().collect(), arg }), dtype)
    }
}
