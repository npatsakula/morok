//! Operation enum and implementation.
//!
//! The [`Op`] enum defines all possible operations in the IR, from basic arithmetic
//! to complex control flow and memory operations.

use std::rc::Rc;

use smallvec::SmallVec;

use crate::types::*;
use crate::uop::UOp;
use morok_device::DeviceSpec;
use morok_dtype::DType;

/// Operation type with typed operands.
///
/// Each operation encodes its operand structure directly in the enum variant.
/// This provides compile-time verification of operand count and types.
///
/// Design choices:
/// - Fixed-arity ops grouped by arity: Unary, Binary, Ternary
/// - Special ops with extra data remain separate: Cast (dtype), MSelect (device_index)
/// - Variable-arity ops use SmallVec: Index { indices: SmallVec<[Rc<UOp>; 4]> }
/// - SmallVec avoids heap allocation for common cases (≤4 children)
/// - Gated operations use separate variants (LoadGated vs Load) for type safety
///
/// Note: PartialEq, Eq, and Hash are NOT derived because Op contains Rc<UOp>.
/// Hash consing uses UOpKey which compares by pointer equality instead.
#[derive(Debug, Clone)]
pub enum Op {
    // Nullary operations (6 variants)
    Const(ConstValueHash),
    Unique(usize),
    Device(DeviceSpec),
    Noop,
    DefineGlobal(usize),
    DefineLocal(usize),

    // Graph organization operations (2 variants)
    Sink {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },
    Group {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },

    // Grouped operations (3 variants)
    Unary(UnaryOp, Rc<UOp>),
    Binary(BinaryOp, Rc<UOp>, Rc<UOp>),
    Ternary(TernaryOp, Rc<UOp>, Rc<UOp>, Rc<UOp>),

    // Type operations (2 variants)
    Cast {
        src: Rc<UOp>,
        dtype: DType,
    },
    BitCast {
        src: Rc<UOp>,
        dtype: DType,
    },

    // Special operations (2 variants)
    MSelect {
        buffer: Rc<UOp>,
        device_index: usize,
    },
    Special {
        end: Rc<UOp>,
        name: String,
    },

    // Buffer operations (high-level, 7 variants)
    Buffer {
        unique: Rc<UOp>,
        device: Rc<UOp>,
        size: usize,
    },
    BufferView {
        buffer: Rc<UOp>,
        size: usize,
        offset: usize,
    },
    Bufferize {
        compute: Rc<UOp>,
        ranges: SmallVec<[Rc<UOp>; 4]>,
        opts: BufferizeOpts,
    },
    Index {
        buffer: Rc<UOp>,
        indices: SmallVec<[Rc<UOp>; 4]>,
        gate: Option<Rc<UOp>>,
    },
    PointerIndex {
        ptr: Rc<UOp>,
        offset: Rc<UOp>,
    },
    Copy {
        src: Rc<UOp>,
        device: Rc<UOp>,
    },
    MStack {
        buffers: SmallVec<[Rc<UOp>; 4]>,
    },

    // Movement/Reshape operations (7 variants)
    Reshape {
        src: Rc<UOp>,
        new_shape: Rc<UOp>,
    },
    Permute {
        src: Rc<UOp>,
        axes: Vec<usize>,
    },
    Expand {
        src: Rc<UOp>,
        new_shape: Rc<UOp>,
    },
    Pad {
        src: Rc<UOp>,
        begin_pads: Rc<UOp>,
        end_pads: Rc<UOp>,
    },
    Shrink {
        src: Rc<UOp>,
        begins: Rc<UOp>,
        ends: Rc<UOp>,
    },
    Flip {
        src: Rc<UOp>,
        axes: Vec<bool>,
    },
    Multi {
        src: Rc<UOp>,
        axis: usize,
    },

    // Reduction operations (3 variants)
    ReduceAxis {
        src: Rc<UOp>,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
    },
    Reduce {
        src: Rc<UOp>,
        ranges: SmallVec<[Rc<UOp>; 4]>,
        reduce_op: ReduceOp,
    },
    AllReduce {
        src: Rc<UOp>,
        device: Rc<UOp>,
        reduce_op: ReduceOp,
    },

    // Control flow operations (5 variants)
    If {
        condition: Rc<UOp>,
        body: SmallVec<[Rc<UOp>; 4]>,
    },
    EndIf {
        if_op: Rc<UOp>,
    },
    Range {
        end: Rc<UOp>,
        axis_id: usize,
        axis_type: AxisType,
    },
    End {
        range_or_reduce: Rc<UOp>,
    },
    Barrier {
        src: Rc<UOp>,
        deps: SmallVec<[Rc<UOp>; 4]>,
    },

    // Vector operations (5 variants)
    Vectorize {
        elements: SmallVec<[Rc<UOp>; 4]>,
    },
    Gep {
        vector: Rc<UOp>,
        indices: Vec<usize>,
    },
    VConst {
        values: Vec<ConstValue>,
    },
    /// Concatenate vectors into larger vector (expander op).
    /// Like VECTORIZE but sources can be vectors themselves.
    /// Output vcount = sum of all input vcounts.
    Cat {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },
    /// Concatenate pointers into vectorized pointer (expander op).
    /// Used for grouping memory accesses in devectorizer.
    PtrCat {
        sources: SmallVec<[Rc<UOp>; 4]>,
    },

    // Symbolic/Define operations (3 variants)
    DefineVar {
        name: String,
        min_val: i64,
        max_val: i64,
    },
    Bind {
        var: Rc<UOp>,
        value: Rc<UOp>,
    },
    DefineReg {
        size: usize,
    },

    // Advanced operations (12 variants)
    Wmma {
        a: Rc<UOp>,
        b: Rc<UOp>,
        c: Rc<UOp>,
        metadata: WmmaMetadata,
    },
    Contract {
        src: Rc<UOp>,
        upcast_ranges: Vec<(usize, usize)>,
    },
    Unroll {
        src: Rc<UOp>,
        unroll_axes: Vec<(usize, usize)>,
    },
    Kernel {
        ast: Option<Rc<UOp>>,
    },
    Assign {
        target: Rc<UOp>,
        value: Rc<UOp>,
    },
    Detach {
        src: Rc<UOp>,
    },
    Contiguous {
        src: Rc<UOp>,
    },
    ContiguousBackward {
        src: Rc<UOp>,
    },
    After {
        passthrough: Rc<UOp>,
        deps: SmallVec<[Rc<UOp>; 4]>,
    },
    Precast {
        src: Rc<UOp>,
    },
    Custom {
        deps: SmallVec<[Rc<UOp>; 4]>,
        code: String,
    },
    CustomI {
        deps: SmallVec<[Rc<UOp>; 4]>,
        code: String,
    },

    // Memory operations (low-level, after kernel splitting, 4 variants)
    Load {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
    },
    LoadGated {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
        gate: Rc<UOp>,
    },
    Store {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
        value: Rc<UOp>,
    },
    StoreGated {
        buffer: Rc<UOp>,
        index: Rc<UOp>,
        value: Rc<UOp>,
        gate: Rc<UOp>,
    },
}

impl Op {
    /// Get all child UOps as a Vec of references.
    ///
    /// This is the convenient API for traversing the graph.
    /// Allocates a Vec but is simple to use.
    pub fn children(&self) -> SmallVec<[&Rc<UOp>; 4]> {
        match self {
            // Nullary operations
            Self::Const(_)
            | Self::Unique(_)
            | Self::Device(_)
            | Self::Noop
            | Self::DefineGlobal(_)
            | Self::DefineLocal(_)
            | Self::VConst { .. }
            | Self::DefineVar { .. }
            | Self::DefineReg { .. } => SmallVec::new(),

            // Graph organization operations
            Self::Sink { sources } | Self::Group { sources } => sources.iter().collect(),

            // Grouped operations
            Self::Unary(_, x) => SmallVec::from_slice(&[x]),
            Self::Binary(_, a, b) => SmallVec::from_slice(&[a, b]),
            Self::Ternary(_, a, b, c) => SmallVec::from_slice(&[a, b, c]),

            // Type operations
            Self::Cast { src, .. } | Self::BitCast { src, .. } => SmallVec::from_slice(&[src]),

            // Special operations
            Self::MSelect { buffer, .. } => SmallVec::from_slice(&[buffer]),
            Self::Special { end, .. } => SmallVec::from_slice(&[end]),

            // Buffer operations
            Self::Buffer { unique, device, .. } => SmallVec::from_slice(&[unique, device]),
            Self::BufferView { buffer, .. } => SmallVec::from_slice(&[buffer]),
            Self::Bufferize { compute, ranges, .. } => {
                let mut children = SmallVec::from_slice(&[compute]);
                children.extend(ranges.iter());
                children
            }
            Self::Index { buffer, indices, gate } => {
                let mut children = SmallVec::from_slice(&[buffer]);
                children.extend(indices.iter());
                children.extend(gate);
                children
            }
            Self::PointerIndex { ptr, offset } => SmallVec::from_slice(&[ptr, offset]),
            Self::Copy { src, device } => SmallVec::from_slice(&[src, device]),
            Self::MStack { buffers } => buffers.iter().collect(),

            // Movement operations
            Self::Reshape { src, new_shape } => SmallVec::from_slice(&[src, new_shape]),
            Self::Permute { src, .. } | Self::Flip { src, .. } | Self::Multi { src, .. } => {
                SmallVec::from_slice(&[src])
            }
            Self::Expand { src, new_shape } => SmallVec::from_slice(&[src, new_shape]),
            Self::Pad { src, begin_pads, end_pads } => SmallVec::from_slice(&[src, begin_pads, end_pads]),
            Self::Shrink { src, begins, ends } => SmallVec::from_slice(&[src, begins, ends]),

            // Reduction operations
            Self::ReduceAxis { src, .. } => SmallVec::from_slice(&[src]),
            Self::Reduce { src, ranges, .. } => {
                let mut children = SmallVec::from_slice(&[src]);
                children.extend(ranges.iter());
                children
            }
            Self::AllReduce { src, device, .. } => SmallVec::from_slice(&[src, device]),

            // Control flow operations
            Self::If { condition, body } => {
                let mut children = SmallVec::from_slice(&[condition]);
                children.extend(body.iter());
                children
            }
            Self::EndIf { if_op } => SmallVec::from_slice(&[if_op]),
            Self::Range { end, .. } => SmallVec::from_slice(&[end]),
            Self::End { range_or_reduce } => SmallVec::from_slice(&[range_or_reduce]),
            Self::Barrier { src, deps } => {
                let mut children = SmallVec::from_slice(&[src]);
                children.extend(deps.iter());
                children
            }

            // Vector operations
            Self::Vectorize { elements } => elements.iter().collect(),
            Self::Gep { vector, .. } => SmallVec::from_slice(&[vector]),
            Self::Cat { sources } | Self::PtrCat { sources } => sources.iter().collect(),

            // Symbolic/Define operations
            Self::Bind { var, value } => SmallVec::from_slice(&[var, value]),

            // Advanced operations
            Self::Wmma { a, b, c, .. } => SmallVec::from_slice(&[a, b, c]),
            Self::Contract { src, .. }
            | Self::Unroll { src, .. }
            | Self::Detach { src }
            | Self::Contiguous { src }
            | Self::ContiguousBackward { src }
            | Self::Precast { src } => SmallVec::from_slice(&[src]),
            Self::Kernel { ast } => {
                let mut children = SmallVec::new();
                children.extend(ast);
                children
            }
            Self::Assign { target, value } => SmallVec::from_slice(&[target, value]),
            Self::After { passthrough, deps } => {
                let mut children = SmallVec::from_slice(&[passthrough]);
                children.extend(deps.iter());
                children
            }
            Self::Custom { deps, .. } | Self::CustomI { deps, .. } => deps.iter().collect(),

            // Memory operations
            Self::Load { buffer, index } => SmallVec::from_slice(&[buffer, index]),
            Self::LoadGated { buffer, index, gate } => SmallVec::from_slice(&[buffer, index, gate]),
            Self::Store { buffer, index, value } => SmallVec::from_slice(&[buffer, index, value]),
            Self::StoreGated { buffer, index, value, gate } => SmallVec::from_slice(&[buffer, index, value, gate]),
        }
    }

    /// Get all child UOps as a Vec of owned Rcs (cloned).
    ///
    /// Similar to `children()` but returns owned Rcs instead of references.
    /// Useful when you need to reconstruct nodes or store sources.
    pub fn sources(&self) -> SmallVec<[Rc<UOp>; 4]> {
        self.children().iter().map(|rc| (*rc).clone()).collect()
    }

    /// Apply a function to each child UOp.
    pub fn map_child<F>(&self, mut f: F)
    where
        F: FnMut(&Rc<UOp>),
    {
        for child in self.children() {
            f(child);
        }
    }

    /// Check if this operation is a movement operation.
    ///
    /// Movement operations transform tensor shapes without changing data values:
    /// - RESHAPE: Change shape with same number of elements
    /// - PERMUTE: Transpose/reorder axes
    /// - EXPAND: Broadcast to larger shape
    /// - PAD: Add padding around tensor
    /// - SHRINK: Extract sub-region
    /// - FLIP: Reverse along axes
    ///
    /// Note: MULTI is not considered a pure movement op as it has different semantics.
    pub fn is_movement(&self) -> bool {
        matches!(
            self,
            Self::Reshape { .. }
                | Self::Permute { .. }
                | Self::Expand { .. }
                | Self::Pad { .. }
                | Self::Shrink { .. }
                | Self::Flip { .. }
        )
    }
}
