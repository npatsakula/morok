//! Operation enum and implementation.
//!
//! The [`Op`] enum defines all possible operations in the IR, from basic arithmetic
//! to complex control flow and memory operations.

use std::sync::Arc;

use smallvec::SmallVec;

use crate::types::*;
use crate::uop::UOp;
use morok_dtype::DType;
use morok_dtype::DeviceSpec;

/// Operation type with typed operands.
///
/// Each operation encodes its operand structure directly in the enum variant.
/// This provides compile-time verification of operand count and types.
///
/// Design choices:
/// - Fixed-arity ops grouped by arity: Unary, Binary, Ternary
/// - Special ops with extra data remain separate: Cast (dtype), MSelect (device_index)
/// - Variable-arity ops use SmallVec: Index { indices: SmallVec<[Arc<UOp>; 4]> }
/// - SmallVec avoids heap allocation for common cases (≤4 children)
/// - Gated operations use separate variants (LoadGated vs Load) for type safety
///
/// Note: PartialEq, Eq, and Hash are NOT derived because Op contains Arc<UOp>.
/// Hash consing uses UOpKey which compares by pointer equality instead.
#[derive(Debug, Clone)]
pub enum Op {
    // Nullary operations (7 variants)
    Const(ConstValueHash),
    Unique(usize),
    Device(DeviceSpec),
    Noop,
    Invalid,
    DefineGlobal(usize),
    DefineLocal(usize),

    // Graph organization operations (2 variants)
    Sink {
        sources: SmallVec<[Arc<UOp>; 4]>,
    },
    Group {
        sources: SmallVec<[Arc<UOp>; 4]>,
    },

    // Grouped operations (3 variants)
    Unary(UnaryOp, Arc<UOp>),
    Binary(BinaryOp, Arc<UOp>, Arc<UOp>),
    Ternary(TernaryOp, Arc<UOp>, Arc<UOp>, Arc<UOp>),

    // Type operations (2 variants)
    Cast {
        src: Arc<UOp>,
        dtype: DType,
    },
    BitCast {
        src: Arc<UOp>,
        dtype: DType,
    },

    // Special operations (2 variants)
    MSelect {
        buffer: Arc<UOp>,
        device_index: usize,
    },
    Special {
        end: Arc<UOp>,
        name: String,
    },

    // Buffer operations (high-level, 7 variants)
    Buffer {
        unique: Arc<UOp>,
        device: Arc<UOp>,
        size: usize,
    },
    BufferView {
        buffer: Arc<UOp>,
        size: usize,
        offset: usize,
    },
    Bufferize {
        compute: Arc<UOp>,
        ranges: SmallVec<[Arc<UOp>; 4]>,
        opts: BufferizeOpts,
    },
    Index {
        buffer: Arc<UOp>,
        indices: SmallVec<[Arc<UOp>; 4]>,
        gate: Option<Arc<UOp>>,
    },
    PointerIndex {
        ptr: Arc<UOp>,
        offset: Arc<UOp>,
    },
    Copy {
        src: Arc<UOp>,
        device: Arc<UOp>,
    },
    MStack {
        buffers: SmallVec<[Arc<UOp>; 4]>,
    },

    // Movement/Reshape operations (7 variants)
    Reshape {
        src: Arc<UOp>,
        new_shape: Arc<UOp>,
    },
    Permute {
        src: Arc<UOp>,
        axes: Vec<usize>,
    },
    Expand {
        src: Arc<UOp>,
        new_shape: Arc<UOp>,
    },
    Pad {
        src: Arc<UOp>,
        begin_pads: Arc<UOp>,
        end_pads: Arc<UOp>,
    },
    Shrink {
        src: Arc<UOp>,
        begins: Arc<UOp>,
        ends: Arc<UOp>,
    },
    Flip {
        src: Arc<UOp>,
        axes: Vec<bool>,
    },
    Multi {
        src: Arc<UOp>,
        axis: usize,
    },

    // Reduction operations (3 variants)
    ReduceAxis {
        src: Arc<UOp>,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
    },
    Reduce {
        src: Arc<UOp>,
        ranges: SmallVec<[Arc<UOp>; 4]>,
        reduce_op: ReduceOp,
    },
    AllReduce {
        src: Arc<UOp>,
        device: Arc<UOp>,
        reduce_op: ReduceOp,
    },

    // Control flow operations (5 variants)
    If {
        condition: Arc<UOp>,
        body: SmallVec<[Arc<UOp>; 4]>,
    },
    EndIf {
        if_op: Arc<UOp>,
    },
    Range {
        end: Arc<UOp>,
        axis_id: AxisId,
        axis_type: AxisType,
    },
    End {
        computation: Arc<UOp>,
        ranges: SmallVec<[Arc<UOp>; 4]>,
    },
    Barrier {
        src: Arc<UOp>,
        deps: SmallVec<[Arc<UOp>; 4]>,
    },

    // Vector operations (5 variants)
    Vectorize {
        elements: SmallVec<[Arc<UOp>; 4]>,
    },
    Gep {
        vector: Arc<UOp>,
        indices: Vec<usize>,
    },
    VConst {
        values: Vec<ConstValue>,
    },
    /// Concatenate vectors into larger vector (expander op).
    /// Like VECTORIZE but sources can be vectors themselves.
    /// Output vcount = sum of all input vcounts.
    Cat {
        sources: SmallVec<[Arc<UOp>; 4]>,
    },
    /// Concatenate pointers into vectorized pointer (expander op).
    /// Used for grouping memory accesses in devectorizer.
    PtrCat {
        sources: SmallVec<[Arc<UOp>; 4]>,
    },

    // Symbolic/Define operations (3 variants)
    DefineVar {
        name: String,
        max_val: i64,
    },
    Bind {
        var: Arc<UOp>,
        value: Arc<UOp>,
    },
    DefineReg {
        size: usize,
    },

    // Advanced operations (12 variants)
    Wmma {
        a: Arc<UOp>,
        b: Arc<UOp>,
        c: Arc<UOp>,
        metadata: WmmaMetadata,
    },
    Contract {
        src: Arc<UOp>,
        upcast_ranges: Vec<(usize, usize)>,
    },
    Unroll {
        src: Arc<UOp>,
        unroll_axes: Vec<(usize, usize)>,
    },
    Kernel {
        sources: SmallVec<[Arc<UOp>; 4]>,
        ast: Arc<UOp>,
    },
    Assign {
        target: Arc<UOp>,
        value: Arc<UOp>,
    },
    Detach {
        src: Arc<UOp>,
    },
    Contiguous {
        src: Arc<UOp>,
    },
    ContiguousBackward {
        src: Arc<UOp>,
    },
    After {
        passthrough: Arc<UOp>,
        deps: SmallVec<[Arc<UOp>; 4]>,
    },
    Precast {
        src: Arc<UOp>,
    },
    Custom {
        deps: SmallVec<[Arc<UOp>; 4]>,
        code: String,
    },
    CustomI {
        deps: SmallVec<[Arc<UOp>; 4]>,
        code: String,
    },

    // Memory operations (low-level, after kernel splitting, 4 variants)
    Load {
        buffer: Arc<UOp>,
        index: Arc<UOp>,
    },
    LoadGated {
        buffer: Arc<UOp>,
        index: Arc<UOp>,
        gate: Arc<UOp>,
    },
    Store {
        buffer: Arc<UOp>,
        index: Arc<UOp>,
        value: Arc<UOp>,
    },
    StoreGated {
        buffer: Arc<UOp>,
        index: Arc<UOp>,
        value: Arc<UOp>,
        gate: Arc<UOp>,
    },
}

impl Op {
    /// Get all child UOps as a Vec of references.
    ///
    /// This is the convenient API for traversing the graph.
    /// Allocates a Vec but is simple to use.
    pub fn children(&self) -> SmallVec<[&Arc<UOp>; 4]> {
        match self {
            // Nullary operations
            Self::Const(_)
            | Self::Unique(_)
            | Self::Device(_)
            | Self::Noop
            | Self::Invalid
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
            Self::End { computation, ranges } => {
                let mut children = SmallVec::from_slice(&[computation]);
                children.extend(ranges.iter());
                children
            }
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
            Self::Kernel { sources, ast } => {
                let mut children: SmallVec<[&Arc<UOp>; 4]> = sources.iter().collect();
                children.push(ast);
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
    pub fn sources(&self) -> SmallVec<[Arc<UOp>; 4]> {
        self.children().iter().map(|rc| (*rc).clone()).collect()
    }

    /// Apply a function to each child UOp.
    pub fn map_child<F>(&self, mut f: F)
    where
        F: FnMut(&Arc<UOp>),
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

    /// Get the source index where ranges start being "ended" by this operation.
    ///
    /// Based on Tinygrad's `range_start` dict (ops.py:28).
    /// Returns `Some(index)` if this operation ends ranges, `None` otherwise.
    ///
    /// Operations that end ranges:
    /// - BUFFERIZE: ranges start at index 1 (compute is 0, ranges are 1+)
    /// - REDUCE: ranges start at index 1 (src is 0, ranges are 1+)
    /// - STORE: ranges start at index 2 (buffer=0, index=1, value=2)
    /// - WMMA: ranges start at index 3 (a=0, b=1, c=2)
    /// - END: ranges start at index 1 (computation=0, ranges=1+)
    ///
    /// These operations mark range boundaries in the computation graph.
    /// Any RANGE operations in sources at or after the returned index
    /// are considered "ended" and removed from scope.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use morok_ir::Op;
    ///
    /// // BUFFERIZE ends ranges starting at source index 1
    /// let bufferize_op = Op::Bufferize { /* ... */ };
    /// assert_eq!(bufferize_op.range_ending_src_index(), Some(1));
    ///
    /// // Regular arithmetic operations don't end ranges
    /// let binary_op = Op::Binary(/* ... */);
    /// assert_eq!(binary_op.range_ending_src_index(), None);
    /// ```
    pub fn range_ending_src_index(&self) -> Option<usize> {
        match self {
            Self::Bufferize { .. } => Some(1),
            Self::Reduce { .. } => Some(1),
            Self::Store { .. } | Self::StoreGated { .. } => Some(2),
            Self::Wmma { .. } => Some(3),
            Self::End { .. } => Some(1),
            _ => None,
        }
    }

    /// Get the "ended ranges" for this operation.
    ///
    /// These are the RANGE operations (and operations containing ranges)
    /// that should be removed from scope after this operation.
    ///
    /// Based on Tinygrad's `ended_ranges` property (ops.py:296-299).
    ///
    /// # Returns
    ///
    /// A SmallVec of references to child UOps that represent ended ranges.
    /// For operations that don't end ranges, returns an empty SmallVec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use morok_ir::{Op, UOp};
    ///
    /// // END operation ends its range arguments
    /// let range = UOp::range(/* ... */);
    /// let computation = UOp::const_(/* ... */);
    /// let end_op = UOp::end(computation, vec![range.clone()]);
    ///
    /// // ended_ranges() returns the ranges that are closed
    /// let ended = end_op.op().ended_ranges();
    /// assert_eq!(ended.len(), 1);
    /// ```
    pub fn ended_ranges(&self) -> SmallVec<[&Arc<UOp>; 4]> {
        if let Some(start_idx) = self.range_ending_src_index() {
            let children = self.children();
            children.into_iter().skip(start_idx).collect()
        } else {
            SmallVec::new()
        }
    }
}
