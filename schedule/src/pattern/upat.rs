//! UPat pattern matching DSL.
//!
//! UPat provides a pattern matching language for UOp graphs, similar to
//! regular expressions but for tree structures. It supports:
//!
//! - Wildcard matching (`UPat::var("x")` matches any UOp)
//! - Operation type matching (`UPat::op(vec![BinaryOp::Add], ...)`)
//! - Constant matching (`UPat::cvar("c")` matches only constants)
//! - Named captures (matched UOps bound to names)
//! - Source structure matching (fixed, repeat, fork patterns)
//! - Argument matching (exact values or predicates)
//!
//! # Example
//!
//! ```ignore
//! // Match: x + 0 (any UOp plus zero constant)
//! let pat = UPat::op(
//!     vec![BinaryOp::Add],
//!     vec![UPat::var("x"), UPat::cvar("zero")],
//! );
//! ```

use std::collections::HashMap;
use std::mem::discriminant;
use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{AxisType, BinaryOp, ConstValue, ConstValueHash, Op, TernaryOp, UOp, UnaryOp};
use smallvec::SmallVec;

/// Filter for matching operation types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpFilter {
    /// Match specific unary operations
    Unary(Vec<UnaryOp>),
    /// Match specific binary operations
    Binary(Vec<BinaryOp>),
    /// Match specific ternary operations
    Ternary(Vec<TernaryOp>),
    /// Match any operation with this discriminant (for ops without sub-types)
    Discriminant(std::mem::Discriminant<Op>),
}

/// Pattern for matching UOp graphs.
///
/// UPat forms a pattern AST that can match against UOp trees.
/// Matching returns all possible bindings of pattern variables to UOps.
#[derive(Debug, Clone)]
pub enum UPat {
    /// Match a UOp with specific constraints.
    ///
    /// All specified constraints must be satisfied. `None` means no constraint.
    Match {
        /// Which operations to match (None = any op)
        op: Option<Vec<OpFilter>>,
        /// Which dtypes to match (None = any dtype)
        dtype: Option<Vec<DType>>,
        /// Source pattern (how to match children)
        src: Option<SrcPattern>,
        /// Argument pattern (how to match operation data)
        arg: Option<ArgPattern>,
        /// Bind matched UOp to this name
        name: Option<String>,
    },
    /// Match any of the provided patterns (OR logic).
    Any(Vec<UPat>),
}

/// Pattern for matching UOp source lists.
#[derive(Debug, Clone)]
pub enum SrcPattern {
    /// Fixed list of patterns - each source must match corresponding pattern.
    /// Length must match exactly.
    Tuple(Vec<UPat>),

    /// All sources must match the same pattern.
    /// Allows variable number of sources (0 or more).
    Repeat(Box<UPat>),

    /// Fork - try each tuple pattern option (OR over source structures).
    /// Useful for matching different arities: `Fork(vec![vec![a], vec![a, b]])`
    /// matches either 1 or 2 sources.
    Fork(Vec<Vec<UPat>>),
}

/// Pattern for matching operation arguments/data.
#[derive(Debug, Clone)]
pub enum ArgPattern {
    /// Match exact constant value
    Const(ConstValue),

    /// Match using custom predicate
    /// Note: Since functions aren't cloneable, we use a named predicate enum instead
    Predicate(ArgPredicate),
}

/// Named predicates for argument matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgPredicate {
    /// Match any constant (for Const ops)
    IsConst,
    /// Match positive integers
    IsPositive,
    /// Match non-zero values
    IsNonZero,
}

impl ArgPredicate {
    /// Check if the UOp matches this predicate.
    pub fn matches(&self, uop: &UOp) -> bool {
        match self {
            ArgPredicate::IsConst => matches!(uop.op(), Op::Const(_)),
            ArgPredicate::IsPositive => {
                if let Op::Const(cv) = uop.op() {
                    match cv.0 {
                        ConstValue::Int(i) => i > 0,
                        ConstValue::Float(f) => f > 0.0,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ArgPredicate::IsNonZero => {
                if let Op::Const(cv) = uop.op() {
                    match cv.0 {
                        ConstValue::Int(i) => i != 0,
                        ConstValue::Float(f) => f != 0.0,
                        _ => true,
                    }
                } else {
                    true
                }
            }
        }
    }
}

impl UPat {
    /// Create a wildcard pattern that matches any UOp.
    ///
    /// The matched UOp is bound to the given name.
    ///
    /// # Example
    /// ```ignore
    /// let pat = UPat::var("x");  // Matches any UOp, binds to "x"
    /// ```
    pub fn var(name: impl Into<String>) -> Self {
        UPat::Match { op: None, dtype: None, src: None, arg: None, name: Some(name.into()) }
    }

    /// Create a constant pattern that matches only Const ops.
    ///
    /// The matched constant is bound to the given name.
    ///
    /// # Example
    /// ```ignore
    /// let pat = UPat::cvar("c");  // Matches any constant, binds to "c"
    /// ```
    pub fn cvar(name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
            dtype: None,
            src: None,
            arg: Some(ArgPattern::Predicate(ArgPredicate::IsConst)),
            name: Some(name.into()),
        }
    }

    /// Create a pattern that matches specific binary operations.
    ///
    /// # Example
    /// ```ignore
    /// use morok_ir::BinaryOp;
    ///
    /// // Match ADD or SUB with two children
    /// let pat = UPat::binary(
    ///     vec![BinaryOp::Add, BinaryOp::Sub],
    ///     vec![UPat::var("a"), UPat::var("b")],
    /// );
    /// ```
    pub fn binary(ops: Vec<BinaryOp>, src: Vec<UPat>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Binary(ops)]),
            dtype: None,
            src: Some(SrcPattern::Tuple(src)),
            arg: None,
            name: None,
        }
    }

    /// Create a pattern that matches specific unary operations.
    pub fn unary(ops: Vec<UnaryOp>, src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Unary(ops)]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Create a pattern that matches specific ternary operations.
    pub fn ternary(ops: Vec<TernaryOp>, src: Vec<UPat>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Ternary(ops)]),
            dtype: None,
            src: Some(SrcPattern::Tuple(src)),
            arg: None,
            name: None,
        }
    }

    /// Create a pattern that matches any of the given patterns (OR logic).
    pub fn any(patterns: Vec<UPat>) -> Self {
        UPat::Any(patterns)
    }

    // ===== Predicate and Constant Helpers =====

    /// Match a zero constant (0 or 0.0) with name binding.
    ///
    /// This is a convenience method that matches constants that are zero,
    /// eliminating the need for manual zero checks in rewrite functions.
    ///
    /// # Example
    /// ```ignore
    /// // Match: x * 0 → 0
    /// pattern!(patterns,
    ///     UPat::var("x") * UPat::zero_const("zero") => |x, zero| {
    ///         let _unused = x;
    ///         Some(Rc::clone(zero))
    ///     }
    /// );
    /// ```
    pub fn zero_const(name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
            dtype: None,
            src: None,
            arg: Some(ArgPattern::Const(ConstValue::Int(0))),
            name: Some(name.into()),
        }
    }

    /// Match a positive constant with name binding.
    ///
    /// # Example
    /// ```ignore
    /// pattern!(patterns,
    ///     UPat::var("x") / UPat::positive_const("n") => |x, n| {
    ///         // n is guaranteed to be positive
    ///         Some(optimize_division(x, n))
    ///     }
    /// );
    /// ```
    pub fn positive_const(name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
            dtype: None,
            src: None,
            arg: Some(ArgPattern::Predicate(ArgPredicate::IsPositive)),
            name: Some(name.into()),
        }
    }

    /// Match a non-zero constant with name binding.
    ///
    /// # Example
    /// ```ignore
    /// pattern!(patterns,
    ///     UPat::var("x") / UPat::nonzero_const("n") => |x, n| {
    ///         // n is guaranteed to be non-zero, safe to divide
    ///         Some(safe_division(x, n))
    ///     }
    /// );
    /// ```
    pub fn nonzero_const(name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
            dtype: None,
            src: None,
            arg: Some(ArgPattern::Predicate(ArgPredicate::IsNonZero)),
            name: Some(name.into()),
        }
    }

    /// Match a specific integer constant value.
    ///
    /// # Example
    /// ```ignore
    /// // Match exactly 42
    /// let pat = UPat::int(42);
    /// ```
    pub fn int(value: i64) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
            dtype: None,
            src: None,
            arg: Some(ArgPattern::Const(ConstValue::Int(value))),
            name: None,
        }
    }

    /// Match a specific float constant value.
    ///
    /// # Example
    /// ```ignore
    /// // Match exactly 1.0
    /// let pat = UPat::float(1.0);
    /// ```
    pub fn float(value: f64) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
            dtype: None,
            src: None,
            arg: Some(ArgPattern::Const(ConstValue::Float(value))),
            name: None,
        }
    }

    /// Match any constant with a specific ConstValue.
    ///
    /// # Example
    /// ```ignore
    /// use morok_ir::ConstValue;
    ///
    /// // Match exactly 0
    /// let pat = UPat::const_val(ConstValue::Int(0));
    /// ```
    pub fn const_val(value: ConstValue) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
            dtype: None,
            src: None,
            arg: Some(ArgPattern::Const(value)),
            name: None,
        }
    }

    // ===== Operation Helpers =====

    /// Match DETACH operation with one source.
    ///
    /// DETACH marks gradient boundaries during autodiff.
    ///
    /// # Example
    /// ```ignore
    /// // Match: DETACH(x) → x
    /// pattern!(patterns,
    ///     UPat::detach(UPat::var("x")) => |x| {
    ///         Some(Rc::clone(x))
    ///     }
    /// );
    /// ```
    pub fn detach(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Detach {
                src: UOp::const_(DType::Void, ConstValue::Int(0)),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match CONTIGUOUS_BACKWARD operation with one source.
    ///
    /// CONTIGUOUS_BACKWARD marks backward pass contiguous requirements.
    ///
    /// # Example
    /// ```ignore
    /// // Match: CONTIGUOUS_BACKWARD(x) → x
    /// pattern!(patterns,
    ///     UPat::contiguous_backward(UPat::var("x")) => |x| {
    ///         Some(Rc::clone(x))
    ///     }
    /// );
    /// ```
    pub fn contiguous_backward(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::ContiguousBackward {
                src: UOp::const_(DType::Void, ConstValue::Int(0)),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match CAST operation with one source.
    ///
    /// Optionally specify a name to bind the matched cast operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: CAST(x)
    /// pattern!(patterns,
    ///     UPat::cast(UPat::var("x")) => |x| {
    ///         // Optimize cast
    ///         Some(optimize_cast(x))
    ///     }
    /// );
    /// ```
    pub fn cast(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Cast {
                src: UOp::const_(DType::Void, ConstValue::Int(0)),
                dtype: DType::Void,
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match CAST operation with one source and bind to a name.
    ///
    /// # Example
    /// ```ignore
    /// // Match: CAST(CAST(x)) - double cast
    /// pattern!(patterns,
    ///     UPat::cast_named(UPat::cast(UPat::var("x")), "outer") => |x, outer| {
    ///         // Access both the inner x and outer cast
    ///         Some(collapse_double_cast(x, outer))
    ///     }
    /// );
    /// ```
    pub fn cast_named(src: UPat, name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Cast {
                src: UOp::const_(DType::Void, ConstValue::Int(0)),
                dtype: DType::Void,
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: Some(name.into()),
        }
    }

    // ===== Kernel Splitting Helpers =====

    /// Match STORE operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: STORE(buffer, index, value)
    /// pattern!(patterns,
    ///     UPat::store(UPat::var("buf"), UPat::var("idx"), UPat::var("val")) => |buf, idx, val| {
    ///         Some(optimize_store(buf, idx, val))
    ///     }
    /// );
    /// ```
    pub fn store(buffer: UPat, index: UPat, value: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Store {
                buffer: UOp::noop(),
                index: UOp::noop(),
                value: UOp::noop(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![buffer, index, value])),
            arg: None,
            name: None,
        }
    }

    /// Match END operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: END(range)
    /// pattern!(patterns,
    ///     UPat::end(UPat::var("range")) => |range| {
    ///         Some(handle_end(range))
    ///     }
    /// );
    /// ```
    pub fn end(computation: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::End {
                computation: UOp::noop(),
                ranges: SmallVec::new(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![computation])),
            arg: None,
            name: None,
        }
    }

    /// Match INDEX operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: INDEX(buffer, indices...)
    /// pattern!(patterns,
    ///     UPat::index(UPat::var("buf"), UPat::var("indices")) => |buf, indices| {
    ///         Some(optimize_index(buf, indices))
    ///     }
    /// );
    /// ```
    pub fn index(buffer: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Index {
                buffer: UOp::noop(),
                indices: SmallVec::new(),
                gate: None,
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![buffer])),
            arg: None,
            name: None,
        }
    }

    /// Match LOAD operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: LOAD(buffer, index)
    /// pattern!(patterns,
    ///     UPat::load(UPat::var("buf"), UPat::var("idx")) => |buf, idx| {
    ///         Some(optimize_load(buf, idx))
    ///     }
    /// );
    /// ```
    pub fn load(buffer: UPat, index: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Load { buffer: UOp::noop(), index: UOp::noop() }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![buffer, index])),
            arg: None,
            name: None,
        }
    }

    /// Match gated LOAD operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: LOAD_GATED(buffer, index, gate)
    /// pattern!(patterns,
    ///     UPat::load_gated(UPat::var("buf"), UPat::var("idx"), UPat::var("gate")) => |buf, idx, gate| {
    ///         Some(optimize_load_gated(buf, idx, gate))
    ///     }
    /// );
    /// ```
    pub fn load_gated(buffer: UPat, index: UPat, gate: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::LoadGated {
                buffer: UOp::noop(),
                index: UOp::noop(),
                gate: UOp::noop(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![buffer, index, gate])),
            arg: None,
            name: None,
        }
    }

    /// Match REDUCE operation.
    ///
    /// Matches any REDUCE operation regardless of reduce_op type.
    /// To match specific reduce operations, check the op in the closure.
    ///
    /// # Example
    /// ```ignore
    /// // Match: REDUCE(src, ranges...)
    /// pattern!(patterns,
    ///     UPat::reduce(UPat::var("src")) => |src| {
    ///         Some(optimize_reduce(src))
    ///     }
    /// );
    /// ```
    pub fn reduce(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Reduce {
                src: UOp::noop(),
                ranges: SmallVec::new(),
                reduce_op: morok_ir::ReduceOp::Add,
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match BUFFERIZE operation.
    ///
    /// Note: This matches any BUFFERIZE regardless of the number of ranges.
    /// Use specific patterns if you need to match exact range counts.
    ///
    /// # Example
    /// ```ignore
    /// // Match: BUFFERIZE(compute, ...)
    /// pattern!(patterns,
    ///     UPat::var("buf") => |buf| {
    ///         if matches!(buf.op(), Op::Bufferize { .. }) {
    ///             Some(convert_bufferize(buf))
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// );
    /// ```
    pub fn bufferize_var(name: impl Into<String>) -> Self {
        UPat::var(name) // Simplified - check op type in pattern closure
    }

    // ===== Movement Operation Helpers =====

    /// Match RESHAPE operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: RESHAPE(src, new_shape)
    /// pattern!(patterns,
    ///     UPat::reshape(UPat::var("src")) => |src| {
    ///         Some(optimize_reshape(src))
    ///     }
    /// );
    /// ```
    pub fn reshape(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Reshape {
                src: UOp::noop(),
                new_shape: UOp::noop(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match PERMUTE operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: PERMUTE(src, axes)
    /// pattern!(patterns,
    ///     UPat::permute(UPat::var("src")) => |src| {
    ///         Some(optimize_permute(src))
    ///     }
    /// );
    /// ```
    pub fn permute(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Permute { src: UOp::noop(), axes: vec![] }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match EXPAND operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: EXPAND(src, new_shape)
    /// pattern!(patterns,
    ///     UPat::expand(UPat::var("src")) => |src| {
    ///         Some(optimize_expand(src))
    ///     }
    /// );
    /// ```
    pub fn expand(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Expand {
                src: UOp::noop(),
                new_shape: UOp::noop(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match PAD operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: PAD(src, begin_pads, end_pads)
    /// pattern!(patterns,
    ///     UPat::pad(UPat::var("src")) => |src| {
    ///         Some(optimize_pad(src))
    ///     }
    /// );
    /// ```
    pub fn pad(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Pad {
                src: UOp::noop(),
                begin_pads: UOp::noop(),
                end_pads: UOp::noop(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match SHRINK operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: SHRINK(src, begins, ends)
    /// pattern!(patterns,
    ///     UPat::shrink(UPat::var("src")) => |src| {
    ///         Some(optimize_shrink(src))
    ///     }
    /// );
    /// ```
    pub fn shrink(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Shrink {
                src: UOp::noop(),
                begins: UOp::noop(),
                ends: UOp::noop(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match FLIP operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: FLIP(src, axes)
    /// pattern!(patterns,
    ///     UPat::flip(UPat::var("src")) => |src| {
    ///         Some(optimize_flip(src))
    ///     }
    /// );
    /// ```
    pub fn flip(src: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Flip { src: UOp::noop(), axes: vec![] }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: None,
        }
    }

    /// Match DEFINE_GLOBAL operation with name binding.
    ///
    /// # Example
    /// ```ignore
    /// // Match: DEFINE_GLOBAL(n)
    /// pattern!(patterns,
    ///     UPat::define_global("global") => |global| {
    ///         Some(use_global(global))
    ///     }
    /// );
    /// ```
    pub fn define_global(name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::DefineGlobal(0)))]),
            dtype: None,
            src: None,
            arg: None,
            name: Some(name.into()),
        }
    }

    /// Match DEFINE_LOCAL operation with name binding.
    ///
    /// # Example
    /// ```ignore
    /// // Match: DEFINE_LOCAL(n)
    /// pattern!(patterns,
    ///     UPat::define_local("local") => |local| {
    ///         Some(use_local(local))
    ///     }
    /// );
    /// ```
    pub fn define_local(name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::DefineLocal(0)))]),
            dtype: None,
            src: None,
            arg: None,
            name: Some(name.into()),
        }
    }

    /// Match RANGE operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: RANGE(end)
    /// pattern!(patterns,
    ///     UPat::range(UPat::var("end")) => |end| {
    ///         Some(optimize_range(end))
    ///     }
    /// );
    /// ```
    pub fn range(end: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Range {
                end: UOp::noop(),
                axis_id: 0,
                axis_type: AxisType::Loop,
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![end])),
            arg: None,
            name: None,
        }
    }

    /// Match BUFFER operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: BUFFER(unique, device)
    /// pattern!(patterns,
    ///     UPat::buffer(UPat::var("unique"), UPat::var("device")) => |unique, device| {
    ///         Some(optimize_buffer(unique, device))
    ///     }
    /// );
    /// ```
    pub fn buffer(unique: UPat, device: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Buffer {
                unique: UOp::noop(),
                device: UOp::noop(),
                size: 0,
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![unique, device])),
            arg: None,
            name: None,
        }
    }

    /// Match AFTER operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: AFTER(passthrough, ...)
    /// pattern!(patterns,
    ///     UPat::after(UPat::var("pass")) => |pass| {
    ///         Some(handle_after(pass))
    ///     }
    /// );
    /// ```
    pub fn after(passthrough: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::After {
                passthrough: UOp::noop(),
                deps: SmallVec::new(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Fork(vec![
                vec![passthrough.clone()],
                // Could also match with deps, but keep it simple for now
            ])),
            arg: None,
            name: None,
        }
    }

    /// Match BIND operation.
    ///
    /// # Example
    /// ```ignore
    /// // Match: BIND(var, value)
    /// pattern!(patterns,
    ///     UPat::bind(UPat::var("var"), UPat::var("value")) => |var, value| {
    ///         Some(remove_bind(value))  // Unbind returns the value
    ///     }
    /// );
    /// ```
    pub fn bind(var: UPat, value: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Bind { var: UOp::noop(), value: UOp::noop() }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![var, value])),
            arg: None,
            name: None,
        }
    }

    // ===== Operator Overloading Helpers =====

    /// Helper to create binary operation patterns (used by operator traits).
    fn binary_op(op: BinaryOp, left: UPat, right: UPat) -> Self {
        UPat::binary(vec![op], vec![left, right])
    }

    /// Helper to create unary operation patterns (used by operator traits).
    fn unary_op(op: UnaryOp, operand: UPat) -> Self {
        UPat::unary(vec![op], operand)
    }

    // ===== Comparison and Math Methods =====

    /// Match less-than operation: `a < b`
    pub fn lt(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Lt, self, rhs)
    }

    /// Match less-than-or-equal operation: `a <= b`
    /// Note: This is typically represented as !(a > b) = !(b < a) in the IR
    /// For now, this is not directly supported - use lt() instead
    pub fn le(self, _rhs: UPat) -> UPat {
        panic!("Le operation not directly supported in IR - express as !(b < a)")
    }

    /// Match greater-than operation: `a > b`
    /// Note: This is typically represented as b < a in the IR
    pub fn gt(self, rhs: UPat) -> UPat {
        // a > b is b < a
        rhs.lt(self)
    }

    /// Match greater-than-or-equal operation: `a >= b`
    /// Note: This is typically represented as !(a < b) in the IR
    /// For now, this is not directly supported - use lt() instead
    pub fn ge(self, _rhs: UPat) -> UPat {
        panic!("Ge operation not directly supported in IR - express as !(a < b)")
    }

    /// Match equality operation: `a == b`
    ///
    /// Note: This is a method, not the `==` operator, because Rust's `==`
    /// is used for structural equality.
    pub fn eq(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Eq, self, rhs)
    }

    /// Match not-equal operation: `a != b`
    pub fn ne(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Ne, self, rhs)
    }

    /// Match maximum operation: `max(a, b)`
    pub fn max(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Max, self, rhs)
    }

    /// Match power operation: `a ** b`
    pub fn pow(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Pow, self, rhs)
    }

    /// Match integer division: `a / b` (integer)
    ///
    /// Use the `/` operator for float division (Fdiv).
    pub fn idiv(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Idiv, self, rhs)
    }

    /// Match this pattern against a UOp.
    ///
    /// Returns all possible variable bindings that satisfy the pattern.
    /// An empty Vec means no match. Multiple bindings can occur with Fork patterns.
    pub fn match_uop(&self, uop: &Rc<UOp>) -> Vec<HashMap<String, Rc<UOp>>> {
        let mut store = HashMap::new();
        self.match_internal(uop, &mut store)
    }

    /// Internal matching implementation.
    ///
    /// Takes a mutable store for accumulating bindings across the match.
    fn match_internal(&self, uop: &Rc<UOp>, store: &mut HashMap<String, Rc<UOp>>) -> Vec<HashMap<String, Rc<UOp>>> {
        match self {
            UPat::Any(patterns) => {
                // Try each pattern, collect all successful matches
                let mut results = Vec::new();
                for pat in patterns {
                    let mut store_copy = store.clone();
                    results.extend(pat.match_internal(uop, &mut store_copy));
                }
                results
            }

            UPat::Match { op: op_filter, dtype: dtype_filter, src: src_pattern, arg: arg_pattern, name } => {
                // 1. Check operation type
                if let Some(filters) = op_filter
                    && !Self::matches_op_filter(uop.op(), filters)
                {
                    return vec![];
                }

                // 2. Check dtype
                if let Some(dtypes) = dtype_filter
                    && !dtypes.contains(&uop.dtype())
                {
                    return vec![];
                }

                // 3. Check argument pattern
                if let Some(arg_pat) = arg_pattern
                    && !Self::matches_arg(uop, arg_pat)
                {
                    return vec![];
                }

                // 4. Check/store named binding
                if let Some(n) = name {
                    if let Some(existing) = store.get(n) {
                        // Name already bound - must match same UOp (by pointer equality)
                        if !Rc::ptr_eq(existing, uop) {
                            return vec![];
                        }
                    } else {
                        // Bind name to this UOp
                        store.insert(n.clone(), uop.clone());
                    }
                }

                // 5. Match sources
                match src_pattern {
                    None => {
                        // No source constraint - match succeeds
                        vec![store.clone()]
                    }

                    Some(SrcPattern::Tuple(patterns)) => {
                        let children = uop.op().children();

                        // Length must match exactly
                        if children.len() != patterns.len() {
                            return vec![];
                        }

                        // Match each child against corresponding pattern
                        Self::match_sources_tuple(&children, patterns, store)
                    }

                    Some(SrcPattern::Repeat(pattern)) => {
                        let children = uop.op().children();

                        // All children must match the same pattern
                        Self::match_sources_repeat(&children, pattern, store)
                    }

                    Some(SrcPattern::Fork(fork_patterns)) => {
                        let children = uop.op().children();

                        // Try each fork option
                        let mut results = Vec::new();
                        for tuple_pattern in fork_patterns {
                            if children.len() == tuple_pattern.len() {
                                let mut store_copy = store.clone();
                                results.extend(Self::match_sources_tuple(&children, tuple_pattern, &mut store_copy));
                            }
                        }
                        results
                    }
                }
            }
        }
    }

    /// Match sources against a tuple pattern (fixed list).
    fn match_sources_tuple(
        children: &[&Rc<UOp>],
        patterns: &[UPat],
        store: &mut HashMap<String, Rc<UOp>>,
    ) -> Vec<HashMap<String, Rc<UOp>>> {
        // Base case: all children matched
        if children.is_empty() {
            return vec![store.clone()];
        }

        // Recursive case: match first child, then rest
        let first_child = children[0];
        let first_pattern = &patterns[0];

        let mut results = Vec::new();
        for mut binding in first_pattern.match_internal(first_child, store) {
            // Match remaining children with updated bindings
            results.extend(Self::match_sources_tuple(&children[1..], &patterns[1..], &mut binding));
        }

        results
    }

    /// Match sources against a repeat pattern (all match same pattern).
    fn match_sources_repeat(
        children: &[&Rc<UOp>],
        pattern: &UPat,
        store: &mut HashMap<String, Rc<UOp>>,
    ) -> Vec<HashMap<String, Rc<UOp>>> {
        // Base case: no children
        if children.is_empty() {
            return vec![store.clone()];
        }

        // Recursive case: match first child, then rest
        let first_child = children[0];

        let mut results = Vec::new();
        for mut binding in pattern.match_internal(first_child, store) {
            // Match remaining children with updated bindings
            results.extend(Self::match_sources_repeat(&children[1..], pattern, &mut binding));
        }

        results
    }

    /// Check if an Op matches any of the given filters.
    fn matches_op_filter(op: &Op, filters: &[OpFilter]) -> bool {
        filters.iter().any(|filter| match filter {
            OpFilter::Unary(unary_ops) => {
                if let Op::Unary(op_type, _) = op {
                    unary_ops.contains(op_type)
                } else {
                    false
                }
            }
            OpFilter::Binary(binary_ops) => {
                if let Op::Binary(op_type, _, _) = op {
                    binary_ops.contains(op_type)
                } else {
                    false
                }
            }
            OpFilter::Ternary(ternary_ops) => {
                if let Op::Ternary(op_type, _, _, _) = op {
                    ternary_ops.contains(op_type)
                } else {
                    false
                }
            }
            OpFilter::Discriminant(disc) => discriminant(op) == *disc,
        })
    }

    /// Check if a UOp matches an argument pattern.
    fn matches_arg(uop: &UOp, arg_pattern: &ArgPattern) -> bool {
        match arg_pattern {
            ArgPattern::Const(expected) => {
                if let Op::Const(actual) = uop.op() {
                    actual.0 == *expected
                } else {
                    false
                }
            }
            ArgPattern::Predicate(pred) => pred.matches(uop),
        }
    }
}

// ===== Operator Trait Implementations =====

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Shl, Shr, Sub};

/// Arithmetic operators
impl Add for UPat {
    type Output = UPat;

    /// Match addition: `a + b`
    fn add(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Add, self, rhs)
    }
}

impl Sub for UPat {
    type Output = UPat;

    /// Match subtraction: `a - b`
    fn sub(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Sub, self, rhs)
    }
}

impl Mul for UPat {
    type Output = UPat;

    /// Match multiplication: `a * b`
    fn mul(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Mul, self, rhs)
    }
}

impl Div for UPat {
    type Output = UPat;

    /// Match division: `a / b` (float division by default)
    ///
    /// For integer division, use `.idiv()` method.
    fn div(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Fdiv, self, rhs)
    }
}

impl Rem for UPat {
    type Output = UPat;

    /// Match modulo: `a % b`
    fn rem(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Mod, self, rhs)
    }
}

impl Neg for UPat {
    type Output = UPat;

    /// Match negation: `-a`
    fn neg(self) -> UPat {
        Self::unary_op(UnaryOp::Neg, self)
    }
}

/// Bitwise operators
impl BitAnd for UPat {
    type Output = UPat;

    /// Match bitwise AND: `a & b`
    fn bitand(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::And, self, rhs)
    }
}

impl BitOr for UPat {
    type Output = UPat;

    /// Match bitwise OR: `a | b`
    fn bitor(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Or, self, rhs)
    }
}

impl BitXor for UPat {
    type Output = UPat;

    /// Match bitwise XOR: `a ^ b`
    fn bitxor(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Xor, self, rhs)
    }
}

impl Shl for UPat {
    type Output = UPat;

    /// Match left shift: `a << b`
    fn shl(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Shl, self, rhs)
    }
}

impl Shr for UPat {
    type Output = UPat;

    /// Match right shift: `a >> b`
    fn shr(self, rhs: UPat) -> UPat {
        Self::binary_op(BinaryOp::Shr, self, rhs)
    }
}
