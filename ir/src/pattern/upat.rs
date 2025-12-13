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
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Shl, Shr, Sub};
use std::sync::Arc;

use crate::types::BufferizeOpts;
use crate::{AxisId, AxisType, BinaryOp, ConstValue, ConstValueHash, Op, TernaryOp, UOp, UnaryOp};
use morok_dtype::DType;
use smallvec::SmallVec;

// ===== UPat Constructor Macros =====

/// Macro for implementing operator traits on UPat.
/// Generates Add, Sub, Mul, etc. implementations in a concise form.
macro_rules! impl_upat_ops {
    (
        binary: [ $(($Trait:ident, $method:ident, $op:expr)),* $(,)? ],
        unary: [ $(($UTrait:ident, $umethod:ident, $uop:expr)),* $(,)? ] $(,)?
    ) => {
        $(
            impl $Trait for UPat {
                type Output = UPat;
                fn $method(self, rhs: UPat) -> UPat {
                    Self::binary_op($op, self, rhs)
                }
            }
        )*
        $(
            impl $UTrait for UPat {
                type Output = UPat;
                fn $umethod(self) -> UPat {
                    Self::unary_op($uop, self)
                }
            }
        )*
    };
}

/// Macro for single-source operation constructors.
/// Generates methods like `detach(src)`, `cast(src)`, `reshape(src)`, etc.
macro_rules! upat_op_single {
    ($name:ident, $op_expr:expr) => {
        pub fn $name(src: UPat) -> Self {
            UPat::Match {
                op: Some(vec![OpFilter::Discriminant(discriminant(&$op_expr))]),
                dtype: None,
                src: Some(SrcPattern::Tuple(vec![src])),
                arg: None,
                name: None,
            }
        }
    };
}

/// Macro for fluent API methods (f_* pattern).
/// Generates methods like `f_cast(self)`, `f_reshape(self)`, etc.
macro_rules! upat_fluent {
    ($name:ident, $op_expr:expr) => {
        pub fn $name(self) -> Self {
            UPat::Match {
                op: Some(vec![OpFilter::Discriminant(discriminant(&$op_expr))]),
                dtype: None,
                src: Some(SrcPattern::Tuple(vec![self])),
                arg: None,
                name: None,
            }
        }
    };
}

/// Macro for constant matchers with predicates.
/// Generates methods like `zero_const("name")`, `one_const("name")`, etc.
macro_rules! upat_const_pred {
    ($name:ident, $pred:expr) => {
        pub fn $name(name: impl Into<String>) -> Self {
            UPat::Match {
                op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
                dtype: None,
                src: None,
                arg: Some(ArgPattern::Predicate($pred)),
                name: Some(name.into()),
            }
        }
    };
}

/// Macro for OR wrapper methods.
/// Generates methods like `or_casted()`, `or_detach()`, etc.
macro_rules! upat_or {
    ($name:ident, $wrapper:ident) => {
        pub fn $name(self) -> Self {
            UPat::any(vec![self.clone(), UPat::$wrapper(self)])
        }
    };
}

// ===== Optimized Binding Storage =====

/// Single binding entry: (variable_index, UOp reference).
/// Uses u8 for index since patterns rarely exceed 255 bindings.
pub type BindingEntry = (u8, Arc<UOp>);

/// Stack-allocated binding storage for typical patterns (up to 4 bindings).
/// Falls back to heap allocation for complex patterns with more bindings.
pub type BindingStore = SmallVec<[BindingEntry; 4]>;

/// Variable name interning table.
/// Maps string names to compact u8 indices for efficient binding storage.
#[derive(Debug, Clone, Default)]
pub struct VarIntern {
    /// Variable names in index order
    names: Vec<String>,
    /// Reverse lookup: name -> index
    indices: HashMap<String, u8>,
}

impl VarIntern {
    /// Create a new empty interning table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create an index for a variable name.
    pub fn get_or_insert(&mut self, name: &str) -> u8 {
        if let Some(&idx) = self.indices.get(name) {
            return idx;
        }
        let idx = self.names.len() as u8;
        self.names.push(name.to_string());
        self.indices.insert(name.to_string(), idx);
        idx
    }

    /// Get index for a name (returns None if not interned).
    pub fn get_index(&self, name: &str) -> Option<u8> {
        self.indices.get(name).copied()
    }

    /// Get name for an index.
    pub fn get_name(&self, idx: u8) -> Option<&str> {
        self.names.get(idx as usize).map(|s| s.as_str())
    }

    /// Convert BindingStore to HashMap (for backward compatibility).
    pub fn to_hashmap(&self, store: &BindingStore) -> HashMap<String, Arc<UOp>> {
        store
            .iter()
            .filter_map(|(idx, uop)| self.names.get(*idx as usize).map(|name| (name.clone(), uop.clone())))
            .collect()
    }

    /// Number of interned variables.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Extension methods for BindingStore.
pub trait BindingStoreExt {
    /// Get binding by index.
    fn get_by_index(&self, idx: u8) -> Option<&Arc<UOp>>;

    /// Insert or update binding at index.
    /// Named `set_binding` to avoid conflict with `SmallVec::insert`.
    fn set_binding(&mut self, idx: u8, uop: Arc<UOp>);

    /// Check if binding exists at index.
    fn contains_index(&self, idx: u8) -> bool;
}

impl BindingStoreExt for BindingStore {
    fn get_by_index(&self, idx: u8) -> Option<&Arc<UOp>> {
        self.iter().find(|(i, _)| *i == idx).map(|(_, uop)| uop)
    }

    fn set_binding(&mut self, idx: u8, uop: Arc<UOp>) {
        // Check for existing entry at this index
        for (i, existing_uop) in self.iter_mut() {
            if *i == idx {
                *existing_uop = uop;
                return;
            }
        }
        // New entry
        self.push((idx, uop));
    }

    fn contains_index(&self, idx: u8) -> bool {
        self.iter().any(|(i, _)| *i == idx)
    }
}

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

    /// Permute - match sources against patterns in any order.
    /// Useful for commutative operations where `x + y` should also match `y + x`.
    /// Tries all permutations of patterns against sources.
    Permute(Vec<UPat>),
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
    /// Match zero constant (0 or 0.0)
    IsZero,
    /// Match one constant (1 or 1.0)
    IsOne,
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
            ArgPredicate::IsZero => {
                if let Op::Const(cv) = uop.op() {
                    match cv.0 {
                        ConstValue::Int(i) => i == 0,
                        ConstValue::Float(f) => f == 0.0,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ArgPredicate::IsOne => {
                if let Op::Const(cv) = uop.op() {
                    match cv.0 {
                        ConstValue::Int(i) => i == 1,
                        ConstValue::Float(f) => f == 1.0,
                        _ => false,
                    }
                } else {
                    false
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

    /// Create a pattern that matches commutative binary operations in any order.
    ///
    /// This matches both `op(a, b)` and `op(b, a)`, useful for commutative
    /// operations like Add, Mul, And, Or, Xor, etc.
    ///
    /// # Example
    /// ```ignore
    /// use morok_ir::BinaryOp;
    ///
    /// // Match x + 0 OR 0 + x (both orderings)
    /// let pat = UPat::binary_commutative(
    ///     vec![BinaryOp::Add],
    ///     vec![UPat::var("x"), UPat::zero_const("zero")],
    /// );
    /// ```
    pub fn binary_commutative(ops: Vec<BinaryOp>, src: Vec<UPat>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Binary(ops)]),
            dtype: None,
            src: Some(SrcPattern::Permute(src)),
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

    upat_const_pred!(zero_const, ArgPredicate::IsZero);
    upat_const_pred!(one_const, ArgPredicate::IsOne);
    upat_const_pred!(positive_const, ArgPredicate::IsPositive);
    upat_const_pred!(nonzero_const, ArgPredicate::IsNonZero);

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

    upat_op_single!(detach, Op::Detach { src: UOp::noop() });
    upat_op_single!(contiguous_backward, Op::ContiguousBackward { src: UOp::noop() });
    upat_op_single!(cast, Op::Cast { src: UOp::noop(), dtype: DType::Void });

    /// Match CAST operation and bind to name.
    pub fn cast_named(src: UPat, name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Cast { src: UOp::noop(), dtype: DType::Void }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![src])),
            arg: None,
            name: Some(name.into()),
        }
    }

    // ===== Kernel Splitting Helpers =====

    /// Match STORE operation: `STORE(buffer, index, value)`
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

    upat_op_single!(end, Op::End { computation: UOp::noop(), ranges: SmallVec::new() });

    /// Match any END operation regardless of source count.
    pub fn end_any() -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::End {
                computation: UOp::noop(),
                ranges: SmallVec::new(),
            }))]),
            dtype: None,
            src: None,
            arg: None,
            name: None,
        }
    }

    upat_op_single!(index, Op::Index { buffer: UOp::noop(), indices: SmallVec::new(), gate: None });

    /// Match LOAD operation: `LOAD(buffer, index)`
    pub fn load(buffer: UPat, index: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Load { buffer: UOp::noop(), index: UOp::noop() }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![buffer, index])),
            arg: None,
            name: None,
        }
    }

    /// Match gated LOAD operation: `LOAD_GATED(buffer, index, gate)`
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

    upat_op_single!(reduce, Op::Reduce { src: UOp::noop(), ranges: SmallVec::new(), reduce_op: crate::ReduceOp::Add });

    /// Match any REDUCE operation regardless of source count.
    pub fn reduce_any() -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Reduce {
                src: UOp::noop(),
                ranges: SmallVec::new(),
                reduce_op: crate::ReduceOp::Add,
            }))]),
            dtype: None,
            src: None,
            arg: None,
            name: None,
        }
    }

    /// Match BUFFERIZE operation (simplified - check op type in closure).
    pub fn bufferize_var(name: impl Into<String>) -> Self {
        UPat::var(name)
    }

    // ===== Movement Operation Helpers =====

    upat_op_single!(reshape, Op::Reshape { src: UOp::noop(), new_shape: UOp::noop() });
    upat_op_single!(permute, Op::Permute { src: UOp::noop(), axes: vec![] });
    upat_op_single!(expand, Op::Expand { src: UOp::noop(), new_shape: UOp::noop() });
    upat_op_single!(pad, Op::Pad { src: UOp::noop(), begin_pads: UOp::noop(), end_pads: UOp::noop() });
    upat_op_single!(shrink, Op::Shrink { src: UOp::noop(), begins: UOp::noop(), ends: UOp::noop() });
    upat_op_single!(flip, Op::Flip { src: UOp::noop(), axes: vec![] });

    /// Match DEFINE_GLOBAL operation with name binding.
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
    pub fn define_local(name: impl Into<String>) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::DefineLocal(0)))]),
            dtype: None,
            src: None,
            arg: None,
            name: Some(name.into()),
        }
    }

    upat_op_single!(range, Op::Range { end: UOp::noop(), axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop });

    /// Match BUFFER operation: `BUFFER(unique, device)`
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

    /// Match AFTER operation with Fork pattern.
    pub fn after(passthrough: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::After {
                passthrough: UOp::noop(),
                deps: SmallVec::new(),
            }))]),
            dtype: None,
            src: Some(SrcPattern::Fork(vec![vec![passthrough.clone()]])),
            arg: None,
            name: None,
        }
    }

    /// Match BIND operation: `BIND(var, value)`
    pub fn bind(var: UPat, value: UPat) -> Self {
        UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Bind { var: UOp::noop(), value: UOp::noop() }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![var, value])),
            arg: None,
            name: None,
        }
    }

    // ===== Fluent API Methods =====

    upat_fluent!(
        f_bufferize,
        Op::Bufferize { compute: UOp::noop(), ranges: SmallVec::new(), opts: BufferizeOpts::local() }
    );
    upat_fluent!(f_index, Op::Index { buffer: UOp::noop(), indices: SmallVec::new(), gate: None });
    upat_fluent!(f_cast, Op::Cast { src: UOp::noop(), dtype: DType::Void });
    upat_fluent!(f_reshape, Op::Reshape { src: UOp::noop(), new_shape: UOp::noop() });
    upat_fluent!(f_permute, Op::Permute { src: UOp::noop(), axes: vec![] });
    upat_fluent!(f_expand, Op::Expand { src: UOp::noop(), new_shape: UOp::noop() });
    upat_fluent!(f_reduce, Op::Reduce { src: UOp::noop(), ranges: SmallVec::new(), reduce_op: crate::ReduceOp::Add });
    upat_fluent!(f_copy, Op::Copy { src: UOp::noop(), device: UOp::noop() });

    /// Bind this pattern to a name.
    ///
    /// This is the Tinygrad `name=` equivalent. The matched UOp will be
    /// available in the rewrite closure via this name.
    ///
    /// # Example
    /// ```ignore
    /// // Match BUFFERIZE and bind it to "buf"
    /// UPat::cvar("c").f_bufferize().named("buf")
    /// ```
    pub fn named(self, name: impl Into<String>) -> Self {
        match self {
            UPat::Match { op, dtype, src, arg, name: _ } => {
                UPat::Match { op, dtype, src, arg, name: Some(name.into()) }
            }
            UPat::Any(patterns) => {
                // For Any, we wrap it in a Match with just a name
                UPat::Match {
                    op: None,
                    dtype: None,
                    src: Some(SrcPattern::Tuple(vec![UPat::Any(patterns)])),
                    arg: None,
                    name: Some(name.into()),
                }
            }
        }
    }

    /// Add a dtype constraint to this pattern.
    ///
    /// The pattern will only match UOps with the specified dtype.
    ///
    /// # Example
    /// ```ignore
    /// // Match float32 constants only
    /// UPat::cvar("c").with_dtype(DType::Float32)
    /// ```
    pub fn with_dtype(self, dtype: DType) -> Self {
        self.with_dtypes(vec![dtype])
    }

    /// Add multiple dtype constraints (matches any of them).
    pub fn with_dtypes(self, dtypes: Vec<DType>) -> Self {
        match self {
            UPat::Match { op, dtype: _, src, arg, name } => UPat::Match { op, dtype: Some(dtypes), src, arg, name },
            UPat::Any(patterns) => {
                // Apply dtype constraint to all sub-patterns
                UPat::Any(patterns.into_iter().map(|p| p.with_dtypes(dtypes.clone())).collect())
            }
        }
    }

    // ===== OR Convenience Methods =====

    upat_or!(or_casted, cast);
    upat_or!(or_after, after);
    upat_or!(or_detach, detach);
    upat_or!(or_contiguous_backward, contiguous_backward);

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

    // ===== Variable Name Collection =====

    /// Collect all variable names from this pattern and build a VarIntern.
    pub fn collect_var_names(&self) -> VarIntern {
        let mut intern = VarIntern::new();
        self.collect_var_names_internal(&mut intern);
        intern
    }

    fn collect_var_names_internal(&self, intern: &mut VarIntern) {
        match self {
            UPat::Any(patterns) => {
                for pat in patterns {
                    pat.collect_var_names_internal(intern);
                }
            }
            UPat::Match { src, name, .. } => {
                if let Some(n) = name {
                    intern.get_or_insert(n);
                }
                if let Some(src_pat) = src {
                    match src_pat {
                        SrcPattern::Tuple(pats) => {
                            for pat in pats {
                                pat.collect_var_names_internal(intern);
                            }
                        }
                        SrcPattern::Repeat(pat) => {
                            pat.collect_var_names_internal(intern);
                        }
                        SrcPattern::Fork(fork_pats) => {
                            for pats in fork_pats {
                                for pat in pats {
                                    pat.collect_var_names_internal(intern);
                                }
                            }
                        }
                        SrcPattern::Permute(pats) => {
                            for pat in pats {
                                pat.collect_var_names_internal(intern);
                            }
                        }
                    }
                }
            }
        }
    }

    // ===== Single-Solution Detection =====

    /// Returns true if this pattern can only produce at most one match.
    ///
    /// A pattern is single-solution if it never branches via:
    /// - `UPat::Any` (OR patterns)
    /// - `SrcPattern::Fork` (multiple source structures)
    /// - `SrcPattern::Permute` (multiple orderings)
    ///
    /// Single-solution patterns can use `match_first()` which avoids
    /// Vec allocation entirely.
    pub fn is_single_solution(&self) -> bool {
        match self {
            UPat::Any(_) => false, // Any always branches
            UPat::Match { src, .. } => {
                match src {
                    None => true, // No source pattern = single solution
                    Some(SrcPattern::Tuple(pats)) => pats.iter().all(|p| p.is_single_solution()),
                    Some(SrcPattern::Repeat(pat)) => pat.is_single_solution(),
                    Some(SrcPattern::Fork(_)) => false,    // Fork branches
                    Some(SrcPattern::Permute(_)) => false, // Permute branches
                }
            }
        }
    }

    // ===== Optimized Matching (BindingStore) =====

    /// Match this pattern against a UOp, returning only the first match.
    ///
    /// This is an optimized path for single-solution patterns that avoids
    /// Vec allocation entirely. For patterns that may produce multiple
    /// solutions, use `match_uop_fast()` instead.
    ///
    /// Returns `Some(bindings)` if matched, `None` otherwise.
    pub fn match_first(&self, uop: &Arc<UOp>, intern: &VarIntern) -> Option<BindingStore> {
        let mut store = BindingStore::new();
        if self.match_first_internal(uop, &mut store, intern) { Some(store) } else { None }
    }

    /// Internal single-solution matching. Returns true if matched.
    fn match_first_internal(&self, uop: &Arc<UOp>, store: &mut BindingStore, intern: &VarIntern) -> bool {
        match self {
            UPat::Any(patterns) => {
                // Try each pattern, return first match
                for pat in patterns {
                    let mut store_copy = store.clone();
                    if pat.match_first_internal(uop, &mut store_copy, intern) {
                        *store = store_copy;
                        return true;
                    }
                }
                false
            }

            UPat::Match { op: op_filter, dtype: dtype_filter, src: src_pattern, arg: arg_pattern, name } => {
                // 1. Check operation type
                if let Some(filters) = op_filter
                    && !Self::matches_op_filter(uop.op(), filters)
                {
                    return false;
                }

                // 2. Check dtype
                if let Some(dtypes) = dtype_filter
                    && !dtypes.contains(&uop.dtype())
                {
                    return false;
                }

                // 3. Check argument pattern
                if let Some(arg_pat) = arg_pattern
                    && !Self::matches_arg(uop, arg_pat)
                {
                    return false;
                }

                // 4. Check/store named binding
                if let Some(n) = name
                    && let Some(idx) = intern.get_index(n)
                {
                    if let Some(existing) = store.get_by_index(idx) {
                        if !Arc::ptr_eq(existing, uop) {
                            return false;
                        }
                    } else {
                        store.set_binding(idx, uop.clone());
                    }
                }

                // 5. Match sources
                match src_pattern {
                    None => true,

                    Some(SrcPattern::Tuple(patterns)) => {
                        let children = uop.op().children();
                        // Prefix matching: require at least N children, match first N
                        if children.len() < patterns.len() {
                            return false;
                        }
                        for (child, pat) in children.iter().zip(patterns.iter()) {
                            if !pat.match_first_internal(child, store, intern) {
                                return false;
                            }
                        }
                        true
                    }

                    Some(SrcPattern::Repeat(pattern)) => {
                        let children = uop.op().children();
                        for child in children {
                            if !pattern.match_first_internal(child, store, intern) {
                                return false;
                            }
                        }
                        true
                    }

                    Some(SrcPattern::Fork(fork_patterns)) => {
                        // Try each fork option, return first match
                        let children = uop.op().children();
                        for tuple_pattern in fork_patterns {
                            if children.len() == tuple_pattern.len() {
                                let mut store_copy = store.clone();
                                let mut all_match = true;
                                for (child, pat) in children.iter().zip(tuple_pattern.iter()) {
                                    if !pat.match_first_internal(child, &mut store_copy, intern) {
                                        all_match = false;
                                        break;
                                    }
                                }
                                if all_match {
                                    *store = store_copy;
                                    return true;
                                }
                            }
                        }
                        false
                    }

                    Some(SrcPattern::Permute(patterns)) => {
                        // Try permutations, return first match
                        let children = uop.op().children();
                        if children.len() != patterns.len() {
                            return false;
                        }
                        Self::match_sources_permute_first(&children, patterns, store, intern)
                    }
                }
            }
        }
    }

    /// Try all permutations for single-solution matching.
    fn match_sources_permute_first(
        children: &SmallVec<[&Arc<UOp>; 4]>,
        patterns: &[UPat],
        store: &mut BindingStore,
        intern: &VarIntern,
    ) -> bool {
        let n = patterns.len();
        if n == 0 {
            return true;
        }

        // Fast path for n=2 (most common: binary commutative)
        if n == 2 {
            // Try [0, 1] order
            let mut store1 = store.clone();
            if patterns[0].match_first_internal(children[0], &mut store1, intern)
                && patterns[1].match_first_internal(children[1], &mut store1, intern)
            {
                *store = store1;
                return true;
            }
            // Try [1, 0] order
            let mut store2 = store.clone();
            if patterns[0].match_first_internal(children[1], &mut store2, intern)
                && patterns[1].match_first_internal(children[0], &mut store2, intern)
            {
                *store = store2;
                return true;
            }
            return false;
        }

        // General case: use Heap's algorithm for permutations
        let mut indices: Vec<usize> = (0..n).collect();
        let mut c = vec![0usize; n];
        let mut i = 0;

        // Try initial permutation
        let mut store_copy = store.clone();
        if Self::try_permutation_first(&indices, children, patterns, &mut store_copy, intern) {
            *store = store_copy;
            return true;
        }

        // Generate remaining permutations
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    indices.swap(0, i);
                } else {
                    indices.swap(c[i], i);
                }

                let mut store_copy = store.clone();
                if Self::try_permutation_first(&indices, children, patterns, &mut store_copy, intern) {
                    *store = store_copy;
                    return true;
                }

                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }

        false
    }

    /// Try a single permutation for first-match.
    fn try_permutation_first(
        indices: &[usize],
        children: &SmallVec<[&Arc<UOp>; 4]>,
        patterns: &[UPat],
        store: &mut BindingStore,
        intern: &VarIntern,
    ) -> bool {
        for (child_idx, &pattern_idx) in indices.iter().enumerate() {
            if !patterns[pattern_idx].match_first_internal(children[child_idx], store, intern) {
                return false;
            }
        }
        true
    }

    /// Match this pattern against a UOp using optimized BindingStore.
    ///
    /// Returns all possible variable bindings that satisfy the pattern.
    /// Use with `collect_var_names()` to build the VarIntern first.
    pub fn match_uop_fast(&self, uop: &Arc<UOp>, intern: &VarIntern) -> Vec<BindingStore> {
        let mut store = BindingStore::new();
        self.match_internal_fast(uop, &mut store, intern)
    }

    fn match_internal_fast(&self, uop: &Arc<UOp>, store: &mut BindingStore, intern: &VarIntern) -> Vec<BindingStore> {
        match self {
            UPat::Any(patterns) => {
                let mut results = Vec::new();
                for pat in patterns {
                    let mut store_copy = store.clone();
                    results.extend(pat.match_internal_fast(uop, &mut store_copy, intern));
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

                // 4. Check/store named binding (using index)
                if let Some(n) = name
                    && let Some(idx) = intern.get_index(n)
                {
                    if let Some(existing) = store.get_by_index(idx) {
                        // Name already bound - must match same UOp (by pointer equality)
                        if !Arc::ptr_eq(existing, uop) {
                            return vec![];
                        }
                    } else {
                        // Bind index to this UOp
                        store.set_binding(idx, uop.clone());
                    }
                }

                // 5. Match sources
                match src_pattern {
                    None => vec![store.clone()],

                    Some(SrcPattern::Tuple(patterns)) => {
                        let children = uop.op().children();
                        // Prefix matching: require at least N children, match first N
                        if children.len() < patterns.len() {
                            return vec![];
                        }
                        Self::match_sources_tuple_fast(&children[..patterns.len()], patterns, store, intern)
                    }

                    Some(SrcPattern::Repeat(pattern)) => {
                        let children = uop.op().children();
                        Self::match_sources_repeat_fast(&children, pattern, store, intern)
                    }

                    Some(SrcPattern::Fork(fork_patterns)) => {
                        let children = uop.op().children();
                        let mut results = Vec::new();
                        for tuple_pattern in fork_patterns {
                            if children.len() == tuple_pattern.len() {
                                let mut store_copy = store.clone();
                                results.extend(Self::match_sources_tuple_fast(
                                    &children,
                                    tuple_pattern,
                                    &mut store_copy,
                                    intern,
                                ));
                            }
                        }
                        results
                    }

                    Some(SrcPattern::Permute(patterns)) => {
                        let children = uop.op().children();
                        if children.len() != patterns.len() {
                            return vec![];
                        }
                        Self::match_sources_permute_fast(&children, patterns, store, intern)
                    }
                }
            }
        }
    }

    fn match_sources_tuple_fast(
        children: &[&Arc<UOp>],
        patterns: &[UPat],
        store: &mut BindingStore,
        intern: &VarIntern,
    ) -> Vec<BindingStore> {
        if children.is_empty() {
            return vec![store.clone()];
        }

        let first_child = children[0];
        let first_pattern = &patterns[0];

        let mut results = Vec::new();
        for mut binding in first_pattern.match_internal_fast(first_child, store, intern) {
            results.extend(Self::match_sources_tuple_fast(&children[1..], &patterns[1..], &mut binding, intern));
        }
        results
    }

    fn match_sources_repeat_fast(
        children: &[&Arc<UOp>],
        pattern: &UPat,
        store: &mut BindingStore,
        intern: &VarIntern,
    ) -> Vec<BindingStore> {
        if children.is_empty() {
            return vec![store.clone()];
        }

        let first_child = children[0];
        let mut results = Vec::new();
        for mut binding in pattern.match_internal_fast(first_child, store, intern) {
            results.extend(Self::match_sources_repeat_fast(&children[1..], pattern, &mut binding, intern));
        }
        results
    }

    fn match_sources_permute_fast(
        children: &[&Arc<UOp>],
        patterns: &[UPat],
        store: &mut BindingStore,
        intern: &VarIntern,
    ) -> Vec<BindingStore> {
        let n = patterns.len();
        if n == 0 {
            return vec![store.clone()];
        }

        // FAST PATH: n=2 is most common (binary commutative: Add, Mul, And, Or, Xor)
        // Inline both orderings without Heap's algorithm overhead
        if n == 2 {
            let mut results = Vec::new();

            // Try [0, 1] order: pattern[0] -> children[0], pattern[1] -> children[1]
            let mut store1 = store.clone();
            let matches1 = Self::match_sources_tuple_fast(children, patterns, &mut store1, intern);
            results.extend(matches1);

            // Try [1, 0] order: pattern[0] -> children[1], pattern[1] -> children[0]
            let swapped_children: [&Arc<UOp>; 2] = [children[1], children[0]];
            let mut store2 = store.clone();
            let matches2 = Self::match_sources_tuple_fast(&swapped_children, patterns, &mut store2, intern);
            results.extend(matches2);

            return results;
        }

        // General case: use Heap's algorithm for generating permutations (n > 2)
        let mut results = Vec::new();
        let mut indices: Vec<usize> = (0..n).collect();
        Self::permute_and_match_fast(children, patterns, &indices, store, intern, &mut results);

        let mut c: Vec<usize> = vec![0; n];
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    indices.swap(0, i);
                } else {
                    indices.swap(c[i], i);
                }
                Self::permute_and_match_fast(children, patterns, &indices, store, intern, &mut results);
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }

        results
    }

    fn permute_and_match_fast(
        children: &[&Arc<UOp>],
        patterns: &[UPat],
        indices: &[usize],
        store: &BindingStore,
        intern: &VarIntern,
        results: &mut Vec<BindingStore>,
    ) {
        let permuted_patterns: Vec<&UPat> = indices.iter().map(|&i| &patterns[i]).collect();
        let mut store_copy = store.clone();
        let matches = Self::match_sources_tuple_ref_fast(children, &permuted_patterns, &mut store_copy, intern);
        results.extend(matches);
    }

    fn match_sources_tuple_ref_fast(
        children: &[&Arc<UOp>],
        patterns: &[&UPat],
        store: &mut BindingStore,
        intern: &VarIntern,
    ) -> Vec<BindingStore> {
        if children.is_empty() {
            return vec![store.clone()];
        }

        let first_child = children[0];
        let first_pattern = patterns[0];

        let mut results = Vec::new();
        for mut binding in first_pattern.match_internal_fast(first_child, store, intern) {
            results.extend(Self::match_sources_tuple_ref_fast(&children[1..], &patterns[1..], &mut binding, intern));
        }
        results
    }

    // ===== Legacy Matching (HashMap) =====

    /// Match this pattern against a UOp.
    ///
    /// Returns all possible variable bindings that satisfy the pattern.
    /// An empty Vec means no match. Multiple bindings can occur with Fork patterns.
    pub fn match_uop(&self, uop: &Arc<UOp>) -> Vec<HashMap<String, Arc<UOp>>> {
        let mut store = HashMap::new();
        self.match_internal(uop, &mut store)
    }

    /// Internal matching implementation.
    ///
    /// Takes a mutable store for accumulating bindings across the match.
    fn match_internal(&self, uop: &Arc<UOp>, store: &mut HashMap<String, Arc<UOp>>) -> Vec<HashMap<String, Arc<UOp>>> {
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
                        if !Arc::ptr_eq(existing, uop) {
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

                        // Prefix matching: require at least N children, match first N
                        if children.len() < patterns.len() {
                            return vec![];
                        }

                        // Match first N children against corresponding patterns
                        Self::match_sources_tuple(&children[..patterns.len()], patterns, store)
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

                    Some(SrcPattern::Permute(patterns)) => {
                        let children = uop.op().children();

                        // Length must match exactly
                        if children.len() != patterns.len() {
                            return vec![];
                        }

                        // Try all permutations of patterns against children
                        Self::match_sources_permute(&children, patterns, store)
                    }
                }
            }
        }
    }

    /// Match sources against a tuple pattern (fixed list).
    fn match_sources_tuple(
        children: &[&Arc<UOp>],
        patterns: &[UPat],
        store: &mut HashMap<String, Arc<UOp>>,
    ) -> Vec<HashMap<String, Arc<UOp>>> {
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
        children: &[&Arc<UOp>],
        pattern: &UPat,
        store: &mut HashMap<String, Arc<UOp>>,
    ) -> Vec<HashMap<String, Arc<UOp>>> {
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

    /// Match sources against patterns in any order (permutation matching).
    ///
    /// Tries all permutations of patterns against children and collects
    /// all successful bindings. This is useful for matching commutative
    /// operations where `x + y` should also match pattern `y + x`.
    fn match_sources_permute(
        children: &[&Arc<UOp>],
        patterns: &[UPat],
        store: &mut HashMap<String, Arc<UOp>>,
    ) -> Vec<HashMap<String, Arc<UOp>>> {
        // Generate all permutations of pattern indices and try each
        let n = patterns.len();
        let mut results = Vec::new();

        // Use Heap's algorithm for generating permutations
        let mut indices: Vec<usize> = (0..n).collect();
        Self::permute_and_match(children, patterns, &indices, store, &mut results);

        // Generate remaining permutations
        let mut c: Vec<usize> = vec![0; n];
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    indices.swap(0, i);
                } else {
                    indices.swap(c[i], i);
                }
                Self::permute_and_match(children, patterns, &indices, store, &mut results);
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }

        results
    }

    /// Try matching children against patterns using given index permutation.
    fn permute_and_match(
        children: &[&Arc<UOp>],
        patterns: &[UPat],
        indices: &[usize],
        store: &HashMap<String, Arc<UOp>>,
        results: &mut Vec<HashMap<String, Arc<UOp>>>,
    ) {
        // Create permuted pattern list
        let permuted_patterns: Vec<&UPat> = indices.iter().map(|&i| &patterns[i]).collect();

        // Try matching with this permutation
        let mut store_copy = store.clone();
        let matches = Self::match_sources_tuple_ref(children, &permuted_patterns, &mut store_copy);
        results.extend(matches);
    }

    /// Match sources against a tuple pattern (using references to patterns).
    fn match_sources_tuple_ref(
        children: &[&Arc<UOp>],
        patterns: &[&UPat],
        store: &mut HashMap<String, Arc<UOp>>,
    ) -> Vec<HashMap<String, Arc<UOp>>> {
        // Base case: all children matched
        if children.is_empty() {
            return vec![store.clone()];
        }

        // Recursive case: match first child, then rest
        let first_child = children[0];
        let first_pattern = patterns[0];

        let mut results = Vec::new();
        for mut binding in first_pattern.match_internal(first_child, store) {
            // Match remaining children with updated bindings
            results.extend(Self::match_sources_tuple_ref(&children[1..], &patterns[1..], &mut binding));
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

impl_upat_ops! {
    binary: [
        (Add, add, BinaryOp::Add),
        (Sub, sub, BinaryOp::Sub),
        (Mul, mul, BinaryOp::Mul),
        (Div, div, BinaryOp::Fdiv),
        (Rem, rem, BinaryOp::Mod),
        (BitAnd, bitand, BinaryOp::And),
        (BitOr, bitor, BinaryOp::Or),
        (BitXor, bitxor, BinaryOp::Xor),
        (Shl, shl, BinaryOp::Shl),
        (Shr, shr, BinaryOp::Shr),
    ],
    unary: [
        (Neg, neg, UnaryOp::Neg),
    ],
}
