//! PatternMatcher for applying rewrite rules.
//!
//! PatternMatcher holds a collection of patterns and their rewrite functions,
//! and efficiently matches them against UOps using operation-based indexing.
//!
//! This implementation follows Tinygrad's pdict approach: patterns are indexed
//! by the specific operations they match, with wildcard patterns stored separately.

use std::collections::HashMap;
use std::mem::{Discriminant, discriminant};
use std::rc::Rc;
use std::sync::OnceLock;

use morok_dtype::DType;
use morok_ir::types::{AxisType, ConstValue, ConstValueHash, ReduceOp};
use morok_ir::{BinaryOp, Op, TernaryOp, UOp, UnaryOp};
use smallvec::SmallVec;

use super::upat::{BindingStore, OpFilter, UPat, VarIntern};

/// Result of attempting to rewrite a UOp.
///
/// This enum distinguishes between three cases:
/// - NoMatch: Pattern didn't match or rewrite function declined
/// - Rewritten: Pattern matched and returned a replacement UOp
/// - Gate: Pattern matched and indicates bottom-up gate (child processing required)
#[derive(Debug, Clone)]
pub enum RewriteResult {
    /// Pattern didn't match or rewrite function declined to rewrite
    NoMatch,
    /// Pattern matched and returned a replacement UOp
    Rewritten(Rc<UOp>),
    /// Pattern matched and indicates bottom-up gate (Tinygrad's BottomUpGate)
    /// This signals that children should be processed before proceeding
    Gate(Rc<UOp>),
}

/// Rewrite function type - generic over context type.
///
/// Takes the variable bindings from a successful pattern match, the
/// variable interning table, and a mutable reference to the context,
/// returning a RewriteResult indicating whether the rewrite should be applied.
///
/// The BindingStore provides O(1) indexed access to bindings, avoiding
/// string hashing overhead during pattern matching hot paths.
///
/// Context is passed from `graph_rewrite()` through the rewrite engine
/// to each pattern handler. Patterns that need state mutation or external
/// information receive it through the context parameter. Patterns that
/// don't need context simply ignore the `_ctx` parameter.
///
/// # Example
///
/// ```ignore
/// // Pattern that uses context
/// fn debuf(b: &BindingStore, i: &VarIntern, ctx: &mut KernelContext) -> RewriteResult {
///     let id = ctx.next_global();  // Direct mutable access
///     // ...
/// }
///
/// // Pattern that ignores context
/// fn add_zero<C>(b: &BindingStore, i: &VarIntern, _ctx: &mut C) -> RewriteResult {
///     // Don't use _ctx
/// }
/// ```
pub type RewriteFn<C> = Box<dyn Fn(&BindingStore, &VarIntern, &mut C) -> RewriteResult>;

/// Fast rewrite for common patterns that avoids closure overhead.
///
/// Many patterns have trivial rewrites like `Add(x, 0) ~> x`. Using an enum
/// instead of a closure avoids the indirection and overhead of boxed closures.
///
/// Note: This enum doesn't derive Clone because `RewriteFn` (boxed closures)
/// can't be cloned. For cloning support, use the specific variants.
pub enum FastRewrite<C> {
    /// Return a bound variable unchanged: `x + 0 ~> x`
    /// The string is the variable name to look up in VarIntern at runtime.
    ReturnBinding(String),

    /// Return binding if two variables are pointer-equal: `x & x ~> x`
    ReturnIfPtrEq {
        /// Variable to return
        var: String,
        /// Variable that must be pointer-equal to `var`
        compare: String,
    },

    /// Fallback to closure for complex rewrites
    Closure(RewriteFn<C>),
}

impl<C> FastRewrite<C> {
    /// Apply this rewrite given bindings, variable interning, and context.
    pub fn apply(&self, bindings: &BindingStore, intern: &VarIntern, ctx: &mut C) -> RewriteResult {
        use super::upat::BindingStoreExt;

        match self {
            FastRewrite::ReturnBinding(var_name) => {
                if let Some(idx) = intern.get_index(var_name)
                    && let Some(uop) = bindings.get_by_index(idx)
                {
                    return RewriteResult::Rewritten(uop.clone());
                }
                RewriteResult::NoMatch
            }

            FastRewrite::ReturnIfPtrEq { var, compare } => {
                let var_idx = intern.get_index(var);
                let compare_idx = intern.get_index(compare);

                if let (Some(vi), Some(ci)) = (var_idx, compare_idx)
                    && let (Some(var_uop), Some(cmp_uop)) = (bindings.get_by_index(vi), bindings.get_by_index(ci))
                    && Rc::ptr_eq(var_uop, cmp_uop)
                {
                    return RewriteResult::Rewritten(var_uop.clone());
                }
                RewriteResult::NoMatch
            }

            FastRewrite::Closure(f) => f(bindings, intern, ctx),
        }
    }
}

/// Operation key for indexing patterns.
///
/// Unlike discriminants, OpKey distinguishes between different operation types
/// (e.g., Add vs Mul) without creating dummy UOps.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum OpKey {
    // Grouped operations
    Unary(UnaryOp),
    Binary(BinaryOp),
    Ternary(TernaryOp),

    // Nullary operations (by discriminant-equivalent enum variant)
    Const,
    Unique,
    Device,
    Noop,
    Invalid,
    DefineGlobal,
    DefineLocal,

    // Graph organization
    Sink,
    Group,

    // Type operations
    Cast,
    BitCast,

    // Special operations
    MSelect,
    Special,

    // Buffer operations
    Buffer,
    BufferView,
    Bufferize,
    Index,
    PointerIndex,
    Copy,
    MStack,

    // Movement operations
    Reshape,
    Permute,
    Expand,
    Pad,
    Shrink,
    Flip,
    Multi,

    // Reduction operations
    ReduceAxis,
    Reduce,
    AllReduce,

    // Control flow
    If,
    EndIf,
    Range,
    End,
    Barrier,

    // Vector operations
    Vectorize,
    Gep,
    VConst,
    Cat,
    PtrCat,

    // Symbolic/Define
    DefineVar,
    Bind,
    DefineReg,

    // Advanced operations
    Wmma,
    Contract,
    Unroll,
    Kernel,
    Assign,
    Detach,
    Contiguous,
    ContiguousBackward,
    After,
    Precast,
    Custom,
    CustomI,

    // Memory operations
    Load,
    LoadGated,
    Store,
    StoreGated,
}

impl OpKey {
    /// Extract the operation key from a UOp.
    fn from_op(op: &Op) -> Self {
        match op {
            // Grouped operations - distinguish by sub-type
            Op::Unary(op_type, _) => OpKey::Unary(*op_type),
            Op::Binary(op_type, _, _) => OpKey::Binary(*op_type),
            Op::Ternary(op_type, _, _, _) => OpKey::Ternary(*op_type),

            // All other operations - one key per variant
            Op::Const(_) => OpKey::Const,
            Op::Unique(_) => OpKey::Unique,
            Op::Device(_) => OpKey::Device,
            Op::Noop => OpKey::Noop,
            Op::Invalid => OpKey::Invalid,
            Op::DefineGlobal(_) => OpKey::DefineGlobal,
            Op::DefineLocal(_) => OpKey::DefineLocal,

            Op::Sink { .. } => OpKey::Sink,
            Op::Group { .. } => OpKey::Group,

            Op::Cast { .. } => OpKey::Cast,
            Op::BitCast { .. } => OpKey::BitCast,

            Op::MSelect { .. } => OpKey::MSelect,
            Op::Special { .. } => OpKey::Special,

            Op::Buffer { .. } => OpKey::Buffer,
            Op::BufferView { .. } => OpKey::BufferView,
            Op::Bufferize { .. } => OpKey::Bufferize,
            Op::Index { .. } => OpKey::Index,
            Op::PointerIndex { .. } => OpKey::PointerIndex,
            Op::Copy { .. } => OpKey::Copy,
            Op::MStack { .. } => OpKey::MStack,

            Op::Reshape { .. } => OpKey::Reshape,
            Op::Permute { .. } => OpKey::Permute,
            Op::Expand { .. } => OpKey::Expand,
            Op::Pad { .. } => OpKey::Pad,
            Op::Shrink { .. } => OpKey::Shrink,
            Op::Flip { .. } => OpKey::Flip,
            Op::Multi { .. } => OpKey::Multi,

            Op::ReduceAxis { .. } => OpKey::ReduceAxis,
            Op::Reduce { .. } => OpKey::Reduce,
            Op::AllReduce { .. } => OpKey::AllReduce,

            Op::If { .. } => OpKey::If,
            Op::EndIf { .. } => OpKey::EndIf,
            Op::Range { .. } => OpKey::Range,
            Op::End { .. } => OpKey::End,
            Op::Barrier { .. } => OpKey::Barrier,

            Op::Vectorize { .. } => OpKey::Vectorize,
            Op::Gep { .. } => OpKey::Gep,
            Op::VConst { .. } => OpKey::VConst,
            Op::Cat { .. } => OpKey::Cat,
            Op::PtrCat { .. } => OpKey::PtrCat,

            Op::DefineVar { .. } => OpKey::DefineVar,
            Op::Bind { .. } => OpKey::Bind,
            Op::DefineReg { .. } => OpKey::DefineReg,

            Op::Wmma { .. } => OpKey::Wmma,
            Op::Contract { .. } => OpKey::Contract,
            Op::Unroll { .. } => OpKey::Unroll,
            Op::Kernel { .. } => OpKey::Kernel,
            Op::Assign { .. } => OpKey::Assign,
            Op::Detach { .. } => OpKey::Detach,
            Op::Contiguous { .. } => OpKey::Contiguous,
            Op::ContiguousBackward { .. } => OpKey::ContiguousBackward,
            Op::After { .. } => OpKey::After,
            Op::Precast { .. } => OpKey::Precast,
            Op::Custom { .. } => OpKey::Custom,
            Op::CustomI { .. } => OpKey::CustomI,

            Op::Load { .. } => OpKey::Load,
            Op::LoadGated { .. } => OpKey::LoadGated,
            Op::Store { .. } => OpKey::Store,
            Op::StoreGated { .. } => OpKey::StoreGated,
        }
    }

    /// Get all OpKeys that match an OpFilter.
    fn from_filter(filter: &OpFilter) -> Vec<OpKey> {
        match filter {
            OpFilter::Unary(ops) => ops.iter().map(|op| OpKey::Unary(*op)).collect(),
            OpFilter::Binary(ops) => ops.iter().map(|op| OpKey::Binary(*op)).collect(),
            OpFilter::Ternary(ops) => ops.iter().map(|op| OpKey::Ternary(*op)).collect(),
            OpFilter::Discriminant(disc) => {
                // Map discriminant to OpKey for proper indexing
                Self::from_discriminant(disc).into_iter().collect()
            }
        }
    }

    /// Map a discriminant to its corresponding OpKey.
    ///
    /// Uses a static HashMap built from dummy Op values to avoid
    /// repeated discriminant computation.
    fn from_discriminant(disc: &Discriminant<Op>) -> Option<OpKey> {
        static DISCRIMINANT_MAP: OnceLock<HashMap<Discriminant<Op>, OpKey>> = OnceLock::new();

        let map = DISCRIMINANT_MAP.get_or_init(|| {
            let mut m = HashMap::new();

            // Helper to create a dummy UOp
            let noop = || UOp::noop();

            // Essential nullary ops used with discriminant filters
            m.insert(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))), OpKey::Const);
            m.insert(discriminant(&Op::Noop), OpKey::Noop);
            m.insert(discriminant(&Op::Invalid), OpKey::Invalid);
            m.insert(discriminant(&Op::Unique(0)), OpKey::Unique);
            m.insert(discriminant(&Op::DefineGlobal(0)), OpKey::DefineGlobal);
            m.insert(discriminant(&Op::DefineLocal(0)), OpKey::DefineLocal);

            // Graph organization
            m.insert(discriminant(&Op::Sink { sources: SmallVec::new() }), OpKey::Sink);
            m.insert(discriminant(&Op::Group { sources: SmallVec::new() }), OpKey::Group);

            // Type operations (commonly used with discriminants)
            m.insert(discriminant(&Op::Cast { src: noop(), dtype: DType::Bool }), OpKey::Cast);
            m.insert(discriminant(&Op::BitCast { src: noop(), dtype: DType::Bool }), OpKey::BitCast);

            // Special operations
            m.insert(discriminant(&Op::MSelect { buffer: noop(), device_index: 0 }), OpKey::MSelect);
            m.insert(discriminant(&Op::Special { end: noop(), name: String::new() }), OpKey::Special);

            // Buffer operations
            m.insert(discriminant(&Op::Index { buffer: noop(), indices: SmallVec::new(), gate: None }), OpKey::Index);
            m.insert(discriminant(&Op::PointerIndex { ptr: noop(), offset: noop() }), OpKey::PointerIndex);
            m.insert(discriminant(&Op::MStack { buffers: SmallVec::new() }), OpKey::MStack);

            // Movement operations
            m.insert(discriminant(&Op::Reshape { src: noop(), new_shape: noop() }), OpKey::Reshape);
            m.insert(discriminant(&Op::Permute { src: noop(), axes: Vec::new() }), OpKey::Permute);
            m.insert(discriminant(&Op::Expand { src: noop(), new_shape: noop() }), OpKey::Expand);
            m.insert(discriminant(&Op::Flip { src: noop(), axes: Vec::new() }), OpKey::Flip);
            m.insert(discriminant(&Op::Multi { src: noop(), axis: 0 }), OpKey::Multi);

            // Reduction operations (key ops for dead_loop_patterns)
            m.insert(
                discriminant(&Op::ReduceAxis { src: noop(), reduce_op: ReduceOp::Add, axes: Vec::new() }),
                OpKey::ReduceAxis,
            );
            m.insert(
                discriminant(&Op::Reduce { src: noop(), ranges: SmallVec::new(), reduce_op: ReduceOp::Add }),
                OpKey::Reduce,
            );

            // Control flow (key ops for dead_loop_patterns)
            m.insert(discriminant(&Op::Range { end: noop(), axis_id: 0, axis_type: AxisType::Loop }), OpKey::Range);
            m.insert(discriminant(&Op::End { computation: noop(), ranges: SmallVec::new() }), OpKey::End);

            // Vector operations
            m.insert(discriminant(&Op::Vectorize { elements: SmallVec::new() }), OpKey::Vectorize);
            m.insert(discriminant(&Op::Gep { vector: noop(), indices: Vec::new() }), OpKey::Gep);
            m.insert(discriminant(&Op::VConst { values: Vec::new() }), OpKey::VConst);
            m.insert(discriminant(&Op::Cat { sources: SmallVec::new() }), OpKey::Cat);
            m.insert(discriminant(&Op::PtrCat { sources: SmallVec::new() }), OpKey::PtrCat);

            // Symbolic/Define
            m.insert(discriminant(&Op::DefineVar { name: String::new(), min_val: 0, max_val: 0 }), OpKey::DefineVar);
            m.insert(discriminant(&Op::Bind { var: noop(), value: noop() }), OpKey::Bind);
            m.insert(discriminant(&Op::DefineReg { size: 0 }), OpKey::DefineReg);

            // Advanced operations
            m.insert(discriminant(&Op::Contract { src: noop(), upcast_ranges: Vec::new() }), OpKey::Contract);
            m.insert(discriminant(&Op::Unroll { src: noop(), unroll_axes: Vec::new() }), OpKey::Unroll);
            m.insert(discriminant(&Op::Assign { target: noop(), value: noop() }), OpKey::Assign);
            m.insert(discriminant(&Op::Detach { src: noop() }), OpKey::Detach);
            m.insert(discriminant(&Op::Contiguous { src: noop() }), OpKey::Contiguous);
            m.insert(discriminant(&Op::ContiguousBackward { src: noop() }), OpKey::ContiguousBackward);
            m.insert(discriminant(&Op::Precast { src: noop() }), OpKey::Precast);
            m.insert(discriminant(&Op::Custom { deps: SmallVec::new(), code: String::new() }), OpKey::Custom);
            m.insert(discriminant(&Op::CustomI { deps: SmallVec::new(), code: String::new() }), OpKey::CustomI);

            // Memory operations
            m.insert(discriminant(&Op::Load { buffer: noop(), index: noop() }), OpKey::Load);
            m.insert(discriminant(&Op::LoadGated { buffer: noop(), index: noop(), gate: noop() }), OpKey::LoadGated);
            m.insert(discriminant(&Op::Store { buffer: noop(), index: noop(), value: noop() }), OpKey::Store);
            m.insert(
                discriminant(&Op::StoreGated { buffer: noop(), index: noop(), value: noop(), gate: noop() }),
                OpKey::StoreGated,
            );

            m
        });

        map.get(disc).cloned()
    }
}

/// Pattern matcher that applies rewrite rules to UOps.
///
/// Generic over context type `C`, enabling compile-time type-safe context
/// passing without `Rc<RefCell<>>` or `dyn Any` downcasting. Default context
/// is `()` for patterns that don't need external state.
///
/// Follows Tinygrad's pdict design:
/// - Patterns are indexed by the specific operations they match
/// - Wildcard patterns (match any op) are stored separately
/// - When rewriting, we check indexed patterns first, then wildcards
///
/// # Context Passing
///
/// Context is passed at rewrite-time through `graph_rewrite()`, not captured
/// in closures. This enables patterns to be simple functions:
///
/// ```ignore
/// // Pattern that uses context
/// fn debuf(b: &BindingStore, i: &VarIntern, ctx: &mut KernelContext) -> RewriteResult {
///     let id = ctx.next_global();  // Direct mutable access
///     // ...
/// }
///
/// // Pattern that ignores context
/// fn add_zero<C>(b: &BindingStore, _: &VarIntern, _ctx: &mut C) -> RewriteResult {
///     // Don't use _ctx
/// }
/// ```
///
/// # Composition
///
/// Matchers with the same context type can be combined:
/// ```ignore
/// let pm1: PatternMatcher<KernelContext> = ...;
/// let pm2: PatternMatcher<KernelContext> = ...;
/// let combined = pm1 + pm2;  // OK: same context type
/// ```
///
/// # Example
///
/// ```ignore
/// use morok_ir::BinaryOp;
/// use schedule::pattern::{UPat, PatternMatcher};
///
/// // Pattern: x + 0 -> x
/// let patterns = vec![(
///     UPat::binary(vec![BinaryOp::Add], vec![
///         UPat::var("x"),
///         UPat::cvar("zero"),
///     ]),
///     Box::new(|bindings, intern, _ctx: &mut ()| {
///         let zero = bindings.get("zero")?;
///         if let Op::Const(cv) = zero.op() {
///             if cv.0 == ConstValue::Int(0) {
///                 return Some(bindings.get("x")?.clone());
///             }
///         }
///         None
///     }),
/// )];
///
/// let matcher: PatternMatcher<()> = PatternMatcher::new(patterns);
/// ```
pub struct PatternMatcher<C = ()> {
    /// All patterns with their rewrite functions and variable interning tables.
    /// Each pattern has its own VarIntern for efficient binding lookup.
    patterns: Vec<(UPat, VarIntern, RewriteFn<C>)>,

    /// Pattern dictionary: maps operation keys to pattern indices.
    /// This is the main optimization - we only try patterns that can match.
    pdict: HashMap<OpKey, Vec<usize>>,

    /// Indices of patterns that match any operation (no op filter).
    /// These are checked for every UOp after the op-specific patterns.
    wildcard_indices: Vec<usize>,

    /// Per-pattern flag indicating if pattern produces at most one solution.
    /// When true, we can use `match_first()` which avoids Vec allocation.
    single_solution: Vec<bool>,
}

impl<C> PatternMatcher<C> {
    /// Create an empty PatternMatcher.
    pub fn empty() -> Self {
        Self { patterns: Vec::new(), pdict: HashMap::new(), wildcard_indices: Vec::new(), single_solution: Vec::new() }
    }

    /// Create a new PatternMatcher from a list of patterns and rewrite functions.
    ///
    /// Patterns are tried in the order given, but indexed patterns are checked
    /// before wildcard patterns for efficiency.
    ///
    /// Each pattern's variable names are interned once during construction for
    /// efficient binding lookup during matching. Single-solution patterns are
    /// detected to enable a faster matching path.
    pub fn new(patterns: Vec<(UPat, RewriteFn<C>)>) -> Self {
        let mut pdict: HashMap<OpKey, Vec<usize>> = HashMap::new();
        let mut wildcard_indices = Vec::new();
        let mut single_solution = Vec::with_capacity(patterns.len());

        // Build pattern dictionary and intern variable names
        let patterns_with_intern: Vec<(UPat, VarIntern, RewriteFn<C>)> = patterns
            .into_iter()
            .enumerate()
            .map(|(idx, (pattern, rewrite_fn))| {
                // Extract op keys before moving pattern
                let op_keys = Self::extract_op_keys(&pattern);

                if op_keys.is_empty() {
                    // Pattern matches any operation - add to wildcard list
                    wildcard_indices.push(idx);
                } else {
                    // Pattern matches specific operations - add to each key's list
                    for key in op_keys {
                        pdict.entry(key).or_default().push(idx);
                    }
                }

                // Detect if pattern is single-solution (no branching)
                single_solution.push(pattern.is_single_solution());

                // Build VarIntern for this pattern (collects all variable names)
                let intern = pattern.collect_var_names();

                (pattern, intern, rewrite_fn)
            })
            .collect();

        Self { patterns: patterns_with_intern, pdict, wildcard_indices, single_solution }
    }

    /// Extract all operation keys that a pattern can match.
    ///
    /// Returns an empty Vec if the pattern is a wildcard (matches any op).
    fn extract_op_keys(pattern: &UPat) -> Vec<OpKey> {
        match pattern {
            UPat::Match { op: Some(filters), .. } => {
                // Extract keys from all filters
                let mut keys = Vec::new();
                for filter in filters {
                    keys.extend(OpKey::from_filter(filter));
                }
                keys
            }
            UPat::Match { op: None, .. } => {
                // Wildcard pattern - matches any op
                vec![]
            }
            UPat::Any(patterns) => {
                // Collect all keys from all sub-patterns
                let mut all_keys = Vec::new();
                for p in patterns {
                    all_keys.extend(Self::extract_op_keys(p));
                }
                all_keys
            }
        }
    }

    /// Try to rewrite a UOp using the patterns in this matcher.
    ///
    /// Takes a mutable reference to the context, which is passed through to
    /// pattern rewrite functions. Patterns that don't need context ignore it.
    ///
    /// Returns the result of the rewrite attempt:
    /// - NoMatch: No pattern matched or all rewrite functions declined
    /// - Rewritten(uop): A pattern matched and returned a replacement
    /// - Gate(uop): A pattern matched and indicates bottom-up gate (process children)
    ///
    /// Patterns are tried in this order:
    /// 1. Patterns indexed under this op's OpKey
    /// 2. Wildcard patterns (match any op)
    ///
    /// For single-solution patterns, uses `match_first()` which avoids Vec allocation.
    pub fn rewrite(&self, uop: &Rc<UOp>, ctx: &mut C) -> RewriteResult {
        let op_key = OpKey::from_op(uop.op());

        // Chain indexed patterns with wildcards - no Vec allocation
        let indexed_iter = self.pdict.get(&op_key).into_iter().flat_map(|v| v.iter());
        let candidates = indexed_iter.chain(self.wildcard_indices.iter());

        // Try each candidate pattern
        for idx in candidates {
            let (pattern, intern, rewrite_fn) = &self.patterns[*idx];

            // Use fast path for single-solution patterns (no Vec allocation)
            if self.single_solution[*idx] {
                if let Some(bindings) = pattern.match_first(uop, intern) {
                    match rewrite_fn(&bindings, intern, ctx) {
                        RewriteResult::NoMatch => continue,
                        result @ (RewriteResult::Rewritten(_) | RewriteResult::Gate(_)) => {
                            return result;
                        }
                    }
                }
            } else {
                // Multi-solution pattern: need to try all matches
                let matches = pattern.match_uop_fast(uop, intern);
                for bindings in matches {
                    match rewrite_fn(&bindings, intern, ctx) {
                        RewriteResult::NoMatch => continue,
                        result @ (RewriteResult::Rewritten(_) | RewriteResult::Gate(_)) => {
                            return result;
                        }
                    }
                }
            }
        }

        RewriteResult::NoMatch
    }

    /// Get access to the patterns (for composition).
    ///
    /// Returns patterns with their VarIntern tables stripped (tuple of pattern and rewrite fn).
    pub fn into_patterns(self) -> Vec<(UPat, RewriteFn<C>)> {
        self.patterns.into_iter().map(|(pat, _intern, f)| (pat, f)).collect()
    }

    /// Get access to the patterns with their interning tables (for composition).
    pub fn into_patterns_with_intern(self) -> Vec<(UPat, VarIntern, RewriteFn<C>)> {
        self.patterns
    }

    /// Create a PatternMatcher from patterns that already have VarIntern tables.
    ///
    /// This is more efficient than `new()` for combining matchers since it
    /// avoids re-computing the variable interning.
    pub fn from_patterns_with_intern(patterns: Vec<(UPat, VarIntern, RewriteFn<C>)>) -> Self {
        let mut pdict: HashMap<OpKey, Vec<usize>> = HashMap::new();
        let mut wildcard_indices = Vec::new();
        let mut single_solution = Vec::with_capacity(patterns.len());

        // Build pattern dictionary (interning already done)
        for (idx, (pattern, _, _)) in patterns.iter().enumerate() {
            let op_keys = Self::extract_op_keys(pattern);

            if op_keys.is_empty() {
                wildcard_indices.push(idx);
            } else {
                for key in op_keys {
                    pdict.entry(key).or_default().push(idx);
                }
            }

            // Compute single-solution flag
            single_solution.push(pattern.is_single_solution());
        }

        Self { patterns, pdict, wildcard_indices, single_solution }
    }
}

// ===== Pattern Matcher Composition =====

impl<C> std::ops::Add for PatternMatcher<C> {
    type Output = PatternMatcher<C>;

    /// Combine two PatternMatchers with the same context type.
    ///
    /// Patterns from `self` are tried first, then patterns from `rhs`.
    /// This allows composing pattern matchers like Tinygrad's `pm1 + pm2`.
    ///
    /// Only matchers with the same context type `C` can be combined - this
    /// is enforced at compile time.
    ///
    /// # Example
    /// ```ignore
    /// let pm1: PatternMatcher<KernelContext> = ...;
    /// let pm2: PatternMatcher<KernelContext> = ...;
    /// let combined = pm1 + pm2;  // OK: same context type
    ///
    /// let pm3: PatternMatcher<()> = ...;
    /// let bad = pm1 + pm3;  // Error: mismatched context types
    /// ```
    fn add(self, rhs: PatternMatcher<C>) -> PatternMatcher<C> {
        let mut patterns = self.patterns;
        patterns.extend(rhs.patterns);
        // Use from_patterns_with_intern to avoid re-computing VarIntern
        PatternMatcher::from_patterns_with_intern(patterns)
    }
}
