//! PatternMatcher for applying rewrite rules.
//!
//! PatternMatcher holds a collection of patterns and their rewrite functions,
//! and efficiently matches them against UOps using operation-based indexing.
//!
//! This implementation follows Tinygrad's pdict approach: patterns are indexed
//! by the specific operations they match, with wildcard patterns stored separately.

use std::collections::HashMap;
use std::rc::Rc;

use morok_ir::{BinaryOp, Op, TernaryOp, UOp, UnaryOp};

use super::upat::{OpFilter, UPat};

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

/// Rewrite function type.
///
/// Takes the variable bindings from a successful pattern match and returns
/// a RewriteResult indicating whether the rewrite should be applied.
pub type RewriteFn = Box<dyn Fn(&HashMap<String, Rc<UOp>>) -> RewriteResult>;

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
            OpFilter::Discriminant(_disc) => {
                // For discriminant-based filters, we can't enumerate all matching keys
                // without knowing which Op variant it matches. This is used for
                // operations that don't have sub-types (like Const, Noop, etc.)
                // We'll handle this by treating it as a wildcard for now.
                // Better solution: OpFilter should use OpKey directly instead of discriminants.
                vec![]
            }
        }
    }
}

/// Pattern matcher that applies rewrite rules to UOps.
///
/// Follows Tinygrad's pdict design:
/// - Patterns are indexed by the specific operations they match
/// - Wildcard patterns (match any op) are stored separately
/// - When rewriting, we check indexed patterns first, then wildcards
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
///     Box::new(|bindings| {
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
/// let matcher = PatternMatcher::new(patterns);
/// ```
pub struct PatternMatcher {
    /// All patterns with their rewrite functions
    patterns: Vec<(UPat, RewriteFn)>,

    /// Pattern dictionary: maps operation keys to pattern indices.
    /// This is the main optimization - we only try patterns that can match.
    pdict: HashMap<OpKey, Vec<usize>>,

    /// Indices of patterns that match any operation (no op filter).
    /// These are checked for every UOp after the op-specific patterns.
    wildcard_indices: Vec<usize>,
}

impl PatternMatcher {
    /// Create a new PatternMatcher from a list of patterns and rewrite functions.
    ///
    /// Patterns are tried in the order given, but indexed patterns are checked
    /// before wildcard patterns for efficiency.
    pub fn new(patterns: Vec<(UPat, RewriteFn)>) -> Self {
        let mut pdict: HashMap<OpKey, Vec<usize>> = HashMap::new();
        let mut wildcard_indices = Vec::new();

        // Build pattern dictionary
        for (idx, (pattern, _)) in patterns.iter().enumerate() {
            let op_keys = Self::extract_op_keys(pattern);

            if op_keys.is_empty() {
                // Pattern matches any operation - add to wildcard list
                wildcard_indices.push(idx);
            } else {
                // Pattern matches specific operations - add to each key's list
                for key in op_keys {
                    pdict.entry(key).or_default().push(idx);
                }
            }
        }

        Self { patterns, pdict, wildcard_indices }
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
    /// Returns the result of the rewrite attempt:
    /// - NoMatch: No pattern matched or all rewrite functions declined
    /// - Rewritten(uop): A pattern matched and returned a replacement
    /// - Gate(uop): A pattern matched and indicates bottom-up gate (process children)
    ///
    /// Patterns are tried in this order:
    /// 1. Patterns indexed under this op's OpKey
    /// 2. Wildcard patterns (match any op)
    pub fn rewrite(&self, uop: &Rc<UOp>) -> RewriteResult {
        let op_key = OpKey::from_op(uop.op());

        // Collect candidate pattern indices
        let mut candidates: Vec<&usize> = Vec::new();

        // First, add patterns specific to this operation
        if let Some(indices) = self.pdict.get(&op_key) {
            candidates.extend(indices.iter());
        }

        // Then, add wildcard patterns
        candidates.extend(self.wildcard_indices.iter());

        // Try each candidate pattern
        for idx in candidates {
            let (pattern, rewrite_fn) = &self.patterns[*idx];

            // Try to match the pattern
            let matches = pattern.match_uop(uop);

            // If pattern matched, try to apply rewrite function
            for bindings in matches {
                match rewrite_fn(&bindings) {
                    RewriteResult::NoMatch => continue,
                    result @ (RewriteResult::Rewritten(_) | RewriteResult::Gate(_)) => {
                        return result;
                    }
                }
            }
        }

        RewriteResult::NoMatch
    }
}
