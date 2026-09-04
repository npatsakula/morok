//! Early reject: a compiled pattern is only dispatched when the root's direct children
//! carry every op kind its fixed-position sources demand.
//!
//! Tinygrad equivalent: `UPat.early_reject` (uop/ops.py:1349-1352) checked against
//! `UOp._src_ops` in `PatternMatcher.rewrite` (uop/ops.py:1480-1482).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use svod_macros::patterns;
use test_case::test_case;

use crate::op::OpMask;
use crate::op::pattern_derived::OpKey;
use crate::pattern::{RewriteResult, SimplifiedPatternMatcher, TypedPatternMatcher};
use crate::rewrite::engine::graph_rewrite;
use crate::types::{BinaryOp, TernaryOp, UnaryOp};
use crate::{ConstValue, Op, UOp};
use svod_dtype::DType;

fn int(value: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(value))
}

fn var(name: &str) -> Arc<UOp> {
    UOp::var(name, DType::Int32, 0, 1024)
}

fn bin(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Binary(op, lhs, rhs), DType::Int32)
}

fn neg(src: &Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Unary(UnaryOp::Neg, src.clone()), DType::Int32)
}

fn mask(keys: &[OpKey]) -> OpMask {
    keys.iter().cloned().collect()
}

/// Root node kinds used by the dispatch tables below.
#[derive(Debug, Clone, Copy)]
enum Node {
    /// `Add(Mul(1, 2), 3)` — children are MUL and CONST.
    AddMulConst,
    /// `Add(3, 4)` — both children are CONST.
    AddConstConst,
    /// `Add(x, y)` — both children are DEFINE_VAR.
    AddVarVar,
    /// `Neg(x)` — one DEFINE_VAR child.
    NegVar,
}

impl Node {
    fn build(self) -> Arc<UOp> {
        match self {
            Self::AddMulConst => bin(BinaryOp::Add, bin(BinaryOp::Mul, int(1), int(2)), int(3)),
            Self::AddConstConst => bin(BinaryOp::Add, int(3), int(4)),
            Self::AddVarVar => bin(BinaryOp::Add, var("a"), var("b")),
            Self::NegVar => neg(&var("a")),
        }
    }

    fn key(self) -> OpKey {
        match self {
            Self::AddMulConst | Self::AddConstConst | Self::AddVarVar => OpKey::Binary(BinaryOp::Add),
            Self::NegVar => OpKey::Unary(UnaryOp::Neg),
        }
    }
}

// =============================================================================
// Dispatch
// =============================================================================

/// Counting closure registered with `early_reject`; returns how many times it was entered.
fn dispatch_count(node: Node, early_reject: &[OpKey]) -> usize {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);

    let mut matcher = SimplifiedPatternMatcher::<()>::new();
    matcher.add_rejecting(&[node.key()], early_reject, move |_uop, _ctx| {
        counter.fetch_add(1, Ordering::Relaxed);
        RewriteResult::NoMatch
    });
    matcher.rewrite(&node.build(), &mut ());
    calls.load(Ordering::Relaxed)
}

#[test_case(Node::AddMulConst, &[], 1; "no requirement always dispatches")]
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Mul)], 1; "required mul child present")]
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Mul), OpKey::Const], 1; "both required present")]
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Add)], 0; "required add child absent")]
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Mul), OpKey::Binary(BinaryOp::Sub)], 0; "one of two required absent")]
#[test_case(Node::AddConstConst, &[OpKey::Const], 1; "const present twice")]
#[test_case(Node::AddConstConst, &[OpKey::Binary(BinaryOp::Mul)], 0; "mul absent among consts")]
#[test_case(Node::AddVarVar, &[OpKey::DefineVar], 1; "define var present")]
#[test_case(Node::AddVarVar, &[OpKey::Const], 0; "const absent among vars")]
#[test_case(Node::NegVar, &[OpKey::DefineVar], 1; "unary child present")]
#[test_case(Node::NegVar, &[OpKey::Const], 0; "unary child absent")]
fn dispatch_respects_early_reject(node: Node, early_reject: &[OpKey], expected: usize) {
    assert_eq!(dispatch_count(node, early_reject), expected);
}

// =============================================================================
// Derivation from `patterns!`
// =============================================================================

/// Requirement the macro derived for the single rule of a one-rule matcher.
fn derived(matcher: &TypedPatternMatcher<()>, key: OpKey) -> OpMask {
    let rejects = matcher.early_rejects(&key);
    assert_eq!(rejects.len(), 1, "expected exactly one entry under {key:?}");
    rejects[0]
}

/// The requirement is the union of the rule's fixed-position sources — from positional
/// args, struct fields and permutations alike — and only sources that pin exactly one
/// op kind contribute (Tinygrad's `len(pp.op) == 1`).
#[test]
fn derived_requirement_is_the_union_of_single_kind_sources() {
    let mul = mask(&[OpKey::Binary(BinaryOp::Mul)]);

    assert_eq!(derived(&patterns! { Add(Mul(a, b), c) => a.mul(&b.add(c)) }, OpKey::Binary(BinaryOp::Add)), mul);
    assert_eq!(
        derived(&patterns! { Add[Mul(a, b), c] => a.mul(&b.add(c)) }, OpKey::Binary(BinaryOp::Add)),
        mul,
        "permuted sources"
    );
    assert_eq!(derived(&patterns! { Add(x, @zero) => x }, OpKey::Binary(BinaryOp::Add)), mask(&[OpKey::Const]));
    assert_eq!(
        derived(
            &patterns! { Reshape { src: Cast { src: inner, dtype: _d }, new_shape: _s } => Some(inner.clone()) },
            OpKey::Reshape
        ),
        mask(&[OpKey::Cast]),
        "struct fields",
    );
    assert_eq!(
        derived(
            &patterns! { Where(Lt(a, b), Cast { src: c, dtype: _d }, e) => Some(a.add(b).add(c).add(e)) },
            OpKey::Ternary(TernaryOp::Where),
        ),
        mask(&[OpKey::Binary(BinaryOp::Lt), OpKey::Cast]),
    );

    // `@anyconst` admits both CONST and VCONST, so it constrains the child set not at all.
    assert!(
        derived(&patterns! { Add(x, c @anyconst(_vals)) => Some(x.add(c)) }, OpKey::Binary(BinaryOp::Add)).is_empty()
    );
    // Verbatim (non-child) fields are not sources.
    assert!(
        derived(&patterns! { Cast { src: x, dtype: DType::Scalar(_) } => Some(x.clone()) }, OpKey::Cast).is_empty()
    );
}

/// Bare-variable sources constrain nothing, so the rule stays dispatchable everywhere.
#[test]
fn wildcard_sources_are_never_rejected() {
    let matcher = patterns! { Add(x, y) => y.add(x), Neg(x) => x.clone() };
    for node in [Node::AddMulConst, Node::AddConstConst, Node::AddVarVar, Node::NegVar] {
        assert!(matcher.early_rejects(&node.key()).iter().all(|reject| reject.is_empty()), "{node:?}");
        assert!(!matches!(matcher.rewrite(&node.build(), &mut ()), RewriteResult::NoMatch), "{node:?}");
    }
}

/// A wildcard rule has no root key at all and must run for every op, including leaves.
#[test]
fn wildcard_rule_runs_on_childless_node() {
    let matcher = patterns! { x if x.op().children().is_empty() => Some(int(7)) };
    assert_eq!(matcher.wildcard_count(), 1);

    for node in [int(1), var("a"), bin(BinaryOp::Add, int(1), int(2))] {
        let expected = node.op().children().is_empty();
        assert_eq!(!matches!(matcher.rewrite(&node, &mut ()), RewriteResult::NoMatch), expected);
    }
}

// =============================================================================
// Equivalence
// =============================================================================

/// Every entry skipped by an early reject could not have matched, so clearing all rejects
/// must yield the pointer-identical graph — hash consing makes `ptr_eq` the exact check.
/// The rules below exercise single-op, const, struct-field, permuted and wildcard sources.
#[test]
fn early_reject_preserves_rewrite_results() {
    let matcher = patterns! {
        Add(x, @zero) => x,
        Mul(x, @one) => x,
        Add[Mul(a, b), Mul(c, d)] if Arc::ptr_eq(a, c) => Some(a.mul(&b.add(d))),
        Sub(Add(a, b), c) if Arc::ptr_eq(b, c) => Some(a.clone()),
        Neg(Neg(x)) => x,
        Cast { src: Cast { src: inner, dtype: _d }, dtype: outer } => inner.cast(outer.clone()),
        Where(Const(ConstValue::Bool(true)), t, _f) => t,
    };
    let permissive = matcher.without_early_reject();
    assert!(permissive.early_rejects(&OpKey::Binary(BinaryOp::Add)).iter().all(|reject| reject.is_empty()));
    assert!(matcher.early_rejects(&OpKey::Binary(BinaryOp::Add)).iter().any(|m| !m.is_empty()));

    let (x, y) = (var("x"), var("y"));
    // Matching, non-matching and nested graphs for the rules above.
    let graphs = [
        bin(BinaryOp::Add, x.clone(), int(0)),
        bin(BinaryOp::Mul, x.clone(), int(1)),
        bin(BinaryOp::Add, bin(BinaryOp::Mul, x.clone(), y.clone()), bin(BinaryOp::Mul, x.clone(), int(3))),
        bin(BinaryOp::Sub, bin(BinaryOp::Add, x.clone(), y.clone()), y.clone()),
        bin(BinaryOp::Sub, bin(BinaryOp::Add, x.clone(), y.clone()), int(5)),
        neg(&neg(&x)),
        neg(&x),
        x.clone().cast(DType::Int64).cast(DType::Float32),
        UOp::try_where(UOp::const_(DType::Bool, ConstValue::Bool(true)), x.clone(), y.clone())
            .expect("where over int32"),
        bin(BinaryOp::Add, bin(BinaryOp::Add, x.clone(), int(0)), bin(BinaryOp::Mul, y.clone(), int(1))),
        bin(BinaryOp::Add, x, y),
    ];
    for graph in graphs {
        let rejected = graph_rewrite(&matcher, graph.clone(), &mut ());
        let permissive = graph_rewrite(&permissive, graph.clone(), &mut ());
        assert!(Arc::ptr_eq(&rejected, &permissive), "diverged on {:?}", graph.op());
    }
}

// =============================================================================
// src_ops / OpMask
// =============================================================================

/// `src_ops` holds the kinds of the direct children only — duplicates collapse, and
/// grandchildren (the ADD under `AddMulConst`'s MUL) do not leak in.
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Mul), OpKey::Const]; "mul and const children")]
#[test_case(Node::AddConstConst, &[OpKey::Const]; "duplicate const children collapse")]
#[test_case(Node::AddVarVar, &[OpKey::DefineVar]; "define var children")]
#[test_case(Node::NegVar, &[OpKey::DefineVar]; "single unary child")]
fn src_ops_holds_direct_child_kinds(node: Node, expected: &[OpKey]) {
    let node = node.build();
    assert_eq!(node.src_ops(), mask(expected));
    assert!(!mask(&[OpKey::Binary(BinaryOp::Add)]).is_subset_of(node.src_ops()), "grandchildren must not leak");
    // Leaves carry the empty mask, which is a subset of everything and so rejects nothing.
    assert!(int(1).src_ops().is_empty());
    assert!(OpMask::EMPTY.is_subset_of(node.src_ops()));
}

/// Every op kind gets its own bit — grouped ops included, so `Add` never masks `Mul`.
#[test]
fn op_keys_have_distinct_bits() {
    let keys = [
        OpKey::Const,
        OpKey::DefineVar,
        OpKey::Cast,
        OpKey::BitCast,
        OpKey::Reshape,
        OpKey::Binary(BinaryOp::Add),
        OpKey::Binary(BinaryOp::Mul),
        OpKey::Unary(UnaryOp::Neg),
        OpKey::Unary(UnaryOp::Sqrt),
        OpKey::Ternary(TernaryOp::Where),
        OpKey::Ternary(TernaryOp::MulAcc),
    ];
    for (i, a) in keys.iter().enumerate() {
        for (j, b) in keys.iter().enumerate() {
            assert_eq!(i == j, mask(std::slice::from_ref(a)).is_subset_of(mask(std::slice::from_ref(b))));
        }
    }
    assert!(keys.iter().all(|key| key.index() < crate::op::pattern_derived::OP_KEY_COUNT));
}
