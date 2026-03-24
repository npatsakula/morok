//! Valid clause simplification (Tinygrad symbolic.py:262-384).
//!
//! Simplifies AND-chains of validity clauses and narrows WHERE-Invalid gates
//! using bound information from the condition.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue};
use morok_ir::uop::cached_property::CachedProperty;
use morok_ir::uop::properties::VminVmaxProperty;
use morok_ir::{Op, UOp, UOpKey};

use crate::TypedPatternMatcher;

/// Parse a validity clause into (expr, is_upper, bound).
///
/// - `X < c` -> `(X, true, c-1)` meaning X <= c-1
/// - `NOT(X < c)` encoded as `(X < c).ne(true)` -> `(X, false, c)` meaning X >= c
fn parse_valid(v: &Arc<UOp>) -> Option<(Arc<UOp>, bool, i64)> {
    // Pattern: (X < c).ne(True) -> X >= c
    if let Op::Binary(BinaryOp::Ne, lhs, rhs) = v.op()
        && let Op::Const(cv) = rhs.op()
        && cv.0 == ConstValue::Bool(true)
        && let Op::Binary(BinaryOp::Lt, x, c) = lhs.op()
        && x.dtype().is_int()
    {
        let (_, c_vmax) = VminVmaxProperty::get(c);
        if let ConstValue::Int(c_val) = c_vmax {
            return Some((x.clone(), false, *c_val));
        }
    }

    // Pattern: NOT(X < c) -> X >= c
    if let Op::Unary(morok_ir::types::UnaryOp::Not, inner) = v.op()
        && let Op::Binary(BinaryOp::Lt, x, c) = inner.op()
        && x.dtype().is_int()
    {
        let (_, c_vmax) = VminVmaxProperty::get(c);
        if let ConstValue::Int(c_val) = c_vmax {
            return Some((x.clone(), false, *c_val));
        }
    }

    // Pattern: X < c -> X <= c-1
    if let Op::Binary(BinaryOp::Lt, x, c) = v.op()
        && x.dtype().is_int()
    {
        let (_, c_vmax) = VminVmaxProperty::get(c);
        if let ConstValue::Int(c_val) = c_vmax {
            return Some((x.clone(), true, *c_val - 1));
        }
    }

    None
}

/// Split an AND-chain into individual clauses.
fn split_and(cond: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match cond.op() {
        Op::Binary(BinaryOp::And, left, right) => {
            let mut result = split_and(left);
            result.extend(split_and(right));
            result
        }
        _ => vec![cond.clone()],
    }
}

/// Combine clauses with AND, returning `true` const if empty.
fn join_and(clauses: &[Arc<UOp>]) -> Arc<UOp> {
    if clauses.is_empty() {
        return UOp::const_(DType::Bool, ConstValue::Bool(true));
    }
    clauses.iter().cloned().reduce(|a, b| a.and_(&b)).unwrap()
}

/// Simplify redundant AND clauses (Tinygrad symbolic.py:320-328).
///
/// Splits the AND into clauses, deduplicates, and for each clause tries to
/// simplify it given the already-accepted clauses as known-true constraints.
pub fn simplify_valid(valid: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Skip if the AND chain references INDEX nodes
    if valid.has_index_in_sources() {
        return None;
    }

    let mut clauses = split_and(valid);

    // Early exit: if no clause can be parsed into bounds, nothing to simplify.
    if !clauses.iter().any(|c| parse_valid(c).is_some()) {
        return None;
    }

    // Sort by priority: clauses whose parsed expr appears in other clauses' backward slices
    // should come first, so they're more likely to simplify later clauses.
    // This matches Tinygrad's _valid_priority.
    // Pre-compute backward slices once per clause (Tinygrad caches these as a property).
    let original_clauses = clauses.clone();
    let backward_slices: Vec<&HashSet<u64>> = original_clauses.iter().map(|c| c.backward_slice_ids()).collect();
    clauses.sort_by_key(|v| {
        let Some((expr, _, _)) = parse_valid(v) else { return 0i32 };
        let expr_id = expr.id;
        let mut priority = 0i32;
        for (i, other) in original_clauses.iter().enumerate() {
            if other.id == v.id {
                continue;
            }
            if backward_slices[i].contains(&expr_id) {
                priority -= 1;
            }
        }
        priority
    });

    // Deduplicate by id
    let mut seen = std::collections::HashSet::new();
    clauses.retain(|c| seen.insert(c.id));

    // Try to simplify each clause given previously accepted clauses
    let mut ret: Vec<Arc<UOp>> = Vec::new();
    for stmt in &clauses {
        let simplified = if ret.is_empty() {
            stmt.clone()
        } else {
            let accumulated_valid = join_and(&ret);
            uop_given_valid(&accumulated_valid, stmt)
        };
        ret.push(simplified);
    }

    // Only return if something changed
    if ret.len() == clauses.len() && ret.iter().zip(clauses.iter()).all(|(a, b)| a.id == b.id) {
        return None;
    }

    Some(join_and(&ret))
}

/// Simplify a UOp given that `valid` is known to be true (Tinygrad symbolic.py:277-314).
///
/// Parses validity clauses into bounds, creates substitute variables with
/// tighter ranges, and rewrites the uop.
///
/// This is the `try_simplex=False` version (used by `gated_given_valid`).
fn uop_given_valid(valid: &Arc<UOp>, uop: &Arc<UOp>) -> Arc<UOp> {
    // Parse valid into {expr: [lower_bound, upper_bound]}
    type Bounds = (Arc<UOp>, Option<i64>, Option<i64>);
    let mut bounds: HashMap<u64, Bounds> = HashMap::new();
    for stmt in split_and(valid) {
        if let Some((expr, is_upper, c)) = parse_valid(&stmt) {
            let entry = bounds.entry(expr.id).or_insert_with(|| (expr.clone(), None, None));
            if is_upper {
                // upper bound: X <= c
                match entry.2 {
                    None => entry.2 = Some(c),
                    Some(existing) if c < existing => entry.2 = Some(c),
                    _ => {}
                }
            } else {
                // lower bound: X >= c
                match entry.1 {
                    None => entry.1 = Some(c),
                    Some(existing) if c > existing => entry.1 = Some(c),
                    _ => {}
                }
            }
        }
    }

    if bounds.is_empty() {
        return uop.clone();
    }

    // Build substitution map: expr -> DefineVar with tighter bounds
    #[allow(clippy::mutable_key_type)]
    let mut sub_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    #[allow(clippy::mutable_key_type)]
    let mut reverse_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for (i, (_id, (expr, lower, upper))) in bounds.iter().enumerate() {
        let (expr_vmin, expr_vmax) = VminVmaxProperty::get(expr);
        let v0 = lower.unwrap_or_else(|| if let ConstValue::Int(v) = expr_vmin { *v } else { i64::MIN });
        let v1 = upper.unwrap_or_else(|| if let ConstValue::Int(v) = expr_vmax { *v } else { i64::MAX });

        // Skip if bounds didn't actually tighten
        let orig_min = if let ConstValue::Int(v) = expr_vmin { *v } else { i64::MIN };
        let orig_max = if let ConstValue::Int(v) = expr_vmax { *v } else { i64::MAX };
        if v0 == orig_min && v1 == orig_max {
            continue;
        }

        let fake_var = UOp::define_var(format!("_valid_fake{i}"), v0, v1);
        // Cast to match expr dtype if needed (define_var creates Index dtype)
        let fake_var = if expr.dtype() != fake_var.dtype() { fake_var.cast(expr.dtype()) } else { fake_var };

        sub_map.insert(UOpKey(expr.clone()), fake_var.clone());
        reverse_map.insert(UOpKey(fake_var), expr.clone());
    }

    if sub_map.is_empty() {
        return uop.clone();
    }

    // Substitute, simplify, substitute back
    let substituted = uop.substitute(&sub_map);
    if Arc::ptr_eq(&substituted, uop) {
        return uop.clone();
    }

    // Run symbolic simplification on substituted expression
    let simplified =
        morok_ir::rewrite::graph_rewrite(crate::symbolic::patterns::symbolic_simple(), substituted, &mut ());

    // Substitute back
    let result = simplified.substitute(&reverse_map);

    // Run simplification again after substituting back
    morok_ir::rewrite::graph_rewrite(crate::symbolic::patterns::symbolic_simple(), result, &mut ())
}

/// Simplify WHERE(cond, x, Invalid) using cond as bounds context
/// (Tinygrad symbolic.py:366-369).
///
/// Rewrites `x` using tighter bounds derived from the condition `cond`.
pub fn gated_given_valid(cond: &Arc<UOp>, x: &Arc<UOp>, invalid: &Arc<UOp>) -> Option<Arc<UOp>> {
    let new_x = uop_given_valid(cond, x);
    if new_x.id == x.id {
        return None;
    }
    UOp::try_where(cond.clone(), new_x, invalid.clone()).ok()
}

/// Pattern matcher for valid simplification.
pub fn pm_simplify_valid() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Simplify AND chains of validity clauses
        valid @ And(_, _) if valid.dtype() == DType::Bool
            => |valid| simplify_valid(valid),

        // Simplify WHERE(cond, x, Invalid) using bounds from cond
        Where(cond, x, inv) if matches!(inv.op(), Op::Invalid)
            => |cond, x, inv| gated_given_valid(cond, x, inv),
    }
}

/// Drop AND clauses from WHERE conditions when the clause's ranges
/// don't overlap with the gated expression's ranges.
///
/// Tinygrad: `pm_drop_and_clauses` (symbolic.py:343-346).
///
/// For `WHERE(AND(c1, c2, ..., cn), expr, INVALID)`:
/// - Keep clause ci if any of ci's RANGE ops also appear in expr's RANGE ops
/// - Drop clause ci if none of ci's RANGE ops appear in expr's RANGE ops
///
/// This is safe because dropped conditions are guarded at a higher level
/// (e.g., VALUE-level WHERE in Concat branching).
fn drop_and_clauses(cond: &Arc<UOp>, x: &Arc<UOp>, invalid: &Arc<UOp>) -> Option<Arc<UOp>> {
    use morok_ir::types::BinaryOp;

    let clauses = cond.split_uop(BinaryOp::And);
    if clauses.len() <= 1 {
        return None;
    }

    let x_range_ids: HashSet<u64> = x.ranges().iter().map(|r| r.id).collect();

    let mut keep = Vec::new();
    let mut dropped = false;
    for clause in &clauses {
        let clause_ranges = clause.ranges();
        if clause_ranges.iter().any(|r| x_range_ids.contains(&r.id)) {
            keep.push(clause.clone());
        } else {
            dropped = true;
        }
    }

    if !dropped {
        return None;
    }

    if keep.is_empty() {
        // All clauses dropped — gated expression has no ranges.
        // Keep the original condition to preserve safety.
        return None;
    }
    let new_cond = {
        let mut acc = keep[0].clone();
        for k in &keep[1..] {
            acc = acc.try_and_op(k).ok()?;
        }
        acc
    };

    UOp::try_where(new_cond, x.clone(), invalid.clone()).ok()
}

/// Pattern matcher that drops irrelevant AND clauses from WHERE-Invalid gates.
///
/// Tinygrad: `pm_drop_and_clauses` (symbolic.py:346).
pub fn pm_drop_and_clauses() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Where(cond, x, inv) if matches!(inv.op(), Op::Invalid)
            => |cond, x, inv| drop_and_clauses(cond, x, inv),
    }
}
