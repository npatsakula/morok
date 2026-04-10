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

/// Check if a UOp is irreducible (Tinygrad: GroupOp.Irreducible = {CONST, DEFINE_VAR, SPECIAL, RANGE}).
fn is_irreducible(op: &Op) -> bool {
    matches!(op, Op::Const(..) | Op::DefineVar { .. } | Op::Special { .. } | Op::Range { .. })
}

/// Split an ADD-chain into individual addends.
fn split_add(expr: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match expr.op() {
        Op::Binary(BinaryOp::Add, left, right) => {
            let mut result = split_add(left);
            result.extend(split_add(right));
            result
        }
        _ => vec![expr.clone()],
    }
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
            uop_given_valid(&accumulated_valid, stmt, true)
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
/// When `try_simplex` is true (called from `simplify_valid`), also tries per-addend
/// simplex decomposition for constraints like `X0 + X1 + ... >= 1`.
fn uop_given_valid(valid: &Arc<UOp>, uop: &Arc<UOp>, try_simplex: bool) -> Arc<UOp> {
    use morok_ir::rewrite::graph_rewrite;

    // Parse valid into {expr: [lower_bound, upper_bound]}
    type BoundsEntry = (Arc<UOp>, Option<i64>, Option<i64>);
    let mut bounds: HashMap<u64, BoundsEntry> = HashMap::new();
    for stmt in split_and(valid) {
        if let Some((expr, is_upper, c)) = parse_valid(&stmt) {
            let entry = bounds.entry(expr.id).or_insert_with(|| (expr.clone(), None, None));
            if is_upper {
                match entry.2 {
                    None => entry.2 = Some(c),
                    Some(existing) if c < existing => entry.2 = Some(c),
                    _ => {}
                }
            } else {
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

    // Build candidate list: (original_expr, fake_var_with_tighter_bounds)
    // Tinygrad symbolic.py:288-292
    let mut all_candidates: Vec<(Arc<UOp>, Arc<UOp>)> = Vec::new();
    let mut uop = uop.clone();

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
        let fake_var = if expr.dtype() != fake_var.dtype() { fake_var.cast(expr.dtype()) } else { fake_var };
        all_candidates.push((expr.clone(), fake_var));

        // Per-candidate simplex logic (Tinygrad symbolic.py:294-309)
        if try_simplex {
            let mut candidate_sets: Vec<Vec<(Arc<UOp>, Arc<UOp>)>> = vec![vec![all_candidates.last().unwrap().clone()]];

            // Simplex detection: X0 + X1 + ... >= 1 where all Xi are irreducible with vmin >= 0
            if let Op::Binary(BinaryOp::Add, ..) = expr.op()
                && v0 == 1
            {
                let addends = split_add(expr);
                let all_irreducible_nonneg = addends.iter().all(|u| {
                    is_irreducible(u.op()) && {
                        let (vmin, _) = VminVmaxProperty::get(u);
                        matches!(vmin, ConstValue::Int(v) if *v >= 0)
                    }
                });
                if all_irreducible_nonneg {
                    let simplex_candidates: Vec<(Arc<UOp>, Arc<UOp>)> = addends
                        .iter()
                        .enumerate()
                        .map(|(j, xi)| {
                            let (_, xi_vmax) = VminVmaxProperty::get(xi);
                            let max_val = if let ConstValue::Int(v) = xi_vmax { *v } else { i64::MAX };
                            let fake = UOp::define_var(format!("_simplex_fake{j}"), 1, max_val);
                            let fake = if xi.dtype() != fake.dtype() { fake.cast(xi.dtype()) } else { fake };
                            (xi.clone(), fake)
                        })
                        .collect();
                    candidate_sets.push(simplex_candidates);
                }
            }

            for candidates in &candidate_sets {
                // Substitute each candidate independently
                let new_uops: Vec<Arc<UOp>> = candidates
                    .iter()
                    .map(|(x, new_x)| {
                        #[allow(clippy::mutable_key_type)]
                        let map: HashMap<UOpKey, Arc<UOp>> = [(UOpKey(x.clone()), new_x.clone())].into();
                        uop.substitute(&map)
                    })
                    .collect();
                // Skip if any branch doesn't contain the expression
                if new_uops.iter().any(|u| Arc::ptr_eq(u, &uop)) {
                    continue;
                }
                // Simplify each branch, substitute back, simplify again
                let simplified: Vec<Arc<UOp>> = candidates
                    .iter()
                    .zip(new_uops.iter())
                    .map(|((x, new_x), u)| {
                        let s = graph_rewrite(crate::symbolic::patterns::symbolic(), u.clone(), &mut ());
                        #[allow(clippy::mutable_key_type)]
                        let rev: HashMap<UOpKey, Arc<UOp>> = [(UOpKey(new_x.clone()), x.clone())].into();
                        graph_rewrite(crate::symbolic::patterns::symbolic(), s.substitute(&rev), &mut ())
                    })
                    .collect();
                // If all branches produce the same result, accept it
                if simplified.windows(2).all(|w| w[0].id == w[1].id) {
                    uop = simplified[0].clone();
                }
                // TODO: VECTORIZE partial simplification (Tinygrad lines 307-309) — add when needed
            }
        }
    }

    if all_candidates.is_empty() {
        return uop;
    }

    // Combined all-candidates substitution (Tinygrad symbolic.py:311-313)
    #[allow(clippy::mutable_key_type)]
    let sub_map: HashMap<UOpKey, Arc<UOp>> =
        all_candidates.iter().map(|(x, f)| (UOpKey(x.clone()), f.clone())).collect();
    let substituted = uop.substitute(&sub_map);
    if Arc::ptr_eq(&substituted, &uop) {
        return uop;
    }

    // Simplify with full symbolic (tier 2) including divmod rules
    let simplified = graph_rewrite(crate::symbolic::patterns::symbolic(), substituted, &mut ());

    // Substitute back and simplify again
    #[allow(clippy::mutable_key_type)]
    let reverse_map: HashMap<UOpKey, Arc<UOp>> =
        all_candidates.iter().map(|(x, f)| (UOpKey(f.clone()), x.clone())).collect();
    let result = simplified.substitute(&reverse_map);
    graph_rewrite(crate::symbolic::patterns::symbolic(), result, &mut ())
}

/// Simplify WHERE(cond, x, Invalid) using cond as bounds context
/// (Tinygrad symbolic.py:366-369).
///
/// Rewrites `x` using tighter bounds derived from the condition `cond`.
pub fn gated_given_valid(cond: &Arc<UOp>, x: &Arc<UOp>, invalid: &Arc<UOp>) -> Option<Arc<UOp>> {
    let new_x = uop_given_valid(cond, x, false);
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
