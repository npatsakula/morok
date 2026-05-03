//! Multi-index linearization pass.
//!
//! Transforms `INDEX(buffer, [i, j, k])` → `INDEX(buffer, [linear_offset])`
//! using row-major linearization.
//!
//! This moves the multi-index → linear offset computation from codegen
//! to schedule, eliminating duplicated logic in LLVM and C backends.
//!
//! # Row-Major Linearization
//!
//! For a 3D index `[i, j, k]` with dimensions `[D0, D1, D2]`:
//! - Strides: `[D1*D2, D2, 1]`
//! - Linear offset: `i*(D1*D2) + j*D2 + k`
//!
//! # Dimension Extraction
//!
//! Dimensions are extracted from index expressions, not buffer shape:
//! - Direct RANGE: use `RANGE.end`
//! - DefineVar: use `max_val + 1`
//! - Complex expressions: multiply all contained RANGE sizes
//! - Fallback: vmin/vmax range analysis
//!
//! # Vectorized Indices
//!
//! When any index is vectorized (from UPCAST), the linearization is applied
//! element-wise, producing a vector of linear offsets for gather/scatter.

use std::sync::Arc;

use morok_ir::{BinaryOp, ConstValue, DType, Op, UOp};
use smallvec::SmallVec;
use tracing::trace;

use crate::TypedPatternMatcher;

/// Count divmod operations (Idiv + Mod) in an expression tree.
///
/// The helper itself is ported from Tinygrad `simplify.py` `count_divmod`,
/// where it gates END/REDUCE range merging in `simplify_merge_adjacent`.
/// Morok reuses the same complexity proxy to decide whether multi-index
/// INDEX flattening introduces redundant `%` / `//` chains; that
/// flattening pass is Morok-specific (tinygrad backends consume multi-
/// index INDEX directly), so the borrowed metric is the only Tinygrad
/// link here.
pub fn count_divmod(uop: &Arc<UOp>) -> usize {
    uop.toposort().iter().filter(|n| matches!(n.op(), Op::Binary(BinaryOp::Idiv | BinaryOp::Mod, _, _))).count()
}

/// Extract dimension from an index expression.
///
/// Index expressions can be:
/// - Direct RANGE - use its size
/// - DefineVar - use max_val + 1
/// - WHERE(cond, idx, Invalid) from PAD - extract actual input dim from validity
/// - Complex expression with RANGE ops (from shift_to) - multiply all RANGE sizes
///
/// This handles the transformation output from rangeify where indices
/// become expressions like `Add(Mul(Thread, stride), Loop)`.
pub fn extract_index_dimension(idx_uop: &Arc<UOp>) -> Option<i64> {
    // Case 0: WHERE(cond, idx, Invalid) from PAD
    // The RANGE inside is the OUTPUT range, which is larger than the buffer dimension.
    // Extract the actual input dimension from the validity condition.
    if let Op::Ternary(morok_ir::TernaryOp::Where, cond, true_val, false_val) = idx_uop.op()
        && matches!(false_val.op(), Op::Invalid)
    {
        return extract_dim_from_validity(cond, true_val);
    }

    // Case 1: Direct RANGE - use its size directly
    if let Op::Range { end, .. } = idx_uop.op() {
        if let Op::Const(cv) = end.op()
            && let ConstValue::Int(size) = cv.0
        {
            return Some(size);
        }
        return None; // Symbolic range size
    }

    // Case 2: DefineVar - use max_val + 1
    if let Op::DefineVar { max_val, .. } = idx_uop.op() {
        return Some(*max_val + 1);
    }

    // Case 3: Expression containing RANGE ops (from shift_to transforms)
    // Multiply all RANGE sizes in the expression to get total iteration count
    let mut product = 1i64;
    let mut found_range = false;

    for node in idx_uop.toposort() {
        if let Op::Range { end, .. } = node.op() {
            if let Op::Const(cv) = end.op()
                && let ConstValue::Int(size) = cv.0
            {
                product *= size;
                found_range = true;
            } else {
                return None; // Symbolic range size
            }
        }
    }

    if found_range && product > 0 {
        Some(product)
    } else {
        // Fallback: try vmin/vmax range analysis
        // vmin/vmax give bounds [min, max], so dimension is max - min + 1
        match (idx_uop.vmin(), idx_uop.vmax()) {
            (ConstValue::Int(min), ConstValue::Int(max)) if max >= min => Some(max - min + 1),
            _ => None,
        }
    }
}

/// Extract the actual buffer dimension from a PAD validity condition.
///
/// PAD creates `WHERE(valid, adjusted_idx, Invalid)` where:
/// - `valid = (rng >= begin) AND (rng < shape + begin)` (possibly simplified)
/// - `adjusted_idx = rng - begin`
/// - The actual buffer dimension is `shape` (not the output range size)
///
/// After symbolic simplification, common patterns:
/// - begin=0: `CMPLT(rng, shape)` → dim = shape
/// - end=0 (begin>0): `CMPGE(rng, begin)` → dim = rng.end - begin
/// - both nonzero: `AND(CMPGE(rng, begin), CMPLT(rng, shape+begin))` → dim = shape
fn extract_dim_from_validity(cond: &Arc<UOp>, true_val: &Arc<UOp>) -> Option<i64> {
    // Pattern 1: CMPLT(rng, CONST(upper)) — begin=0, dim = upper
    if let Op::Binary(BinaryOp::Lt, _rng, upper) = cond.op()
        && let Some(u) = const_int(upper)
    {
        return Some(u);
    }

    // Pattern 2: AND(CMPGE(rng, CONST(begin)), CMPLT(rng, CONST(upper))) — dim = upper - begin
    if let Op::Binary(BinaryOp::And, left, right) = cond.op()
        && let Some((begin, upper)) = extract_ge_lt_bounds(left, right).or_else(|| extract_ge_lt_bounds(right, left))
    {
        return Some(upper - begin);
    }

    // Pattern 3: CMPGE(rng, CONST(begin)) — end=0, extract dim from true_val
    // adjusted_idx = rng - begin, dim = rng.end - begin
    if let Op::Binary(BinaryOp::Ge, rng, begin_uop) = cond.op()
        && let Some(begin) = const_int(begin_uop)
        && let Op::Range { end, .. } = rng.op()
        && let Some(rng_end) = const_int(end)
    {
        return Some(rng_end - begin);
    }

    // Fallback: use vmin/vmax of the true branch (adjusted index)
    match (true_val.vmin(), true_val.vmax()) {
        (ConstValue::Int(min), ConstValue::Int(max)) if max >= min => Some(max - min + 1),
        _ => None,
    }
}

fn as_index_expr(expr: Arc<UOp>) -> Arc<UOp> {
    if expr.dtype() == DType::Index { expr } else { expr.cast(DType::Index) }
}

fn extract_index_dimension_expr(idx_uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Case 0: WHERE(cond, idx, Invalid) from PAD
    if let Op::Ternary(morok_ir::TernaryOp::Where, cond, true_val, false_val) = idx_uop.op()
        && matches!(false_val.op(), Op::Invalid)
    {
        return extract_dim_from_validity(cond, true_val).map(UOp::index_const);
    }

    // Case 1: Direct RANGE - use its end expression directly (supports symbolic).
    if let Op::Range { end, .. } = idx_uop.op() {
        return Some(as_index_expr(end.clone()));
    }

    // Case 2: DefineVar - conservative bound fallback.
    if let Op::DefineVar { max_val, .. } = idx_uop.op() {
        return Some(UOp::index_const(*max_val + 1));
    }

    // Case 3: Expression containing RANGE ops - multiply all RANGE ends.
    let mut product: Option<Arc<UOp>> = None;
    for node in idx_uop.toposort() {
        if let Op::Range { end, .. } = node.op() {
            let dim = as_index_expr(end.clone());
            product = Some(match product {
                None => dim,
                Some(acc) => mul_index_expr(acc, dim),
            });
        }
    }
    if product.is_some() {
        return product;
    }

    // Fallback: vmin/vmax range analysis for fully concrete bounds.
    match (idx_uop.vmin(), idx_uop.vmax()) {
        (ConstValue::Int(min), ConstValue::Int(max)) if max >= min => Some(UOp::index_const(max - min + 1)),
        _ => None,
    }
}

/// Extract an i64 constant from a UOp.
fn const_int(uop: &Arc<UOp>) -> Option<i64> {
    if let Op::Const(cv) = uop.op()
        && let ConstValue::Int(v) = cv.0
    {
        return Some(v);
    }
    None
}

fn const_index_value(uop: &Arc<UOp>) -> Option<i64> {
    if let Op::Const(cv) = uop.op() {
        match cv.0 {
            ConstValue::Int(v) => Some(v),
            ConstValue::UInt(v) => i64::try_from(v).ok(),
            _ => None,
        }
    } else {
        None
    }
}

fn mul_index_expr(lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    match (const_index_value(&lhs), const_index_value(&rhs)) {
        (Some(0), _) | (_, Some(0)) => UOp::index_const(0),
        (Some(1), _) => rhs,
        (_, Some(1)) => lhs,
        (Some(a), Some(b)) => a
            .checked_mul(b)
            .map(UOp::index_const)
            .unwrap_or_else(|| UOp::new(Op::Binary(BinaryOp::Mul, lhs, rhs), DType::Index)),
        _ => UOp::new(Op::Binary(BinaryOp::Mul, lhs, rhs), DType::Index),
    }
}

fn add_index_expr(lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    match (const_index_value(&lhs), const_index_value(&rhs)) {
        (Some(0), _) => rhs,
        (_, Some(0)) => lhs,
        (Some(a), Some(b)) => a
            .checked_add(b)
            .map(UOp::index_const)
            .unwrap_or_else(|| UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::Index)),
        _ => UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::Index),
    }
}

/// Extract begin and upper bounds from a pair that might be (CMPGE, CMPLT).
fn extract_ge_lt_bounds(maybe_ge: &Arc<UOp>, maybe_lt: &Arc<UOp>) -> Option<(i64, i64)> {
    let Op::Binary(BinaryOp::Ge, range_ge, begin_uop) = maybe_ge.op() else { return None };
    let Op::Binary(BinaryOp::Lt, range_lt, upper_uop) = maybe_lt.op() else { return None };
    // Both conditions must reference the same RANGE variable
    if !Arc::ptr_eq(range_ge, range_lt) {
        return None;
    }
    let begin = const_int(begin_uop)?;
    let upper = const_int(upper_uop)?;
    Some((begin, upper))
}

/// Compute row-major strides from dimensions.
///
/// For dims `[D0, D1, D2]`, strides are `[D1*D2, D2, 1]`.
pub fn compute_row_major_strides(dims: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

fn compute_row_major_strides_expr(dims: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    let mut strides: Vec<Arc<UOp>> = (0..dims.len()).map(|_| UOp::index_const(1)).collect();
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = mul_index_expr(strides[i + 1].clone(), dims[i + 1].clone());
    }
    strides
}

/// Check if any index in the list is vectorized.
fn any_index_vectorized(indices: &[Arc<UOp>]) -> bool {
    indices.iter().any(|idx| idx.dtype().vcount() > 1)
}

/// Get the vector count from the first vectorized index, or 1 if none.
fn get_vector_count(indices: &[Arc<UOp>]) -> usize {
    indices
        .iter()
        .find_map(|idx| {
            let vc = idx.dtype().vcount();
            if vc > 1 { Some(vc) } else { None }
        })
        .unwrap_or(1)
}

fn build_linear_index_expr(indices: &[Arc<UOp>], strides: &[Arc<UOp>]) -> Arc<UOp> {
    let mut linear = UOp::index_const(0);
    for (idx, stride) in indices.iter().zip(strides.iter()) {
        let term = match const_index_value(stride) {
            Some(0) => continue,
            Some(1) => idx.clone(),
            _ => mul_index_expr(idx.clone(), stride.clone()),
        };
        linear = add_index_expr(linear, term);
    }
    linear
}

/// Build a linear index expression from multi-dimensional indices and strides.
///
/// Computes: `indices[0] * strides[0] + indices[1] * strides[1] + ...`
pub fn build_linear_index(indices: &[Arc<UOp>], strides: &[i64]) -> Arc<UOp> {
    // Start with zero
    let mut linear = UOp::index_const(0);

    for (idx, &stride) in indices.iter().zip(strides.iter()) {
        if stride == 0 {
            // Skip zero-stride dimensions (scalar broadcast)
            continue;
        }

        let term = if stride == 1 {
            // Optimization: avoid multiplication by 1
            idx.clone()
        } else {
            let stride_const = UOp::index_const(stride);
            UOp::new(Op::Binary(BinaryOp::Mul, idx.clone(), stride_const), DType::Index)
        };

        // Check if linear is still zero (first iteration)
        if let Op::Const(cv) = linear.op()
            && matches!(cv.0, ConstValue::Int(0))
        {
            linear = term;
        } else {
            linear = UOp::new(Op::Binary(BinaryOp::Add, linear, term), DType::Index);
        }
    }

    linear
}

fn build_vectorized_linear_index_expr(indices: &[Arc<UOp>], strides: &[Arc<UOp>], vcount: usize) -> Arc<UOp> {
    let lane_indices: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
        .map(|lane| {
            let scalar_indices: Vec<Arc<UOp>> = indices
                .iter()
                .map(|idx| if idx.dtype().vcount() > 1 { idx.gep(vec![lane]) } else { idx.clone() })
                .collect();
            build_linear_index_expr(&scalar_indices, strides)
        })
        .collect();

    UOp::vectorize(lane_indices)
}

/// Pattern matcher to linearize multi-index INDEX operations.
///
/// Transforms:
/// - `INDEX(buffer, [i, j, k])` → `INDEX(buffer, [linear])`
///
/// Where `linear = i * (D1*D2) + j * D2 + k` for row-major layout.
///
/// This pass normalizes every resolvable multi-index to a single index expression.
/// Codegen assumes this normalization and does not re-linearize indices.
pub fn pm_linearize_multi_index() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // Match INDEX with multiple indices
        idx @ Index { buffer, indices, gate } if indices.len() > 1 => |idx, buffer, indices, gate| {
            // Extract dimensions from index expressions.
            let dims: Option<Vec<Arc<UOp>>> = indices
                .iter()
                .map(extract_index_dimension_expr)
                .collect();

            let dims = match dims {
                Some(d) => d,
                None => {
                    trace!(
                        uop_id = idx.id,
                        buffer_id = buffer.id,
                        "linearize_multi_index: couldn't extract all dimensions, skipping"
                    );
                    return None;
                }
            };

            // Compute row-major strides from dimension expressions.
            let strides = compute_row_major_strides_expr(&dims);

            // Check if any index is vectorized
            let is_vectorized = any_index_vectorized(indices);

            let linear_index = if is_vectorized {
                let vcount = get_vector_count(indices);
                build_vectorized_linear_index_expr(indices, &strides, vcount)
            } else {
                build_linear_index_expr(indices, &strides)
            };

            // Only flatten when divmod complexity does not increase.
            // Conv/attention indices that already factor into clean RANGE
            // components can otherwise pick up redundant `% / //` chains
            // after linearization. Backends linearize at render time as a
            // fallback.
            let original_divmod: usize = indices.iter().map(count_divmod).sum();
            let linearized_divmod = count_divmod(&linear_index);
            if linearized_divmod > original_divmod {
                trace!(
                    uop_id = idx.id,
                    original_divmod,
                    linearized_divmod,
                    "linearize_multi_index: rejected (would increase divmod), keeping multi-index"
                );
                return None;
            }

            trace!(
                uop_id = idx.id,
                index_dims = ?dims.iter().map(|d| format!("{:?}", d.op())).collect::<Vec<_>>(),
                original_divmod,
                linearized_divmod,
                "linearize_multi_index: linearizing {}-dimensional index",
                indices.len()
            );

            // Create new INDEX with single linear index, preserving gate and dtype
            let new_op = Op::Index {
                buffer: buffer.clone(),
                indices: smallvec::smallvec![linear_index],
                gate: gate.clone(),
            };

            Some(UOp::new(new_op, idx.dtype().clone()))
        },
    }
}

#[cfg(test)]
#[path = "../test/unit/passes/linearize_index_internal.rs"]
mod tests;
