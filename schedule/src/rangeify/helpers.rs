//! Helper functions for rangeify pattern matching and transformations.

use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, Op, SInt, UOp};
use std::rc::Rc;

/// Check if value is identity for op (Add: 0, Mul: 1, And: -1, Or/Xor: 0).
pub fn is_identity_value(value: &ConstValue, op: &BinaryOp, is_right: bool) -> bool {
    match (op, value) {
        // Addition: x + 0 = x, 0 + x = x
        (BinaryOp::Add, ConstValue::Int(0)) => true,
        (BinaryOp::Add, ConstValue::Float(f)) if *f == 0.0 => true,

        // Subtraction: x - 0 = x (only right identity)
        (BinaryOp::Sub, ConstValue::Int(0)) if is_right => true,
        (BinaryOp::Sub, ConstValue::Float(f)) if is_right && *f == 0.0 => true,

        // Multiplication: x * 1 = x, 1 * x = x
        (BinaryOp::Mul, ConstValue::Int(1)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 1.0 => true,

        // Division: x / 1 = x (only right identity)
        (BinaryOp::Idiv, ConstValue::Int(1)) if is_right => true,
        (BinaryOp::Fdiv, ConstValue::Float(f)) if is_right && *f == 1.0 => true,

        // Bitwise Or: x | 0 = x, 0 | x = x
        (BinaryOp::Or, ConstValue::Int(0)) => true,

        // Bitwise Xor: x ^ 0 = x, 0 ^ x = x
        (BinaryOp::Xor, ConstValue::Int(0)) => true,

        // Bitwise And: x & all_ones = x, all_ones & x = x
        (BinaryOp::And, ConstValue::Int(-1)) => true, // -1 has all bits set

        _ => false,
    }
}

/// Check if value is zero/annihilator for op (Mul: 0, And: 0).
pub fn is_zero_value(value: &ConstValue, op: &BinaryOp) -> bool {
    match (op, value) {
        // Multiplication: x * 0 = 0
        (BinaryOp::Mul, ConstValue::Int(0)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 0.0 => true,

        // Bitwise And: x & 0 = 0
        (BinaryOp::And, ConstValue::Int(0)) => true,

        _ => false,
    }
}

/// Extract the constant value from a UOp if it's a CONST operation.
pub fn get_const_value(uop: &Rc<UOp>) -> Option<ConstValue> {
    match uop.op() {
        Op::Const(cv) => Some(cv.0),
        _ => None,
    }
}

/// Check if a UOp is a constant with a specific value.
pub fn is_const(uop: &Rc<UOp>, value: &ConstValue) -> bool {
    get_const_value(uop).as_ref() == Some(value)
}

/// Check if a UOp represents a zero-size tensor.
///
/// A tensor has zero size if any dimension in its shape is 0.
pub fn is_zero_size(uop: &Rc<UOp>) -> bool {
    uop.shape().ok().flatten().map(|shape| shape.iter().any(|dim| matches!(dim, SInt::Const(0)))).unwrap_or(false)
}

/// Check if a dtype is void (used for side-effecting operations).
pub fn is_void(dtype: &DType) -> bool {
    *dtype == DType::Void
}

/// Get the binary operation from a UOp if it's a BINARY operation.
pub fn get_binary_op(uop: &Rc<UOp>) -> Option<BinaryOp> {
    match uop.op() {
        Op::Binary(op, _, _) => Some(*op),
        _ => None,
    }
}

/// Transform ranges through a movement op (SHRINK, PERMUTE, FLIP, EXPAND, PAD, RESHAPE).
pub fn apply_movement_op(op: &Op, in_shape: &[morok_ir::SInt], rngs: &[Rc<UOp>]) -> Vec<Rc<UOp>> {
    use morok_ir::SInt;

    match op {
        // SHRINK: ranges[i] = rng[i] + begin[i]
        Op::Shrink { begins, .. } => {
            let begin_vals = extract_shape_values(begins);
            rngs.iter()
                .zip(begin_vals.iter())
                .map(|(rng, &begin)| {
                    if begin == 0 {
                        Rc::clone(rng)
                    } else {
                        let begin_uop = UOp::index_const(begin as i64);
                        rng.try_add(&begin_uop).unwrap()
                    }
                })
                .collect()
        }

        // PERMUTE: ranges = [rngs[inv_perm[i]] for i in 0..len]
        Op::Permute { axes, .. } => {
            let inv_perm = argsort(axes);
            inv_perm.iter().map(|&i| Rc::clone(&rngs[i])).collect()
        }

        // FLIP: ranges[i] = (shape[i]-1) - rng[i] if flip[i] else rng[i]
        Op::Flip { axes: flips, .. } => rngs
            .iter()
            .zip(in_shape.iter())
            .zip(flips.iter())
            .map(|((rng, shape), &flip)| {
                if !flip {
                    Rc::clone(rng)
                } else {
                    let shape_minus_1 = match shape {
                        SInt::Const(n) => UOp::index_const(*n as i64 - 1),
                        SInt::Symbolic(uop) => {
                            let one = UOp::index_const(1);
                            uop.try_sub(&one).unwrap()
                        }
                    };
                    shape_minus_1.try_sub(rng).unwrap()
                }
            })
            .collect(),

        // EXPAND: ranges[i] = 0 if expanding else rng[i]
        Op::Expand { new_shape, .. } => {
            let new_shape_vals = extract_shape_from_uop(new_shape);
            rngs.iter()
                .zip(in_shape.iter())
                .zip(new_shape_vals.iter())
                .map(|((rng, in_sh), out_sh)| {
                    let expanding = match (in_sh, out_sh) {
                        (SInt::Const(1), SInt::Const(n)) if *n > 1 => true,
                        (SInt::Const(1), SInt::Symbolic(_)) => true,
                        _ => false,
                    };
                    if expanding { UOp::index_const(0) } else { Rc::clone(rng) }
                })
                .collect()
        }

        // PAD: ranges[i] = valid_check.where(rng[i] - pad_begin, INVALID)
        Op::Pad { begin_pads, end_pads, .. } => {
            let begin_vals = extract_shape_values(begin_pads);
            let end_vals = extract_shape_values(end_pads);
            rngs.iter()
                .zip(in_shape.iter())
                .zip(begin_vals.iter().zip(end_vals.iter()))
                .map(|((rng, shape), (&begin, &end))| {
                    if begin == 0 && end == 0 {
                        return Rc::clone(rng);
                    }
                    let begin_uop = UOp::index_const(begin as i64);
                    let shape_plus_begin = match shape {
                        SInt::Const(n) => UOp::index_const(*n as i64 + begin as i64),
                        SInt::Symbolic(uop) => uop.try_add(&begin_uop).unwrap(),
                    };
                    // rng >= begin  (use !(rng < begin) implemented as (rng < begin) XOR true)
                    let too_low = rng.try_cmplt(&begin_uop).unwrap();
                    let true_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
                    let valid_low = too_low.try_xor_op(&true_val).unwrap();
                    // rng < shape + begin
                    let valid_high = rng.try_cmplt(&shape_plus_begin).unwrap();
                    // valid = valid_low & valid_high
                    let valid = valid_low.try_and_op(&valid_high).unwrap();
                    // Subtract padding: rng - begin
                    let adjusted_rng = rng.try_sub(&begin_uop).unwrap();
                    // Use invalid marker for out-of-bounds regions (will be masked by valid check)
                    UOp::try_where(valid, adjusted_rng, UOp::invalid_marker()).unwrap()
                })
                .collect()
        }

        // RESHAPE: Complex multi-dimensional index arithmetic
        Op::Reshape { new_shape, .. } => {
            let new_shape_vals = extract_shape_from_uop(new_shape);

            // Optimization: If in_shape == new_shape, this is a no-op reshape
            // Just pass through the ranges directly without modulo arithmetic
            if in_shape.len() == new_shape_vals.len() {
                let mut is_same_shape = true;
                for (in_dim, out_dim) in in_shape.iter().zip(new_shape_vals.iter()) {
                    match (in_dim, out_dim) {
                        (SInt::Const(a), SInt::Const(b)) if a == b => continue,
                        (SInt::Symbolic(a), SInt::Symbolic(b)) if a.id == b.id => continue,
                        _ => {
                            is_same_shape = false;
                            break;
                        }
                    }
                }
                if is_same_shape {
                    // No-op reshape: return ranges as-is
                    return rngs.to_vec();
                }
            }

            // Step 1: Flatten output indices
            let mut acc = UOp::index_const(1);
            let mut axes_in = Vec::new();
            for (shape_dim, rng) in new_shape_vals.iter().zip(rngs.iter()).rev() {
                let weighted = acc.try_mul(rng).unwrap();
                axes_in.push(weighted);
                acc = match shape_dim {
                    SInt::Const(n) => {
                        let n_uop = UOp::index_const(*n as i64);
                        acc.try_mul(&n_uop).unwrap()
                    }
                    SInt::Symbolic(uop) => acc.try_mul(uop).unwrap(),
                };
            }
            let combined_axes =
                axes_in.into_iter().reduce(|a, b| a.try_add(&b).unwrap()).unwrap_or_else(|| UOp::index_const(0));
            // Step 2: Unflatten into input shape dimensions
            let mut axes_out = Vec::new();
            let mut combined = combined_axes;
            for shape_dim in in_shape.iter().rev() {
                let shape_uop = match shape_dim {
                    SInt::Const(n) => UOp::index_const(*n as i64),
                    SInt::Symbolic(uop) => Rc::clone(uop),
                };
                let mod_result = combined.try_mod(&shape_uop).unwrap();
                axes_out.push(mod_result);
                combined = combined.try_div(&shape_uop).unwrap();
            }
            axes_out.reverse();
            axes_out
        }

        _ => panic!("apply_movement_op called with non-movement op: {:?}", op),
    }
}

/// Extract shape values from a UOp (for SHRINK begins/ends, PAD pads).
fn extract_shape_values(uop: &Rc<UOp>) -> Vec<usize> {
    match uop.op() {
        Op::Vectorize { elements } => elements
            .iter()
            .map(|elem| match elem.op() {
                Op::Const(cv) => match cv.0 {
                    ConstValue::Int(n) => n as usize,
                    _ => panic!("Expected int constant in vectorize"),
                },
                _ => panic!("Expected constant element in vectorize"),
            })
            .collect(),
        Op::Const(cv) => {
            // Single constant value - treat as 1-element vector
            match cv.0 {
                ConstValue::Int(n) => vec![n as usize],
                _ => panic!("Expected int constant"),
            }
        }
        _ => panic!("Expected vectorize or constant for shape values, got {:?}", uop.op()),
    }
}

/// Extract shape from a UOp (for RESHAPE new_shape, EXPAND new_shape).
fn extract_shape_from_uop(uop: &Rc<UOp>) -> Vec<morok_ir::SInt> {
    use morok_ir::SInt;
    match uop.op() {
        Op::Vectorize { elements } => elements
            .iter()
            .map(|elem| match elem.op() {
                Op::Const(cv) => match cv.0 {
                    ConstValue::Int(n) => SInt::Const(n as usize),
                    _ => SInt::Symbolic(Rc::clone(elem)),
                },
                _ => SInt::Symbolic(Rc::clone(elem)),
            })
            .collect(),
        Op::Const(cv) => {
            // Single constant - treat as 1-element shape
            match cv.0 {
                ConstValue::Int(n) => vec![SInt::Const(n as usize)],
                _ => panic!("Expected int constant for shape"),
            }
        }
        _ => panic!("Expected vectorize or constant for shape, got {:?}", uop.op()),
    }
}

/// Compute inverse permutation (argsort).
fn argsort(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Check if two range lists are pointer-equal (same UOps).
pub fn ranges_equal(ranges1: &[Rc<UOp>], ranges2: &[Rc<UOp>]) -> bool {
    ranges1.len() == ranges2.len() && ranges1.iter().zip(ranges2).all(|(r1, r2)| Rc::ptr_eq(r1, r2))
}

/// Check if op should always run (CONTIGUOUS, COPY, ASSIGN, NOOP).
pub fn is_always_run_op(op: &Op) -> bool {
    matches!(op, Op::Contiguous { .. } | Op::Copy { .. } | Op::Assign { .. } | Op::Noop)
}

/// Check if range is dead (size ≤ 1). Uses vmax analysis.
pub fn is_dead_axis(range: &Rc<UOp>) -> bool {
    if !matches!(range.op(), Op::Range { .. }) {
        return false;
    }

    // Use vmax analysis to detect dead ranges
    // A range is dead if vmax ≤ 0 (iterates 0 or 1 time)
    // For RANGE(end), vmax = end - 1, so vmax ≤ 0 means end ≤ 1
    match range.vmax() {
        ConstValue::Int(v) => *v <= 0,
        ConstValue::UInt(v) => *v == 0, // UInt can't be negative
        _ => false,                     // Symbolic or non-numeric vmax - can't prove dead
    }
}

/// Check if op is cheap to inline (CONST, Unary, Binary, Ternary, Cast, Gep, Vectorize).
pub fn is_cheap_to_inline(op: &Op) -> bool {
    matches!(
        op,
        // Nullary - always cheap
        Op::Const(_)
            | Op::Unique(_)
            | Op::Device(_)
            | Op::Noop
            | Op::DefineVar { .. }
            | Op::DefineReg { .. }
            | Op::VConst { .. }
            // Simple operations - cheap to recompute
            | Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::Cast { .. }
            | Op::BitCast { .. }
            // Vector operations - cheap
            | Op::Gep { .. }
            | Op::Vectorize { .. }
            // Index/pointer operations - cheap
            | Op::PointerIndex { .. }
    )
}

/// Check if a BUFFERIZE operation is for local memory (intermediate buffer).
///
/// Only LOCAL BUFFERIZE operations are candidates for removal by the
/// "cheap to inline" optimization. GLOBAL BUFFERIZE operations represent
/// user-requested output materialization and should never be removed.
pub fn is_local_bufferize(uop: &Rc<UOp>) -> bool {
    if let Op::Bufferize { opts, .. } = uop.op() { opts.addrspace == morok_ir::AddrSpace::Local } else { false }
}

/// Check if UOp has no RANGE dependencies.
pub fn no_range(uop: &Rc<UOp>) -> bool {
    #[allow(clippy::mutable_key_type)]
    let in_scope_ranges = uop.in_scope_ranges();

    // Check if any of the in-scope ranges are actual RANGE operations
    !in_scope_ranges.iter().any(|key| matches!(key.0.op(), Op::Range { .. }))
}

/// Extract RANGE size as i64. Returns None for symbolic ranges.
pub fn range_size_as_i64(range: &Rc<UOp>) -> Option<i64> {
    if let Op::Range { end, .. } = range.op() {
        get_const_value(end).and_then(|cv| match cv {
            ConstValue::Int(n) => Some(n),
            ConstValue::UInt(n) => Some(n as i64),
            _ => None,
        })
    } else {
        None
    }
}

/// Check if all ranges have identical index expressions (ignoring validity masks).
pub fn all_ranges_same(ranges: &[Rc<UOp>]) -> bool {
    if ranges.is_empty() {
        return true;
    }

    // Extract index expressions (strip validity masks)
    let first_idx = ranges[0].get_idx();

    ranges.iter().skip(1).all(|r| {
        let idx = r.get_idx();
        // Check pointer equality first (fast path)
        Rc::ptr_eq(&first_idx, &idx) || uop_equal(&first_idx, &idx)
    })
}

/// Deep structural equality check for UOps.
pub fn uop_equal(a: &Rc<UOp>, b: &Rc<UOp>) -> bool {
    // Fast path: pointer equality
    if Rc::ptr_eq(a, b) {
        return true;
    }

    // Check operation type (discriminant)
    if std::mem::discriminant(a.op()) != std::mem::discriminant(b.op()) {
        return false;
    }

    // Check dtype
    if a.dtype() != b.dtype() {
        return false;
    }

    // Special case: Const operations need value comparison
    if let (Op::Const(cv_a), Op::Const(cv_b)) = (a.op(), b.op()) {
        return cv_a.0 == cv_b.0;
    }

    // For Range operations, we need to compare all fields including axis_id
    // which is not included in sources(). Use simplified equality:
    // just compare sources since that's what matters for range merging.
    // Different axis_ids with same end value are still compatible ranges.

    // Check sources recursively
    let a_srcs = a.op().sources();
    let b_srcs = b.op().sources();

    if a_srcs.len() != b_srcs.len() {
        return false;
    }

    a_srcs.iter().zip(b_srcs.iter()).all(|(sa, sb)| uop_equal(sa, sb))
}
