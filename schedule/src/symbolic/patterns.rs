//! Symbolic simplification pattern definitions.
//!
//! Defines the core symbolic simplification patterns for algebraic optimization.
//!
//! This module contains:
//! - Identity element folding (x + 0 → x, x * 1 → x)
//! - Zero propagation (x * 0 → 0, x & 0 → 0)
//! - Constant folding (coming soon)
//!
//! These patterns are separated from rangeify patterns because they apply
//! universally to any UOp graph, not just during schedule transformation.

use morok_ir::BinaryOp;

use crate::pattern::UPat;
use crate::pattern::matcher::PatternMatcher;
use crate::rangeify::helpers::{get_const_value, is_identity_value};
use std::rc::Rc;

/// Pattern matcher for simple symbolic simplifications.
///
/// Contains algebraic identities and zero propagation rules:
/// - x + 0 → x, 0 + x → x
/// - x - 0 → x
/// - x * 1 → x, 1 * x → x
/// - x / 1 → x (both Idiv and Fdiv)
/// - x | 0 → x, 0 | x → x
/// - x ^ 0 → x, 0 ^ x → x
/// - x * 0 → 0, 0 * x → 0
/// - x & 0 → 0, 0 & x → 0
pub fn symbolic_simple() -> PatternMatcher {
    let mut patterns = vec![];

    // ========== Identity folding ==========

    // x + 0 → x
    pattern!(patterns,
        UPat::var("x") + UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Add, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x - 0 → x
    pattern!(patterns,
        UPat::var("x") - UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Sub, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x * 1 → x
    pattern!(patterns,
        UPat::var("x") * UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Mul, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x / 1 → x (int division)
    pattern!(patterns,
        UPat::var("x").idiv(UPat::cvar("c")) => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Idiv, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x / 1 → x (float division)
    pattern!(patterns,
        UPat::var("x") / UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Fdiv, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x | 0 → x
    pattern!(patterns,
        UPat::var("x") | UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Or, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x ^ 0 → x
    pattern!(patterns,
        UPat::var("x") ^ UPat::cvar("c") => |x, c| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Xor, true) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // 0 + x → x (left identity for Add)
    pattern!(patterns,
        UPat::cvar("c") + UPat::var("x") => |c, x| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Add, false) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // 1 * x → x (left identity for Mul)
    pattern!(patterns,
        UPat::cvar("c") * UPat::var("x") => |c, x| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Mul, false) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // 0 | x → x (left identity for Or)
    pattern!(patterns,
        UPat::cvar("c") | UPat::var("x") => |c, x| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Or, false) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // 0 ^ x → x (left identity for Xor)
    pattern!(patterns,
        UPat::cvar("c") ^ UPat::var("x") => |c, x| {
            let const_val = get_const_value(c)?;
            if is_identity_value(&const_val, &BinaryOp::Xor, false) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x * 0 → 0 (zero propagation for Mul)
    pattern!(patterns,
        UPat::var("x") * UPat::zero_const("zero") => |x, zero| {
            let _unused = x;
            Some(Rc::clone(zero))
        }
    );

    // 0 * x → 0 (zero propagation for Mul, left side)
    pattern!(patterns,
        UPat::zero_const("zero") * UPat::var("x") => |zero, x| {
            let _unused = x;
            Some(Rc::clone(zero))
        }
    );

    // x & 0 → 0 (zero propagation for And)
    pattern!(patterns,
        UPat::var("x") & UPat::zero_const("zero") => |x, zero| {
            let _unused = x;
            Some(Rc::clone(zero))
        }
    );

    // 0 & x → 0 (zero propagation for And, left side)
    pattern!(patterns,
        UPat::zero_const("zero") & UPat::var("x") => |zero, x| {
            let _unused = x;
            Some(Rc::clone(zero))
        }
    );

    // ====== Self-folding operations ======

    // x // x → 1
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Idiv], vec![UPat::var("x"), UPat::var("x2")]) => |x: &Rc<morok_ir::UOp>, x2: &Rc<morok_ir::UOp>| {
            use morok_ir::{ConstValue, UOp};
            // Check if both operands are the same (pointer equality)
            if Rc::ptr_eq(x, x2) {
                // Return 1 in the same dtype as x
                Some(UOp::const_(x.dtype(), ConstValue::Int(1)))
            } else {
                None
            }
        }
    );

    // x // -1 → -x
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Idiv], vec![UPat::var("x"), UPat::cvar("divisor")]) => |x: &Rc<morok_ir::UOp>, divisor: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, UnaryOp, UOp, ConstValue};
            // Check if divisor is -1
            if let Op::Const(cv) = divisor.op() {
                if cv.0 == ConstValue::Int(-1) {
                    return Some(UOp::new(Op::Unary(UnaryOp::Neg, Rc::clone(x)), x.dtype()));
                }
            }
            None
        }
    );

    // (x % y) % y → x % y
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Mod], vec![
            UPat::binary(vec![BinaryOp::Mod], vec![UPat::var("x"), UPat::var("y")]),
            UPat::var("y2")
        ]) => |x: &Rc<morok_ir::UOp>, y: &Rc<morok_ir::UOp>, y2: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, UOp};
            // Check if outer y is same as inner y (pointer equality)
            if Rc::ptr_eq(y, y2) {
                // Return the inner (x % y) modulo
                Some(UOp::new(Op::Binary(BinaryOp::Mod, Rc::clone(x), Rc::clone(y)), x.dtype()))
            } else {
                None
            }
        }
    );

    // x & x → x
    pattern!(patterns,
        UPat::var("x") & UPat::var("x2") => |x: &Rc<morok_ir::UOp>, x2: &Rc<morok_ir::UOp>| {
            if Rc::ptr_eq(x, x2) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // x | x → x
    pattern!(patterns,
        UPat::var("x") | UPat::var("x2") => |x: &Rc<morok_ir::UOp>, x2: &Rc<morok_ir::UOp>| {
            if Rc::ptr_eq(x, x2) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // ====== ZERO FOLDING  ======

    // x < x → False
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Lt], vec![UPat::var("x"), UPat::var("x2")]) => |x: &Rc<morok_ir::UOp>, x2: &Rc<morok_ir::UOp>| {
            use morok_ir::{ConstValue, UOp};
            if Rc::ptr_eq(x, x2) {
                Some(UOp::const_(x.dtype(), ConstValue::Bool(false)))
            } else {
                None
            }
        }
    );

    // x % x → 0
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Mod], vec![UPat::var("x"), UPat::var("x2")]) => |x: &Rc<morok_ir::UOp>, x2: &Rc<morok_ir::UOp>| {
            use morok_ir::{ConstValue, UOp};
            if Rc::ptr_eq(x, x2) {
                Some(UOp::const_(x.dtype(), ConstValue::Int(0)))
            } else {
                None
            }
        }
    );

    // x != x → False (for integers)
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Ne], vec![UPat::var("x"), UPat::var("x2")]) => |x: &Rc<morok_ir::UOp>, x2: &Rc<morok_ir::UOp>| {
            use morok_ir::{ConstValue, UOp};
            if Rc::ptr_eq(x, x2) {
                // Only for integer types (floats can have NaN != NaN)
                if x.dtype().is_int() {
                    Some(UOp::const_(x.dtype(), ConstValue::Bool(false)))
                } else {
                    None
                }
            } else {
                None
            }
        }
    );

    // ====== DIVISION PATTERNS  ======

    // x / x → 1.0 (float division)
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Fdiv], vec![UPat::var("x"), UPat::var("x2")]) => |x: &Rc<morok_ir::UOp>, x2: &Rc<morok_ir::UOp>| {
            use morok_ir::{ConstValue, UOp};
            if Rc::ptr_eq(x, x2) {
                Some(UOp::const_(x.dtype(), ConstValue::Float(1.0)))
            } else {
                None
            }
        }
    );

    // (x * y) / y → x
    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Fdiv, BinaryOp::Idiv], vec![
            UPat::binary(vec![BinaryOp::Mul], vec![UPat::var("x"), UPat::var("y")]),
            UPat::var("y2")
        ]) => |x: &Rc<morok_ir::UOp>, y: &Rc<morok_ir::UOp>, y2: &Rc<morok_ir::UOp>| {
            // Check if divisor is same as multiplier
            if Rc::ptr_eq(y, y2) {
                Some(Rc::clone(x))
            } else {
                None
            }
        }
    );

    // ====== CAST OPTIMIZATION  ======

    // cast(const) → const
    // Constant folding for cast operations
    pattern!(patterns,
        UPat::cast_named(UPat::cvar("c"), "cast") => |c: &Rc<morok_ir::UOp>, cast: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Get the target dtype from the cast
            if let Op::Cast { dtype: target_dtype, .. } = cast.op() {
                // Get the constant value
                if let Op::Const(cv) = c.op() {
                    // Convert the constant to the target type
                    let new_value = match &cv.0 {
                        ConstValue::Int(i) => {
                            if target_dtype.is_float() {
                                Some(ConstValue::Float(*i as f64))
                            } else if target_dtype.is_int() {
                                Some(ConstValue::Int(*i))
                            } else {
                                None
                            }
                        },
                        ConstValue::Float(f) => {
                            if target_dtype.is_int() {
                                Some(ConstValue::Int(*f as i64))
                            } else if target_dtype.is_float() {
                                Some(ConstValue::Float(*f))
                            } else {
                                None
                            }
                        },
                        ConstValue::Bool(b) => {
                            if target_dtype.is_int() {
                                Some(ConstValue::Int(if *b { 1 } else { 0 }))
                            } else {
                                None
                            }
                        },
                        _ => None,
                    };

                    if let Some(val) = new_value {
                        return Some(UOp::const_(target_dtype.clone(), val));
                    }
                }
            }
            None
        }
    );

    // x.cast(dtype) → x if same dtype
    pattern!(patterns,
        UPat::cast_named(UPat::var("x"), "cast") => |x: &Rc<morok_ir::UOp>, cast: &Rc<morok_ir::UOp>| {
            use morok_ir::Op;

            // Check if cast is to the same dtype
            if let Op::Cast { dtype: target_dtype, .. } = cast.op() {
                if x.dtype() == *target_dtype {
                    return Some(Rc::clone(x));
                }
            }
            None
        }
    );

    // x.cast(a).cast(b) → x.cast(b)
    // Collapse double cast
    pattern!(patterns,
        UPat::cast_named(UPat::cast(UPat::var("x")), "outer_cast") => |x: &Rc<morok_ir::UOp>, outer_cast: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, UOp};

            // Get the final target dtype from outer cast
            if let Op::Cast { dtype: final_dtype, .. } = outer_cast.op() {
                // Create a single cast from x to final_dtype
                return Some(UOp::new(
                    Op::Cast {
                        src: Rc::clone(x),
                        dtype: final_dtype.clone(),
                    },
                    final_dtype.clone(),
                ));
            }
            None
        }
    );

    // ========== Combine terms ==========
    // Critical for RESHAPE - combines like terms in addition/multiplication

    // x + x → 2*x
    // Combine identical terms in addition
    pattern!(patterns,
        UPat::var("x") + UPat::var("y") => |x: &Rc<morok_ir::UOp>, y: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Check if x and y are the same UOp (pointer equality for hash-consed nodes)
            if Rc::ptr_eq(x, y) {
                // x + x → 2*x
                let two = UOp::const_(x.dtype(), ConstValue::Int(2));
                return Some(UOp::new(
                    Op::Binary(BinaryOp::Mul, two, Rc::clone(x)),
                    x.dtype(),
                ));
            }
            None
        }
    );

    // (c1 * x) + (c2 * x) → (c1 + c2) * x
    // Combine terms with same variable but different coefficients
    pattern!(patterns,
        (UPat::cvar("c1") * UPat::var("x")) + (UPat::cvar("c2") * UPat::var("y"))
            => |c1: &Rc<morok_ir::UOp>, x: &Rc<morok_ir::UOp>,
                c2: &Rc<morok_ir::UOp>, y: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Check if x and y are the same variable
            if Rc::ptr_eq(x, y) {
                // Both c1 and c2 must be constants
                if let (Op::Const(cv1), Op::Const(cv2)) = (c1.op(), c2.op()) {
                    if let (ConstValue::Int(i1), ConstValue::Int(i2)) = (&cv1.0, &cv2.0) {
                        // (c1 * x) + (c2 * x) → (c1 + c2) * x
                        let sum = UOp::const_(c1.dtype(), ConstValue::Int(i1 + i2));
                        return Some(UOp::new(
                            Op::Binary(BinaryOp::Mul, sum, Rc::clone(x)),
                            x.dtype(),
                        ));
                    }
                }
            }
            None
        }
    );

    // (x * c1) + (x * c2) → x * (c1 + c2)
    // Same as above but with reversed multiplication order
    pattern!(patterns,
        (UPat::var("x") * UPat::cvar("c1")) + (UPat::var("y") * UPat::cvar("c2"))
            => |x: &Rc<morok_ir::UOp>, c1: &Rc<morok_ir::UOp>,
                y: &Rc<morok_ir::UOp>, c2: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Check if x and y are the same variable
            if Rc::ptr_eq(x, y) {
                // Both c1 and c2 must be constants
                if let (Op::Const(cv1), Op::Const(cv2)) = (c1.op(), c2.op()) {
                    if let (ConstValue::Int(i1), ConstValue::Int(i2)) = (&cv1.0, &cv2.0) {
                        // (x * c1) + (x * c2) → x * (c1 + c2)
                        let sum = UOp::const_(c1.dtype(), ConstValue::Int(i1 + i2));
                        return Some(UOp::new(
                            Op::Binary(BinaryOp::Mul, Rc::clone(x), sum),
                            x.dtype(),
                        ));
                    }
                }
            }
            None
        }
    );

    // ========== Two-stage ALU folding ==========
    // Fold constants in associative operation chains

    // (x + c1) + c2 → x + (c1 + c2)
    // Fold constants in addition chains
    pattern!(patterns,
        (UPat::var("x") + UPat::cvar("c1")) + UPat::cvar("c2") => |x: &Rc<morok_ir::UOp>, c1: &Rc<morok_ir::UOp>, c2: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Get constant values
            if let (Op::Const(cv1), Op::Const(cv2)) = (c1.op(), c2.op()) {
                if let (ConstValue::Int(i1), ConstValue::Int(i2)) = (&cv1.0, &cv2.0) {
                    // (x + c1) + c2 → x + (c1 + c2)
                    let sum = UOp::const_(c1.dtype(), ConstValue::Int(i1 + i2));
                    return Some(UOp::new(
                        Op::Binary(BinaryOp::Add, Rc::clone(x), sum),
                        x.dtype(),
                    ));
                }
            }
            None
        }
    );

    // (x * c1) * c2 → x * (c1 * c2)
    // Fold constants in multiplication chains
    pattern!(patterns,
        (UPat::var("x") * UPat::cvar("c1")) * UPat::cvar("c2") => |x: &Rc<morok_ir::UOp>, c1: &Rc<morok_ir::UOp>, c2: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Get constant values
            if let (Op::Const(cv1), Op::Const(cv2)) = (c1.op(), c2.op()) {
                if let (ConstValue::Int(i1), ConstValue::Int(i2)) = (&cv1.0, &cv2.0) {
                    // (x * c1) * c2 → x * (c1 * c2)
                    let product = UOp::const_(c1.dtype(), ConstValue::Int(i1 * i2));
                    return Some(UOp::new(
                        Op::Binary(BinaryOp::Mul, Rc::clone(x), product),
                        x.dtype(),
                    ));
                }
            }
            None
        }
    );

    // (x - c1) + c2 → x + (c2 - c1) or x - (c1 - c2)
    // Fold constants in mixed add/sub chains
    pattern!(patterns,
        (UPat::var("x") - UPat::cvar("c1")) + UPat::cvar("c2") => |x: &Rc<morok_ir::UOp>, c1: &Rc<morok_ir::UOp>, c2: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Get constant values
            if let (Op::Const(cv1), Op::Const(cv2)) = (c1.op(), c2.op()) {
                if let (ConstValue::Int(i1), ConstValue::Int(i2)) = (&cv1.0, &cv2.0) {
                    let diff = i2 - i1;
                    if diff >= 0 {
                        // (x - c1) + c2 → x + (c2 - c1)
                        let const_val = UOp::const_(c1.dtype(), ConstValue::Int(diff));
                        return Some(UOp::new(
                            Op::Binary(BinaryOp::Add, Rc::clone(x), const_val),
                            x.dtype(),
                        ));
                    } else {
                        // (x - c1) + c2 → x - (c1 - c2)
                        let const_val = UOp::const_(c1.dtype(), ConstValue::Int(-diff));
                        return Some(UOp::new(
                            Op::Binary(BinaryOp::Sub, Rc::clone(x), const_val),
                            x.dtype(),
                        ));
                    }
                }
            }
            None
        }
    );

    // (x + c1) - c2 → x + (c1 - c2) or x - (c2 - c1)
    // Fold constants in mixed add/sub chains (reversed)
    pattern!(patterns,
        (UPat::var("x") + UPat::cvar("c1")) - UPat::cvar("c2") => |x: &Rc<morok_ir::UOp>, c1: &Rc<morok_ir::UOp>, c2: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Get constant values
            if let (Op::Const(cv1), Op::Const(cv2)) = (c1.op(), c2.op()) {
                if let (ConstValue::Int(i1), ConstValue::Int(i2)) = (&cv1.0, &cv2.0) {
                    let diff = i1 - i2;
                    if diff >= 0 {
                        // (x + c1) - c2 → x + (c1 - c2)
                        let const_val = UOp::const_(c1.dtype(), ConstValue::Int(diff));
                        return Some(UOp::new(
                            Op::Binary(BinaryOp::Add, Rc::clone(x), const_val),
                            x.dtype(),
                        ));
                    } else {
                        // (x + c1) - c2 → x - (c2 - c1)
                        let const_val = UOp::const_(c1.dtype(), ConstValue::Int(-diff));
                        return Some(UOp::new(
                            Op::Binary(BinaryOp::Sub, Rc::clone(x), const_val),
                            x.dtype(),
                        ));
                    }
                }
            }
            None
        }
    );

    // ========== Basic division patterns ==========
    // Division and modulo simplifications using helper methods

    // (a * b) // b → a (when b divides evenly)
    // Simplify division when divisor cancels with multiplication
    pattern!(patterns,
        (UPat::var("a") * UPat::var("b")).idiv(UPat::var("c")) => |a: &Rc<morok_ir::UOp>, b: &Rc<morok_ir::UOp>, c: &Rc<morok_ir::UOp>| {
            // Check if b and c are the same (b cancels out)
            if Rc::ptr_eq(b, c) {
                return Some(Rc::clone(a));
            }
            None
        }
    );

    // (a // b) // c → a // (b * c)
    // Combine division chains
    pattern!(patterns,
        UPat::var("a").idiv(UPat::cvar("b")).idiv(UPat::cvar("c")) => |a: &Rc<morok_ir::UOp>, b: &Rc<morok_ir::UOp>, c: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Get constant values
            if let (Op::Const(cv_b), Op::Const(cv_c)) = (b.op(), c.op()) {
                if let (ConstValue::Int(ib), ConstValue::Int(ic)) = (&cv_b.0, &cv_c.0) {
                    if *ib != 0 && *ic != 0 {
                        // (a // b) // c → a // (b * c)
                        let product = UOp::const_(b.dtype(), ConstValue::Int(ib * ic));
                        return Some(UOp::new(
                            Op::Binary(BinaryOp::Idiv, Rc::clone(a), product),
                            a.dtype(),
                        ));
                    }
                }
            }
            None
        }
    );

    // (a % b) % b → a % b
    // Already handled earlier, but this is explicit for clarity

    // (a * c) // c → a (using divides helper)
    // Use divides() to check if division is exact
    pattern!(patterns,
        UPat::var("expr").idiv(UPat::cvar("divisor")) => |expr: &Rc<morok_ir::UOp>, divisor: &Rc<morok_ir::UOp>| {
            // Use the divides() helper method to check exact division
            expr.divides(divisor)
        }
    );

    // (a + b) % c → (a % c + b % c) % c (for certain cases)
    // Distribute modulo over addition when const_factor allows
    pattern!(patterns,
        (UPat::var("a") + UPat::var("b")) % UPat::cvar("c") => |a: &Rc<morok_ir::UOp>, b: &Rc<morok_ir::UOp>, c: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, ConstValue, UOp};

            // Only apply if c is a small constant to avoid explosion
            if let Op::Const(cv) = c.op() {
                if let ConstValue::Int(modulus) = &cv.0 {
                    if *modulus > 0 && *modulus <= 256 {
                        // Check if either operand is already a multiple of the modulus
                        let a_factor = a.const_factor();
                        let b_factor = b.const_factor();

                        if a_factor % modulus == 0 {
                            // a is divisible by c, so (a + b) % c = b % c
                            return Some(UOp::new(
                                Op::Binary(BinaryOp::Mod, Rc::clone(b), Rc::clone(c)),
                                a.dtype(),
                            ));
                        }

                        if b_factor % modulus == 0 {
                            // b is divisible by c, so (a + b) % c = a % c
                            return Some(UOp::new(
                                Op::Binary(BinaryOp::Mod, Rc::clone(a), Rc::clone(c)),
                                a.dtype(),
                            ));
                        }
                    }
                }
            }
            None
        }
    );

    // ========== Distribute operations ==========
    // Distribution patterns for multiplication and division

    // (a + b) // c → (a // c) + (b // c) when both divide evenly
    // Distribute division over addition (only when exact)
    pattern!(patterns,
        (UPat::var("a") + UPat::var("b")).idiv(UPat::cvar("c")) => |a: &Rc<morok_ir::UOp>, b: &Rc<morok_ir::UOp>, c: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, UOp};

            // Check if both a and b are divisible by c
            if let Some(a_div) = a.divides(c) {
                if let Some(b_div) = b.divides(c) {
                    // (a + b) // c → (a // c) + (b // c)
                    return Some(UOp::new(
                        Op::Binary(BinaryOp::Add, a_div, b_div),
                        a.dtype(),
                    ));
                }
            }
            None
        }
    );

    // (a - b) // c → (a // c) - (b // c) when both divide evenly
    // Distribute division over subtraction (only when exact)
    pattern!(patterns,
        (UPat::var("a") - UPat::var("b")).idiv(UPat::cvar("c")) => |a: &Rc<morok_ir::UOp>, b: &Rc<morok_ir::UOp>, c: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, UOp};

            // Check if both a and b are divisible by c
            if let Some(a_div) = a.divides(c) {
                if let Some(b_div) = b.divides(c) {
                    // (a - b) // c → (a // c) - (b // c)
                    return Some(UOp::new(
                        Op::Binary(BinaryOp::Sub, a_div, b_div),
                        a.dtype(),
                    ));
                }
            }
            None
        }
    );

    // c * (a + b) → (c * a) + (c * b)
    // Distribute multiplication over addition
    // Note: Distributes unconditionally without size checks
    pattern!(patterns,
        UPat::cvar("c") * (UPat::var("a") + UPat::var("b")) => |c: &Rc<morok_ir::UOp>, a: &Rc<morok_ir::UOp>, b: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, UOp};

            let ca = UOp::new(
                Op::Binary(BinaryOp::Mul, Rc::clone(c), Rc::clone(a)),
                a.dtype(),
            );
            let cb = UOp::new(
                Op::Binary(BinaryOp::Mul, Rc::clone(c), Rc::clone(b)),
                b.dtype(),
            );
            Some(UOp::new(
                Op::Binary(BinaryOp::Add, ca, cb),
                a.dtype(),
            ))
        }
    );

    // (a + b) * c → (a * c) + (b * c)
    // Distribute multiplication over addition (reversed operand order)
    // Note: Distributes unconditionally without size checks
    pattern!(patterns,
        (UPat::var("a") + UPat::var("b")) * UPat::cvar("c") => |a: &Rc<morok_ir::UOp>, b: &Rc<morok_ir::UOp>, c: &Rc<morok_ir::UOp>| {
            use morok_ir::{Op, UOp};

            let ac = UOp::new(
                Op::Binary(BinaryOp::Mul, Rc::clone(a), Rc::clone(c)),
                a.dtype(),
            );
            let bc = UOp::new(
                Op::Binary(BinaryOp::Mul, Rc::clone(b), Rc::clone(c)),
                b.dtype(),
            );
            Some(UOp::new(
                Op::Binary(BinaryOp::Add, ac, bc),
                a.dtype(),
            ))
        }
    );

    PatternMatcher::new(patterns)
}

pub fn symbolic() -> PatternMatcher {
    // TODO: Add more complex symbolic patterns
    PatternMatcher::new(vec![])
}
