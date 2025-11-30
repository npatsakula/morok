use crate::{
    UPat,
    pattern::upat::{ArgPattern, OpFilter},
};
use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, ConstValueHash, Op, UOp};
use std::{mem::discriminant, rc::Rc};

/// Helper to create a const UOp
fn const_uop(val: i64) -> Rc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(val))
}

/// Helper to create a binary op UOp
fn binary_uop(op: BinaryOp, a: Rc<UOp>, b: Rc<UOp>) -> Rc<UOp> {
    let dtype = a.dtype().clone();
    UOp::new(Op::Binary(op, a, b), dtype)
}

#[test]
fn test_var_matches_any() {
    let uop = const_uop(42);
    let pat = UPat::var("x");

    let matches = pat.match_uop(&uop);
    assert_eq!(matches.len(), 1);
    assert!(Rc::ptr_eq(matches[0].get("x").unwrap(), &uop));
}

#[test]
fn test_cvar_matches_const_only() {
    let const_42 = const_uop(42);
    let non_const = binary_uop(BinaryOp::Add, const_uop(1), const_uop(2));

    let pat = UPat::cvar("c");

    // Should match constant
    let matches = pat.match_uop(&const_42);
    assert_eq!(matches.len(), 1);

    // Should not match non-constant
    let matches = pat.match_uop(&non_const);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_op_type_filter() {
    let add = binary_uop(BinaryOp::Add, const_uop(1), const_uop(2));
    let mul = binary_uop(BinaryOp::Mul, const_uop(1), const_uop(2));

    // Match only ADD operations
    let pat = UPat::binary(vec![BinaryOp::Add], vec![UPat::var("a"), UPat::var("b")]);

    // Should match ADD
    let matches = pat.match_uop(&add);
    assert_eq!(matches.len(), 1);

    // Should not match MUL
    let matches = pat.match_uop(&mul);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_source_tuple_matching() {
    let zero = const_uop(0);
    let five = const_uop(5);
    let add = binary_uop(BinaryOp::Add, zero.clone(), five.clone());

    // Match: 0 + x (zero constant plus any value)
    let pat = UPat::binary(
        vec![BinaryOp::Add],
        vec![
            UPat::Match {
                op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Const(ConstValueHash(ConstValue::Int(0)))))]),
                dtype: None,
                src: None,
                arg: Some(ArgPattern::Const(ConstValue::Int(0))),
                name: None,
            },
            UPat::var("x"),
        ],
    );

    let matches = pat.match_uop(&add);
    assert_eq!(matches.len(), 1);
    // Check that "x" was bound to the five constant
    let x = matches[0].get("x").unwrap();
    assert!(matches!(x.op(), Op::Const(ConstValueHash(ConstValue::Int(5)))));
}

#[test]
fn test_named_binding_consistency() {
    // Match: x + x (same variable twice)
    let five = const_uop(5);
    let three = const_uop(3);

    // Using the same Rc twice (same UOp)
    let add_same = binary_uop(BinaryOp::Add, five.clone(), five.clone());

    // Using different UOps
    let add_diff = binary_uop(BinaryOp::Add, five, three);

    let pat = UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::var("x")]);

    // Should match when both are same UOp
    let matches = pat.match_uop(&add_same);
    assert_eq!(matches.len(), 1);

    // Should not match when different UOps
    let matches = pat.match_uop(&add_diff);
    assert_eq!(matches.len(), 0);
}

// ===== Operator Overloading Tests =====

#[test]
fn test_operator_add() {
    // Test: x + y creates correct Add pattern
    let pat = UPat::var("x") + UPat::var("y");

    // Should match ADD operations
    let add = binary_uop(BinaryOp::Add, const_uop(1), const_uop(2));
    let matches = pat.match_uop(&add);
    assert_eq!(matches.len(), 1);

    // Should not match other operations
    let mul = binary_uop(BinaryOp::Mul, const_uop(1), const_uop(2));
    let matches = pat.match_uop(&mul);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_operator_sub() {
    let pat = UPat::var("x") - UPat::cvar("c");
    let sub = binary_uop(BinaryOp::Sub, const_uop(5), const_uop(0));
    let matches = pat.match_uop(&sub);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_mul() {
    let pat = UPat::var("x") * UPat::cvar("c");
    let mul = binary_uop(BinaryOp::Mul, const_uop(5), const_uop(1));
    let matches = pat.match_uop(&mul);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_div() {
    // Test Fdiv (float division)
    let pat = UPat::var("x") / UPat::cvar("c");

    // Create float division UOp
    let five_f = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let one_f = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let fdiv = five_f.try_div(&one_f).unwrap();

    let matches = pat.match_uop(&fdiv);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_idiv() {
    // Test Idiv (integer division via method)
    let pat = UPat::var("x").idiv(UPat::cvar("c"));
    let idiv = binary_uop(BinaryOp::Idiv, const_uop(5), const_uop(1));
    let matches = pat.match_uop(&idiv);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_rem() {
    let pat = UPat::var("x") % UPat::cvar("c");
    let rem = binary_uop(BinaryOp::Mod, const_uop(5), const_uop(2));
    let matches = pat.match_uop(&rem);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_neg() {
    let pat = -UPat::var("x");

    let five = const_uop(5);
    let neg = five.neg();

    let matches = pat.match_uop(&neg);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_bitand() {
    let pat = UPat::var("x") & UPat::cvar("zero");
    let and = binary_uop(BinaryOp::And, const_uop(5), const_uop(0));
    let matches = pat.match_uop(&and);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_bitor() {
    let pat = UPat::var("x") | UPat::cvar("c");
    let or = binary_uop(BinaryOp::Or, const_uop(5), const_uop(0));
    let matches = pat.match_uop(&or);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_bitxor() {
    let pat = UPat::var("x") ^ UPat::cvar("c");
    let xor = binary_uop(BinaryOp::Xor, const_uop(5), const_uop(0));
    let matches = pat.match_uop(&xor);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_shl() {
    let pat = UPat::var("x") << UPat::cvar("n");
    let shl = binary_uop(BinaryOp::Shl, const_uop(1), const_uop(3));
    let matches = pat.match_uop(&shl);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_operator_shr() {
    let pat = UPat::var("x") >> UPat::cvar("n");
    let shr = binary_uop(BinaryOp::Shr, const_uop(8), const_uop(2));
    let matches = pat.match_uop(&shr);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_comparison_methods() {
    // Test .lt()
    let pat = UPat::var("x").lt(UPat::cvar("c"));
    let lt = binary_uop(BinaryOp::Lt, const_uop(5), const_uop(10));
    let matches = pat.match_uop(&lt);
    assert_eq!(matches.len(), 1);

    // Test .eq()
    let pat = UPat::var("x").eq(UPat::cvar("c"));
    let eq = binary_uop(BinaryOp::Eq, const_uop(5), const_uop(5));
    let matches = pat.match_uop(&eq);
    assert_eq!(matches.len(), 1);

    // Test .ne()
    let pat = UPat::var("x").ne(UPat::cvar("c"));
    let ne = binary_uop(BinaryOp::Ne, const_uop(5), const_uop(3));
    let matches = pat.match_uop(&ne);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_math_methods() {
    // Test .max()
    let pat = UPat::var("x").max(UPat::var("y"));
    let max = binary_uop(BinaryOp::Max, const_uop(5), const_uop(3));
    let matches = pat.match_uop(&max);
    assert_eq!(matches.len(), 1);

    // Test .pow()
    let pat = UPat::var("x").pow(UPat::cvar("n"));
    let pow = binary_uop(BinaryOp::Pow, const_uop(2), const_uop(3));
    let matches = pat.match_uop(&pow);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_chained_operators() {
    // Test: (a + b) * c
    let pat = (UPat::var("a") + UPat::var("b")) * UPat::var("c");

    // Create UOp: (1 + 2) * 3
    let one = const_uop(1);
    let two = const_uop(2);
    let three = const_uop(3);
    let add = binary_uop(BinaryOp::Add, one, two);
    let mul = binary_uop(BinaryOp::Mul, add.clone(), three);

    let matches = pat.match_uop(&mul);
    assert_eq!(matches.len(), 1);

    // Verify bindings - check that we matched the correct UOps
    let bindings = &matches[0];
    let a = bindings.get("a").unwrap();
    let b = bindings.get("b").unwrap();
    let c = bindings.get("c").unwrap();

    // a and b should be the children of the add operation
    if let Op::Const(ConstValueHash(ConstValue::Int(v))) = a.op() {
        assert_eq!(*v, 1);
    } else {
        panic!("Expected a to be const 1");
    }

    if let Op::Const(ConstValueHash(ConstValue::Int(v))) = b.op() {
        assert_eq!(*v, 2);
    } else {
        panic!("Expected b to be const 2");
    }

    // c should be const 3
    assert!(Rc::ptr_eq(c, &const_uop(3)));
}

// ===== Helper Method Tests =====

#[test]
fn test_zero_const() {
    let pat = UPat::zero_const("z");

    // Should match zero
    let zero = const_uop(0);
    let matches = pat.match_uop(&zero);
    assert_eq!(matches.len(), 1);
    assert!(Rc::ptr_eq(matches[0].get("z").unwrap(), &zero));

    // Should not match non-zero
    let five = const_uop(5);
    let matches = pat.match_uop(&five);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_positive_const() {
    let pat = UPat::positive_const("p");

    // Should match positive
    let five = const_uop(5);
    let matches = pat.match_uop(&five);
    assert_eq!(matches.len(), 1);

    // Should not match zero
    let zero = const_uop(0);
    let matches = pat.match_uop(&zero);
    assert_eq!(matches.len(), 0);

    // Should not match negative
    let neg = const_uop(-5);
    let matches = pat.match_uop(&neg);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_nonzero_const() {
    let pat = UPat::nonzero_const("nz");

    // Should match positive
    let five = const_uop(5);
    let matches = pat.match_uop(&five);
    assert_eq!(matches.len(), 1);

    // Should match negative
    let neg = const_uop(-5);
    let matches = pat.match_uop(&neg);
    assert_eq!(matches.len(), 1);

    // Should not match zero
    let zero = const_uop(0);
    let matches = pat.match_uop(&zero);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_int_const() {
    let pat = UPat::int(42);

    // Should match exactly 42
    let forty_two = const_uop(42);
    let matches = pat.match_uop(&forty_two);
    assert_eq!(matches.len(), 1);

    // Should not match different value
    let five = const_uop(5);
    let matches = pat.match_uop(&five);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_float_const() {
    let pat = UPat::float(1.5);

    // Create a float constant UOp
    let one_point_five = UOp::const_(DType::Float32, ConstValue::Float(1.5));
    let matches = pat.match_uop(&one_point_five);
    assert_eq!(matches.len(), 1);

    // Should not match different value
    let two_point_zero = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let matches = pat.match_uop(&two_point_zero);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_const_val() {
    let pat = UPat::const_val(ConstValue::Int(99));

    let ninety_nine = const_uop(99);
    let matches = pat.match_uop(&ninety_nine);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_detach_helper() {
    let pat = UPat::detach(UPat::var("x"));

    let five = const_uop(5);
    let detach = UOp::detach(five.clone());

    let matches = pat.match_uop(&detach);
    assert_eq!(matches.len(), 1);
    assert!(Rc::ptr_eq(matches[0].get("x").unwrap(), &five));

    // Should not match non-detach operations
    let add = binary_uop(BinaryOp::Add, const_uop(1), const_uop(2));
    let matches = pat.match_uop(&add);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_contiguous_backward_helper() {
    let pat = UPat::contiguous_backward(UPat::var("x"));

    let five = const_uop(5);
    let contiguous = UOp::contiguous_backward(five.clone());

    let matches = pat.match_uop(&contiguous);
    assert_eq!(matches.len(), 1);
    assert!(Rc::ptr_eq(matches[0].get("x").unwrap(), &five));
}
