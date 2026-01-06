use crate::patterns::parser::*;

/// Helper to extract a rule from items at a given index.
fn get_rule(input: &PatternList, idx: usize) -> &PatternRule {
    match &input.items[idx] {
        PatternItem::Rule(rule) => rule,
        PatternItem::ForBlock(_) => panic!("Expected Rule, got ForBlock"),
    }
}

#[test]
fn test_parse_simple_pattern_infallible() {
    let input: PatternList = syn::parse_quote! {
        Add(x, Const(0)) ~> x
    };
    assert_eq!(input.items.len(), 1);
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Infallible);
}

#[test]
fn test_parse_simple_pattern_fallible() {
    let input: PatternList = syn::parse_quote! {
        Add(x, Const(0)) => Some(x)
    };
    assert_eq!(input.items.len(), 1);
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Fallible);
}

#[test]
fn test_parse_binding_pattern() {
    let input: PatternList = syn::parse_quote! {
        Mul(_, zero @ Const(0)) ~> zero
    };
    assert_eq!(input.items.len(), 1);
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Infallible);
}

#[test]
fn test_parse_struct_pattern() {
    let input: PatternList = syn::parse_quote! {
        Bufferize { compute: c, .. } ~> c
    };
    assert_eq!(input.items.len(), 1);
}

#[test]
fn test_parse_with_guard_infallible() {
    // Guard on LHS with infallible arrow
    let input: PatternList = syn::parse_quote! {
        Cast { src: x, dtype } if x.dtype() == dtype ~> x
    };
    assert_eq!(input.items.len(), 1);
    assert!(get_rule(&input, 0).guard.is_some());
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Infallible);
}

#[test]
fn test_parse_with_guard_fallible() {
    // Guard on LHS with fallible arrow
    let input: PatternList = syn::parse_quote! {
        Idiv(x, x2) if Rc::ptr_eq(x, x2) => Some(one_const())
    };
    assert_eq!(input.items.len(), 1);
    assert!(get_rule(&input, 0).guard.is_some());
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Fallible);
}

#[test]
fn test_parse_multiple_patterns() {
    let input: PatternList = syn::parse_quote! {
        Add(x, Const(0)) ~> x,
        Mul(x, Const(1)) ~> x,
        Mul(_, zero @ Const(0)) ~> zero,
    };
    assert_eq!(input.items.len(), 3);
}

#[test]
fn test_parse_special_constants() {
    // Test @zero
    let input: PatternList = syn::parse_quote! {
        Add(x, @zero) ~> x
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { op, args, .. } => {
            assert_eq!(op.to_string(), "Add");
            assert!(matches!(&args[1], Pattern::Const(ConstPattern::Zero)));
        }
        _ => panic!("Expected OpTuple"),
    }

    // Test @one
    let input: PatternList = syn::parse_quote! {
        Mul(x, @one) ~> x
    };
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { args, .. } => {
            assert!(matches!(&args[1], Pattern::Const(ConstPattern::One)));
        }
        _ => panic!("Expected OpTuple"),
    }

    // Test @const
    let input: PatternList = syn::parse_quote! {
        Add(x, @const) ~> x
    };
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { args, .. } => {
            assert!(matches!(&args[1], Pattern::Const(ConstPattern::Any)));
        }
        _ => panic!("Expected OpTuple"),
    }
}

#[test]
fn test_parse_binding_with_special_constant() {
    // Test binding: zero @ @zero
    let input: PatternList = syn::parse_quote! {
        Mul(_, zero @ @zero) ~> zero
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { args, .. } => match &args[1] {
            Pattern::Binding { name, pattern } => {
                assert_eq!(name.to_string(), "zero");
                assert!(matches!(pattern.as_ref(), Pattern::Const(ConstPattern::Zero)));
            }
            _ => panic!("Expected Binding pattern"),
        },
        _ => panic!("Expected OpTuple"),
    }
}

// ========== ConstWithValue Tests ==========

#[test]
fn test_parse_const_with_value() {
    // Test c@const(cv) syntax
    let input: PatternList = syn::parse_quote! {
        Neg(c@const(cv)) ~> c
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { args, .. } => match &args[0] {
            Pattern::ConstWithValue { uop_name, value_name } => {
                assert_eq!(uop_name.to_string(), "c");
                assert_eq!(value_name.to_string(), "cv");
            }
            _ => panic!("Expected ConstWithValue pattern"),
        },
        _ => panic!("Expected OpTuple"),
    }
}

#[test]
fn test_parse_const_with_value_underscore() {
    // Test _c@const(cv) syntax (unused UOp)
    let input: PatternList = syn::parse_quote! {
        Neg(_c@const(v)) ~> v
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { args, .. } => match &args[0] {
            Pattern::ConstWithValue { uop_name, value_name } => {
                assert_eq!(uop_name.to_string(), "_c");
                assert_eq!(value_name.to_string(), "v");
            }
            _ => panic!("Expected ConstWithValue pattern"),
        },
        _ => panic!("Expected OpTuple"),
    }
}

#[test]
fn test_parse_const_with_value_in_struct() {
    // Test c@const(cv) in struct pattern
    let input: PatternList = syn::parse_quote! {
        Cast { src: c@const(cv), dtype } => expr
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).lhs {
        Pattern::OpStruct { fields, .. } => match &fields[0].pattern {
            Pattern::ConstWithValue { uop_name, value_name } => {
                assert_eq!(uop_name.to_string(), "c");
                assert_eq!(value_name.to_string(), "cv");
            }
            _ => panic!("Expected ConstWithValue pattern"),
        },
        _ => panic!("Expected OpStruct"),
    }
}

// ========== For-Block Tests ==========

#[test]
fn test_parse_for_block_unary() {
    let input: PatternList = syn::parse_quote! {
        for op in unary [Neg, Sqrt] {
            op(c @ @const) ~> c
        }
    };
    assert_eq!(input.items.len(), 1);
    match &input.items[0] {
        PatternItem::ForBlock(fb) => {
            assert_eq!(fb.var.to_string(), "op");
            match &fb.iter_kind {
                IterKind::Unary(ops) => {
                    assert_eq!(ops.len(), 2);
                    assert_eq!(ops[0].to_string(), "Neg");
                    assert_eq!(ops[1].to_string(), "Sqrt");
                }
                _ => panic!("Expected Unary"),
            }
            assert_eq!(fb.body.len(), 1);
            assert_eq!(fb.body[0].arrow, ArrowKind::Infallible);
        }
        _ => panic!("Expected ForBlock"),
    }
}

#[test]
fn test_parse_for_block_binary() {
    let input: PatternList = syn::parse_quote! {
        for op in binary [Add, Mul, Sub] {
            op(a @ @const, b @ @const) ~> a
        }
    };
    assert_eq!(input.items.len(), 1);
    match &input.items[0] {
        PatternItem::ForBlock(fb) => match &fb.iter_kind {
            IterKind::Binary(ops) => assert_eq!(ops.len(), 3),
            _ => panic!("Expected Binary"),
        },
        _ => panic!("Expected ForBlock"),
    }
}

#[test]
fn test_parse_for_block_ternary() {
    let input: PatternList = syn::parse_quote! {
        for op in ternary [Where, MulAcc] {
            op(a, b, c) ~> a
        }
    };
    assert_eq!(input.items.len(), 1);
    match &input.items[0] {
        PatternItem::ForBlock(fb) => match &fb.iter_kind {
            IterKind::Ternary(ops) => assert_eq!(ops.len(), 2),
            _ => panic!("Expected Ternary"),
        },
        _ => panic!("Expected ForBlock"),
    }
}

#[test]
fn test_parse_mixed_rules_and_for_blocks() {
    let input: PatternList = syn::parse_quote! {
        Add(x, @zero) ~> x,
        for op in unary [Neg, Sqrt] {
            op(c @ @const) ~> c
        },
        Mul(x, @one) ~> x,
    };
    assert_eq!(input.items.len(), 3);
    assert!(matches!(&input.items[0], PatternItem::Rule(_)));
    assert!(matches!(&input.items[1], PatternItem::ForBlock(_)));
    assert!(matches!(&input.items[2], PatternItem::Rule(_)));
}

#[test]
fn test_parse_op_var_pattern() {
    let input: PatternList = syn::parse_quote! {
        for op in unary [Neg] {
            op(x @ @const) ~> x
        }
    };
    match &input.items[0] {
        PatternItem::ForBlock(fb) => match &fb.body[0].lhs {
            Pattern::OpVar { var_name, args } => {
                assert_eq!(var_name.to_string(), "op");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected OpVar pattern"),
        },
        _ => panic!("Expected ForBlock"),
    }
}

#[test]
fn test_parse_for_block_with_multiple_rules() {
    let input: PatternList = syn::parse_quote! {
        for op in binary [Add, Mul] {
            op(x, @zero) ~> x,
            op(@zero, x) ~> x,
        }
    };
    match &input.items[0] {
        PatternItem::ForBlock(fb) => {
            assert_eq!(fb.body.len(), 2);
        }
        _ => panic!("Expected ForBlock"),
    }
}

// ========== Arrow Kind Tests ==========

#[test]
fn test_parse_infallible_with_complex_guard() {
    let input: PatternList = syn::parse_quote! {
        Lt(x, x2) if Rc::ptr_eq(x, x2) && !x.dtype().is_float() ~> UOp::const_(DType::Bool, false)
    };
    assert_eq!(input.items.len(), 1);
    assert!(get_rule(&input, 0).guard.is_some());
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Infallible);
}

#[test]
fn test_parse_fallible_with_block() {
    let input: PatternList = syn::parse_quote! {
        Where(cond, t, f) => {
            match vmin_vmax(cond) {
                (true, true) => Some(t.clone()),
                (false, false) => Some(f.clone()),
                _ => None,
            }
        }
    };
    assert_eq!(input.items.len(), 1);
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Fallible);
}

#[test]
fn test_parse_mixed_arrow_kinds() {
    let input: PatternList = syn::parse_quote! {
        Add(x, @zero) ~> x,
        Idiv(x, x2) => Rc::ptr_eq(x, x2).then(|| one()),
        Mul(x, @one) ~> x,
    };
    assert_eq!(input.items.len(), 3);
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Infallible);
    assert_eq!(get_rule(&input, 1).arrow, ArrowKind::Fallible);
    assert_eq!(get_rule(&input, 2).arrow, ArrowKind::Infallible);
}

// ========== Rest Pattern Tests ==========

#[test]
fn test_parse_rest_pattern_end() {
    // Test End(computation, ..) syntax
    let input: PatternList = syn::parse_quote! {
        End(computation, ..) ~> computation
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { op, args, rest } => {
            assert_eq!(op.to_string(), "End");
            assert_eq!(args.len(), 1);
            assert!(matches!(&args[0], Pattern::Var(v) if v == "computation"));
            assert!(*rest);
        }
        _ => panic!("Expected OpTuple"),
    }
}

#[test]
fn test_parse_rest_pattern_reduce() {
    // Test Reduce(src, ..) syntax
    let input: PatternList = syn::parse_quote! {
        r @ Reduce(src, ..) if cond(r) ~> src
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).lhs {
        Pattern::Binding { name, pattern } => {
            assert_eq!(name.to_string(), "r");
            match pattern.as_ref() {
                Pattern::OpTuple { op, args, rest } => {
                    assert_eq!(op.to_string(), "Reduce");
                    assert_eq!(args.len(), 1);
                    assert!(*rest);
                }
                _ => panic!("Expected OpTuple inside Binding"),
            }
        }
        _ => panic!("Expected Binding"),
    }
    assert!(get_rule(&input, 0).guard.is_some());
}

#[test]
fn test_parse_rest_pattern_wildcard_arg() {
    // Test Reduce(_, ..) with wildcard first arg
    let input: PatternList = syn::parse_quote! {
        Reduce(_, ..) ~> x
    };
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { args, rest, .. } => {
            assert!(matches!(&args[0], Pattern::Wildcard));
            assert!(*rest);
        }
        _ => panic!("Expected OpTuple"),
    }
}

#[test]
fn test_parse_no_rest_pattern() {
    // Normal pattern without .. should have rest = false
    let input: PatternList = syn::parse_quote! {
        Add(x, y) ~> x
    };
    match &get_rule(&input, 0).lhs {
        Pattern::OpTuple { rest, .. } => {
            assert!(!*rest);
        }
        _ => panic!("Expected OpTuple"),
    }
}

// ========== Closure RHS Tests ==========

#[test]
fn test_parse_closure_simple() {
    // Simple closure RHS
    let input: PatternList = syn::parse_quote! {
        Add(x, @zero) ~> |x| x.clone()
    };
    assert_eq!(input.items.len(), 1);
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Infallible);
    assert!(matches!(get_rule(&input, 0).rhs, RewriteExpr::Closure(_)));
}

#[test]
fn test_parse_closure_multiple_params() {
    // Closure with multiple parameters
    let input: PatternList = syn::parse_quote! {
        Mul(a, Add(b, c)) ~> |a, b, c| a.try_mul(&b)
    };
    assert_eq!(input.items.len(), 1);
    match &get_rule(&input, 0).rhs {
        RewriteExpr::Closure(closure) => {
            assert_eq!(closure.inputs.len(), 3);
        }
        _ => panic!("Expected Closure"),
    }
}

#[test]
fn test_parse_closure_with_type_annotation() {
    // Closure with explicit type annotation
    let input: PatternList = syn::parse_quote! {
        Add(x, @zero) ~> |x: &Arc<UOp>| x.clone()
    };
    assert!(matches!(get_rule(&input, 0).rhs, RewriteExpr::Closure(_)));
}

#[test]
fn test_parse_closure_fallible() {
    // Fallible closure RHS
    let input: PatternList = syn::parse_quote! {
        Mod(x, y) => |x, y| x.try_mod(y).ok()
    };
    assert_eq!(get_rule(&input, 0).arrow, ArrowKind::Fallible);
    assert!(matches!(get_rule(&input, 0).rhs, RewriteExpr::Closure(_)));
}

#[test]
fn test_parse_closure_with_block_body() {
    // Closure with block body
    let input: PatternList = syn::parse_quote! {
        Add(x, y) ~> |x, y, ctx| {
            ctx.stats += 1;
            x.clone()
        }
    };
    assert!(matches!(get_rule(&input, 0).rhs, RewriteExpr::Closure(_)));
}
