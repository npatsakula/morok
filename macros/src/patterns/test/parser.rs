use crate::patterns::parser::*;

fn rule(input: &PatternList, idx: usize) -> &PatternRule {
    match &input.items[idx] {
        PatternItem::Rule(rule) => rule,
        PatternItem::ForBlock(_) => panic!("expected a rule, got a for-block"),
    }
}

fn alu_args(pattern: &Pattern) -> &[Pattern] {
    match pattern {
        Pattern::Alu { args, .. } => args,
        other => panic!("expected an ALU pattern, got {other:?}"),
    }
}

#[test]
fn bare_binding_and_expression_rewrites() {
    let bare: PatternList = syn::parse_quote! { Add(x, @zero) => x };
    let expr: PatternList = syn::parse_quote! { Add(x, @zero) => Some(x.clone()) };
    for list in [&bare, &expr] {
        assert_eq!(list.items.len(), 1);
        assert!(matches!(rule(list, 0).lhs, Pattern::Alu { op: OpRef::Named(_), commutative: false, .. }));
    }
    assert!(matches!(rule(&bare, 0).rhs, RewriteExpr::Var(_)));
    assert!(matches!(rule(&expr, 0).rhs, RewriteExpr::Expr(_)));
    assert!(syn::parse_str::<PatternList>("Add(x, @zero) ~> x").is_err());
}

#[test]
fn binding_and_wildcard_arguments() {
    let input: PatternList = syn::parse_quote! { Mul(_, zero @ @zero) => zero };
    let args = alu_args(&rule(&input, 0).lhs);
    assert!(matches!(args[0], Pattern::Wildcard));
    assert!(
        matches!(&args[1], Pattern::Binding { name, pattern } if name == "zero" && matches!(**pattern, Pattern::Zero))
    );
}

#[test]
fn special_constants() {
    let input: PatternList = syn::parse_quote! { Add(x, @zero) => x, Mul(x, @one) => x };
    assert!(matches!(alu_args(&rule(&input, 0).lhs)[1], Pattern::Zero));
    assert!(matches!(alu_args(&rule(&input, 1).lhs)[1], Pattern::One));
}

#[test]
fn unknown_at_form_is_rejected() {
    assert!(syn::parse_str::<PatternList>("Add(x, @const) => x").is_err());
}

#[test]
fn const_takes_a_rust_pattern() {
    let input: PatternList = syn::parse_quote! {
        Add(x, Const(_)) => x,
        Add(x, Const(ConstValue::Int(0))) => x,
        Add(x, c @ Const(v)) => x,
    };
    assert!(matches!(alu_args(&rule(&input, 0).lhs)[1], Pattern::Const(syn::Pat::Wild(_))));
    assert!(matches!(alu_args(&rule(&input, 1).lhs)[1], Pattern::Const(syn::Pat::TupleStruct(_))));
    assert!(matches!(
        &alu_args(&rule(&input, 2).lhs)[1],
        Pattern::Binding { name, pattern } if name == "c" && matches!(**pattern, Pattern::Const(syn::Pat::Ident(_)))
    ));
}

#[test]
fn value_extractors() {
    let input: PatternList = syn::parse_quote! {
        Neg(c@const(cv)) => c,
        Neg(_c @vconst(vs)) => v,
        Neg(a @anyconst(vals)) => a,
    };
    assert!(
        matches!(&alu_args(&rule(&input, 0).lhs)[0], Pattern::ConstValue { uop, value } if uop == "c" && value == "cv")
    );
    assert!(
        matches!(&alu_args(&rule(&input, 1).lhs)[0], Pattern::VConst { uop, values } if uop == "_c" && values == "vs")
    );
    assert!(
        matches!(&alu_args(&rule(&input, 2).lhs)[0], Pattern::AnyConst { uop, values } if uop == "a" && values == "vals")
    );
}

#[test]
fn commutative_uses_brackets() {
    let input: PatternList = syn::parse_quote! { Add[x, @zero] => x, Sub(x, @zero) => x };
    assert!(matches!(rule(&input, 0).lhs, Pattern::Alu { commutative: true, .. }));
    assert!(matches!(rule(&input, 1).lhs, Pattern::Alu { commutative: false, .. }));
}

#[test]
fn unit_op_versus_variable() {
    let input: PatternList = syn::parse_quote! { noop @ Noop => noop, x if cond(x) => x };
    assert!(matches!(&rule(&input, 0).lhs, Pattern::Binding { pattern, .. } if matches!(**pattern, Pattern::Unit(_))));
    assert!(matches!(rule(&input, 1).lhs, Pattern::Var(_)));
    assert!(rule(&input, 1).guard.is_some());
}

/// Struct fields: a shorthand or snake-case name is a child binding, nested op forms are
/// child patterns, and anything else stays a verbatim Rust pattern.
#[test]
fn struct_fields_split_into_child_and_verbatim() {
    let input: PatternList = syn::parse_quote! {
        Load { index: Index { buffer, .. }, alt: None, gate: Some(g) } => g,
        Range { end, axis_type: AxisType::Upcast, axis_id: id @ AxisId::Renumbered(_), .. } => end,
        Reduce { .. } => x,
    };
    let Pattern::Struct { op, fields } = &rule(&input, 0).lhs else { panic!("expected struct") };
    assert_eq!(op.to_string(), "Load");
    assert!(matches!(&fields[0].pattern, FieldPat::Child(Pattern::Struct { .. })));
    assert!(matches!(&fields[1].pattern, FieldPat::Verbatim(syn::Pat::Ident(_))));
    assert!(matches!(&fields[2].pattern, FieldPat::Child(Pattern::Some(inner)) if matches!(**inner, Pattern::Var(_))));

    let Pattern::Struct { fields, .. } = &rule(&input, 1).lhs else { panic!("expected struct") };
    assert!(matches!(&fields[0].pattern, FieldPat::Child(Pattern::Var(name)) if name == "end"));
    assert!(matches!(&fields[1].pattern, FieldPat::Verbatim(syn::Pat::Path(_))));
    assert!(matches!(&fields[2].pattern, FieldPat::Verbatim(syn::Pat::Ident(ident)) if ident.subpat.is_some()));

    let Pattern::Struct { fields, .. } = &rule(&input, 2).lhs else { panic!("expected struct") };
    assert!(fields.is_empty());
}

#[test]
fn guards_end_at_the_arrow() {
    let input: PatternList = syn::parse_quote! {
        Lt(x, x2) if Rc::ptr_eq(x, x2) && !x.dtype().is_float() => UOp::const_(DType::Bool, false),
        Cast { src: x, dtype } if x.dtype() == dtype => Some(x.clone()),
    };
    assert!(rule(&input, 0).guard.is_some());
    assert!(rule(&input, 1).guard.is_some());
}

#[test]
fn for_blocks() {
    let input: PatternList = syn::parse_quote! {
        Add(x, @zero) => x,
        for op in unary [Neg, Sqrt] {
            op(c @ Const(_)) => c
        },
        for op in binary [*] {
            op(x, @zero) => x,
            op(@zero, x) => x,
        },
    };
    assert_eq!(input.items.len(), 3);
    let PatternItem::ForBlock(unary) = &input.items[1] else { panic!("expected for-block") };
    assert_eq!(unary.var.to_string(), "op");
    assert_eq!(unary.kind.to_string(), "Unary");
    assert_eq!(unary.ops.as_ref().map(Vec::len), Some(2));
    assert!(matches!(&unary.body[0].lhs, Pattern::Alu { op: OpRef::Var(var), .. } if var == "op"));

    let PatternItem::ForBlock(binary) = &input.items[2] else { panic!("expected for-block") };
    assert_eq!(binary.kind.to_string(), "Binary");
    assert!(binary.ops.is_none());
    assert_eq!(binary.body.len(), 2);
}

#[test]
fn for_block_rejects_unknown_kind() {
    assert!(syn::parse_str::<PatternList>("for op in quaternary [A] { op(x) => x }").is_err());
}

#[test]
fn context_declaration() {
    let input: PatternList = syn::parse_quote! {
        @context MyCtx;
        Add(x, y) => { ctx.stats += 1; x.clone() }
    };
    assert!(input.context_type.is_some());
    assert!(matches!(rule(&input, 0).rhs, RewriteExpr::Expr(syn::Expr::Block(_))));
}

#[test]
fn rewrite_expression_forms() {
    let input: PatternList = syn::parse_quote! {
        Add(x, @zero) => x,
        Mul(a, Add(b, c)) => a.try_mul(&b),
        Where(cond, t, f) => {
            match vmin_vmax(cond) {
                (true, true) => Some(t.clone()),
                _ => None,
            }
        },
        FloorDiv(x, x2) => Rc::ptr_eq(x, x2).then(|| one()),
    };
    assert!(matches!(rule(&input, 0).rhs, RewriteExpr::Var(_)));
    assert!(matches!(rule(&input, 1).rhs, RewriteExpr::Expr(syn::Expr::MethodCall(_))));
    assert!(matches!(rule(&input, 2).rhs, RewriteExpr::Expr(syn::Expr::Block(_))));
    assert!(matches!(rule(&input, 3).rhs, RewriteExpr::Expr(syn::Expr::MethodCall(_))));
}
