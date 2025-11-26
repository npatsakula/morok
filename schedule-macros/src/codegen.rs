//! Code generation for the pattern DSL.
//!
//! Converts the parsed pattern AST into Rust code that constructs
//! a `PatternMatcher`.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Error, Ident, Result};

use crate::parser::{
    ArrowKind, ConstPattern, FieldPattern, ForBlock, IterKind, Pattern, PatternItem, PatternList, PatternRule,
    RewriteExpr,
};

/// Internal binding names used by the code generator.
mod binding_names {
    /// Name for the root UOp in struct patterns (used for field extraction)
    pub const STRUCT_ROOT: &str = "__struct_root";
    /// Name for any constant binding
    pub const CONST: &str = "__const";
    /// Name for zero constant binding
    pub const ZERO: &str = "__zero";
    /// Name for one constant binding
    pub const ONE: &str = "__one";
}

/// Check if struct pattern has extractable fields.
///
/// Extractable fields are shorthand fields after the first one where
/// the pattern is a variable with the same name as the field.
/// Example: `Cast { src: x, dtype }` - `dtype` is extractable.
fn has_extractable_fields(fields: &[FieldPattern]) -> bool {
    fields.iter().skip(1).any(|f| matches!(&f.pattern, Pattern::Var(var) if *var == f.name))
}

/// Output from pattern generation containing both code and collected names.
///
/// This struct enables single-pass pattern generation where name collection
/// happens alongside code generation, eliminating the need for a separate
/// `collect_var_names` traversal.
///
/// The split between `self_name` and `child_names` models the `.named()` semantic:
/// - `.named("x")` **replaces** the pattern's own name with "x"
/// - `.named("x")` **preserves** names from child patterns
struct PatternOutput {
    /// Generated TokenStream2 for UPat construction
    code: TokenStream2,
    /// The top-level name of this pattern (replaceable by `.named()`)
    self_name: Option<Ident>,
    /// Names from nested/child patterns (not replaceable by `.named()`)
    child_names: Vec<Ident>,
}

impl PatternOutput {
    /// Get all names in collection order (self first, then children).
    /// This matches the order that `UPat::collect_var_names_internal` uses.
    fn all_names(&self) -> Vec<Ident> {
        let mut result = Vec::new();
        if let Some(ref name) = self.self_name {
            result.push(name.clone());
        }
        result.extend(self.child_names.iter().cloned());
        result
    }

    /// Create output with no names (e.g., wildcard patterns).
    fn no_names(code: TokenStream2) -> Self {
        Self { code, self_name: None, child_names: vec![] }
    }

    /// Create output with only a self name (e.g., variable patterns).
    fn self_only(code: TokenStream2, name: Ident) -> Self {
        Self { code, self_name: Some(name), child_names: vec![] }
    }

    /// Create output with only child names (e.g., operation patterns).
    fn children_only(code: TokenStream2, children: Vec<Ident>) -> Self {
        Self { code, self_name: None, child_names: children }
    }
}

/// Context for iteration variable substitution.
#[derive(Clone)]
struct IterContext {
    var_name: Ident,
    op_ident: Ident,
    op_kind: OpKind,
}

/// The kind of operation being iterated over.
#[derive(Clone, Copy)]
enum OpKind {
    Unary,
    Binary,
    Ternary,
}

/// Generate a `PatternMatcher` from the parsed pattern list.
pub fn generate_pattern_matcher(patterns: &PatternList) -> Result<TokenStream2> {
    let mut pattern_exprs = Vec::new();

    for item in &patterns.items {
        match item {
            PatternItem::Rule(rule) => {
                pattern_exprs.push(generate_rule(rule, None)?);
            }
            PatternItem::ForBlock(for_block) => {
                let expanded = expand_for_block(for_block)?;
                pattern_exprs.extend(expanded);
            }
        }
    }

    Ok(quote! {
        {
            let mut __patterns: Vec<(
                morok_schedule::pattern::UPat,
                morok_schedule::pattern::matcher::RewriteFn
            )> = Vec::new();
            #(#pattern_exprs)*
            morok_schedule::pattern::PatternMatcher::new(__patterns)
        }
    })
}

/// Expand a for-block into multiple pattern rules.
fn expand_for_block(for_block: &ForBlock) -> Result<Vec<TokenStream2>> {
    let var_name = &for_block.var;
    let mut results = Vec::new();

    let (ops, kind) = match &for_block.iter_kind {
        IterKind::Unary(ops) => (ops, OpKind::Unary),
        IterKind::Binary(ops) => (ops, OpKind::Binary),
        IterKind::Ternary(ops) => (ops, OpKind::Ternary),
    };

    for op_ident in ops {
        for rule in &for_block.body {
            let ctx = IterContext { var_name: var_name.clone(), op_ident: op_ident.clone(), op_kind: kind };
            results.push(generate_rule(rule, Some(&ctx))?);
        }
    }

    Ok(results)
}

/// Generate code for a single pattern rule.
fn generate_rule(rule: &PatternRule, iter_ctx: Option<&IterContext>) -> Result<TokenStream2> {
    let pattern_output = generate_pattern(&rule.lhs, iter_ctx)?;
    let var_names = pattern_output.all_names();
    let pattern_code = pattern_output.code;

    let (bindings, rewrite_code) =
        generate_rewrite(&rule.lhs, &rule.rhs, &rule.guard, rule.arrow, iter_ctx, &var_names)?;

    Ok(quote! {
        __patterns.push((
            #pattern_code,
            Box::new(|__bindings: &morok_schedule::pattern::BindingStore, __intern: &morok_schedule::pattern::VarIntern| {
                use morok_schedule::pattern::BindingStoreExt;
                #(#bindings)*
                #rewrite_code
            }) as morok_schedule::pattern::matcher::RewriteFn,
        ));
    })
}

/// Generate a UPat expression from a pattern, returning both code and collected names.
fn generate_pattern(pattern: &Pattern, iter_ctx: Option<&IterContext>) -> Result<PatternOutput> {
    match pattern {
        Pattern::Wildcard => {
            // Wildcard matches any UOp but doesn't bind
            let code = quote! {
                morok_schedule::pattern::UPat::Match {
                    op: None,
                    dtype: None,
                    src: None,
                    arg: None,
                    name: None,
                }
            };
            Ok(PatternOutput::no_names(code))
        }

        Pattern::Var(name) => {
            let name_str = name.to_string();
            let code = quote! { morok_schedule::pattern::UPat::var(#name_str) };
            Ok(PatternOutput::self_only(code, name.clone()))
        }

        Pattern::Binding { name, pattern } => {
            let inner = generate_pattern(pattern, iter_ctx)?;
            let name_str = name.to_string();
            let inner_code = inner.code;
            let code = quote! { #inner_code.named(#name_str) };
            // KEY: .named() replaces inner's self_name, keeps inner's children
            Ok(PatternOutput { code, self_name: Some(name.clone()), child_names: inner.child_names })
        }

        Pattern::OpTuple { op, args, rest } => generate_op_tuple_pattern(op, args, iter_ctx, *rest),

        Pattern::OpStruct { op, fields, rest } => generate_op_struct_pattern(op, fields, *rest),

        Pattern::Const(const_pat) => generate_const_pattern(const_pat),

        Pattern::OpVar { var_name, args } => generate_op_var_pattern(var_name, args, iter_ctx),

        Pattern::ConstWithValue { uop_name, .. } => {
            // Generate same pattern as @const but with the uop_name binding
            let name_str = uop_name.to_string();
            let code = quote! { morok_schedule::pattern::UPat::cvar(#name_str) };
            // Only uop_name is bound; value_name is extracted separately in generate_rewrite
            Ok(PatternOutput::self_only(code, uop_name.clone()))
        }
    }
}

/// Classification of operations for code generation.
enum OpClass {
    /// Binary IR operations (Add, Sub, etc.) - uses BinaryOp enum
    Binary,
    /// Unary IR operations (Neg, Sqrt, etc.) - uses UnaryOp enum
    Unary,
    /// Ternary IR operations (Where, MulAcc) - uses TernaryOp enum
    Ternary,
    /// Single-source operations with a named UPat helper (cast, reshape, etc.)
    SingleSource(&'static str),
    /// Operations with special handling (Store, Load, etc.)
    Special,
}

/// Binary IR operations.
const BINARY_OPS: &[&str] = &[
    "Add", "Sub", "Mul", "Div", "Mod", "Max", "Lt", "Eq", "Ne", "And", "Or", "Xor", "Shl", "Shr", "Idiv", "Fdiv", "Pow",
];

/// Unary IR operations.
const UNARY_OPS: &[&str] = &["Neg", "Not", "Abs", "Sqrt", "Exp", "Log", "Sin", "Cos"];

/// Ternary IR operations.
const TERNARY_OPS: &[&str] = &["Where", "MulAcc"];

/// Single-source operations mapped to their UPat helper method names.
const SINGLE_SOURCE_OPS: &[(&str, &str)] = &[
    ("Detach", "detach"),
    ("ContiguousBackward", "contiguous_backward"),
    ("Cast", "cast"),
    ("Reshape", "reshape"),
    ("Permute", "permute"),
    ("Expand", "expand"),
    ("Pad", "pad"),
    ("Shrink", "shrink"),
    ("Flip", "flip"),
    ("Index", "index"),
    ("Range", "range"),
    ("After", "after"),
];

/// Classify an operation name into its category.
fn classify_op(op_name: &str) -> OpClass {
    if BINARY_OPS.contains(&op_name) {
        OpClass::Binary
    } else if UNARY_OPS.contains(&op_name) {
        OpClass::Unary
    } else if TERNARY_OPS.contains(&op_name) {
        OpClass::Ternary
    } else if let Some((_, helper)) = SINGLE_SOURCE_OPS.iter().find(|(name, _)| *name == op_name) {
        OpClass::SingleSource(helper)
    } else {
        OpClass::Special
    }
}

/// Generate pattern for tuple-style op: `Add(x, y)` or `End(comp, ..)`
fn generate_op_tuple_pattern(
    op: &Ident,
    args: &[Pattern],
    iter_ctx: Option<&IterContext>,
    rest: bool,
) -> Result<PatternOutput> {
    let op_name = op.to_string();

    // Collect all child patterns and their names
    let arg_outputs: Vec<PatternOutput> = args.iter().map(|a| generate_pattern(a, iter_ctx)).collect::<Result<_>>()?;

    // Collect all names from children in order
    let child_names: Vec<Ident> = arg_outputs.iter().flat_map(|o| o.all_names()).collect();

    // Extract codes for use in quote!
    let arg_codes: Vec<&TokenStream2> = arg_outputs.iter().map(|o| &o.code).collect();

    // Generate code based on operation classification
    let code = match classify_op(&op_name) {
        OpClass::Binary => {
            if args.len() != 2 {
                return Err(Error::new_spanned(op, format!("{} requires exactly 2 arguments", op_name)));
            }
            let left = &arg_codes[0];
            let right = &arg_codes[1];
            let binary_op = format_ident!("{}", op_name);
            quote! {
                morok_schedule::pattern::UPat::binary(
                    vec![morok_ir::BinaryOp::#binary_op],
                    vec![#left, #right]
                )
            }
        }

        OpClass::Unary => {
            if args.len() != 1 {
                return Err(Error::new_spanned(op, format!("{} requires exactly 1 argument", op_name)));
            }
            let arg = &arg_codes[0];
            let unary_op = format_ident!("{}", op_name);
            quote! {
                morok_schedule::pattern::UPat::unary(
                    vec![morok_ir::UnaryOp::#unary_op],
                    #arg
                )
            }
        }

        OpClass::Ternary => {
            if args.len() != 3 {
                return Err(Error::new_spanned(op, format!("{} requires exactly 3 arguments", op_name)));
            }
            let a = &arg_codes[0];
            let b = &arg_codes[1];
            let c = &arg_codes[2];
            let ternary_op = format_ident!("{}", op_name);
            quote! {
                morok_schedule::pattern::UPat::ternary(
                    vec![morok_ir::TernaryOp::#ternary_op],
                    vec![#a, #b, #c]
                )
            }
        }

        OpClass::SingleSource(helper) => {
            if args.len() != 1 {
                return Err(Error::new_spanned(op, format!("{} requires exactly 1 argument", op_name)));
            }
            let src = &arg_codes[0];
            let helper_ident = format_ident!("{}", helper);
            quote! { morok_schedule::pattern::UPat::#helper_ident(#src) }
        }

        OpClass::Special => generate_special_op_pattern(op, &op_name, args, &arg_codes, rest)?,
    };

    Ok(PatternOutput::children_only(code, child_names))
}

/// Generate pattern for special operations that need custom handling.
fn generate_special_op_pattern(
    op: &Ident,
    op_name: &str,
    args: &[Pattern],
    arg_codes: &[&TokenStream2],
    rest: bool,
) -> Result<TokenStream2> {
    match op_name {
        "Reduce" => {
            if rest {
                Ok(quote! { morok_schedule::pattern::UPat::reduce_any() })
            } else {
                if args.len() != 1 {
                    return Err(Error::new_spanned(
                        op,
                        "Reduce requires exactly 1 argument (or use `..` for variable ranges)",
                    ));
                }
                let src = &arg_codes[0];
                Ok(quote! { morok_schedule::pattern::UPat::reduce(#src) })
            }
        }

        "End" => {
            if rest {
                Ok(quote! { morok_schedule::pattern::UPat::end_any() })
            } else {
                if args.len() != 1 {
                    return Err(Error::new_spanned(
                        op,
                        "End requires exactly 1 argument (or use `..` for variable ranges)",
                    ));
                }
                let computation = &arg_codes[0];
                Ok(quote! { morok_schedule::pattern::UPat::end(#computation) })
            }
        }

        "Store" => {
            if args.len() != 3 {
                return Err(Error::new_spanned(op, "Store requires exactly 3 arguments (buffer, index, value)"));
            }
            let buffer = &arg_codes[0];
            let index = &arg_codes[1];
            let value = &arg_codes[2];
            Ok(quote! { morok_schedule::pattern::UPat::store(#buffer, #index, #value) })
        }

        "Load" => {
            if args.len() != 2 {
                return Err(Error::new_spanned(op, "Load requires exactly 2 arguments (buffer, index)"));
            }
            let buffer = &arg_codes[0];
            let index = &arg_codes[1];
            Ok(quote! { morok_schedule::pattern::UPat::load(#buffer, #index) })
        }

        "Bind" => {
            if args.len() != 2 {
                return Err(Error::new_spanned(op, "Bind requires exactly 2 arguments (var, value)"));
            }
            let var = &arg_codes[0];
            let value = &arg_codes[1];
            Ok(quote! { morok_schedule::pattern::UPat::bind(#var, #value) })
        }

        "Const" => Err(Error::new_spanned(op, "Use Const(value) or Const(_) syntax for constants")),

        _ => Err(Error::new_spanned(op, format!("Unknown operation: {}", op_name))),
    }
}

/// Generate pattern for struct-style op: `Bufferize { compute: x, .. }`
///
/// This generates a UPat that matches the op and binds it to "__struct_root",
/// allowing field extraction in the rewrite closure.
fn generate_op_struct_pattern(op: &Ident, fields: &[FieldPattern], _rest: bool) -> Result<PatternOutput> {
    let op_name = op.to_string();

    // Find the first UOp child field (the main source pattern)
    let first_field =
        fields.first().ok_or_else(|| Error::new_spanned(op, "Struct pattern must have at least one field"))?;

    let first_output = generate_pattern(&first_field.pattern, None)?;
    let first_code = &first_output.code;

    // Collect child names from the first field (these are nested, so use all_names)
    let child_names: Vec<Ident> = first_output.all_names();

    // Generate the base pattern
    let base = match op_name.as_str() {
        "Bufferize" => quote! { #first_code.f_bufferize() },
        "Index" => quote! { #first_code.f_index() },
        "Cast" => quote! { #first_code.f_cast() },
        "Reshape" => quote! { #first_code.f_reshape() },
        "Permute" => quote! { #first_code.f_permute() },
        "Expand" => quote! { #first_code.f_expand() },
        "Reduce" => quote! { #first_code.f_reduce() },
        "Detach" => quote! { morok_schedule::pattern::UPat::detach(#first_code) },
        "ContiguousBackward" => quote! { morok_schedule::pattern::UPat::contiguous_backward(#first_code) },
        _ => {
            return Err(Error::new_spanned(op, format!("Unknown operation for struct pattern: {}", op_name)));
        }
    };

    // If there are fields beyond the main UOp child that need extraction,
    // bind the root op so we can extract them in the rewrite closure
    if has_extractable_fields(fields) {
        let name = binding_names::STRUCT_ROOT;
        let code = quote! { #base.named(#name) };
        // __struct_root becomes the self_name (top-level binding)
        Ok(PatternOutput {
            code,
            self_name: Some(Ident::new(binding_names::STRUCT_ROOT, proc_macro2::Span::call_site())),
            child_names,
        })
    } else {
        // No self_name, just child names from the first field
        Ok(PatternOutput::children_only(base, child_names))
    }
}

/// Generate pattern for constant: `Const(_)`, `Const(0)`, `@zero`, `@one`
fn generate_const_pattern(const_pat: &ConstPattern) -> Result<PatternOutput> {
    match const_pat {
        ConstPattern::Any => {
            let name = binding_names::CONST;
            let code = quote! { morok_schedule::pattern::UPat::cvar(#name) };
            Ok(PatternOutput::self_only(code, Ident::new(binding_names::CONST, proc_macro2::Span::call_site())))
        }
        ConstPattern::Int(0) | ConstPattern::Zero => {
            let name = binding_names::ZERO;
            let code = quote! { morok_schedule::pattern::UPat::zero_const(#name) };
            Ok(PatternOutput::self_only(code, Ident::new(binding_names::ZERO, proc_macro2::Span::call_site())))
        }
        ConstPattern::Int(value) => {
            let code = quote! { morok_schedule::pattern::UPat::int(#value) };
            // Int patterns (except 0) don't bind - no names
            Ok(PatternOutput::no_names(code))
        }
        ConstPattern::Float(value) => {
            let code = quote! { morok_schedule::pattern::UPat::float(#value) };
            // Float patterns don't bind - no names
            Ok(PatternOutput::no_names(code))
        }
        ConstPattern::One => {
            let name = binding_names::ONE;
            let code = quote! { morok_schedule::pattern::UPat::one_const(#name) };
            Ok(PatternOutput::self_only(code, Ident::new(binding_names::ONE, proc_macro2::Span::call_site())))
        }
    }
}

/// Generate pattern for operation variable: `op(x, y)` where `op` is an iteration variable.
fn generate_op_var_pattern(
    var_name: &Ident,
    args: &[Pattern],
    iter_ctx: Option<&IterContext>,
) -> Result<PatternOutput> {
    let ctx = iter_ctx.ok_or_else(|| Error::new_spanned(var_name, "Operation variable used outside of for-block"))?;

    // Verify the variable name matches the iteration variable
    if var_name != &ctx.var_name {
        return Err(Error::new_spanned(
            var_name,
            format!("Unknown operation variable '{}', expected '{}'", var_name, ctx.var_name),
        ));
    }

    let op_ident = &ctx.op_ident;

    // Collect all child patterns and their names
    let arg_outputs: Vec<PatternOutput> = args.iter().map(|a| generate_pattern(a, iter_ctx)).collect::<Result<_>>()?;

    // Collect all names from children in order
    let child_names: Vec<Ident> = arg_outputs.iter().flat_map(|o| o.all_names()).collect();

    // Extract codes for use in quote!
    let arg_codes: Vec<&TokenStream2> = arg_outputs.iter().map(|o| &o.code).collect();

    let code = match ctx.op_kind {
        OpKind::Unary => {
            if args.len() != 1 {
                return Err(Error::new_spanned(var_name, "Unary operation requires exactly 1 argument"));
            }
            let arg = &arg_codes[0];
            quote! {
                morok_schedule::pattern::UPat::unary(
                    vec![morok_ir::UnaryOp::#op_ident],
                    #arg
                )
            }
        }
        OpKind::Binary => {
            if args.len() != 2 {
                return Err(Error::new_spanned(var_name, "Binary operation requires exactly 2 arguments"));
            }
            let left = &arg_codes[0];
            let right = &arg_codes[1];
            quote! {
                morok_schedule::pattern::UPat::binary(
                    vec![morok_ir::BinaryOp::#op_ident],
                    vec![#left, #right]
                )
            }
        }
        OpKind::Ternary => {
            if args.len() != 3 {
                return Err(Error::new_spanned(var_name, "Ternary operation requires exactly 3 arguments"));
            }
            let cond = &arg_codes[0];
            let then_val = &arg_codes[1];
            let else_val = &arg_codes[2];
            quote! {
                morok_schedule::pattern::UPat::ternary(
                    vec![morok_ir::TernaryOp::#op_ident],
                    vec![#cond, #then_val, #else_val]
                )
            }
        }
    };

    Ok(PatternOutput::children_only(code, child_names))
}

/// Generate variable bindings and rewrite code.
fn generate_rewrite(
    lhs: &Pattern,
    rhs: &RewriteExpr,
    guard: &Option<syn::Expr>,
    arrow: ArrowKind,
    iter_ctx: Option<&IterContext>,
    var_names: &[Ident],
) -> Result<(Vec<TokenStream2>, TokenStream2)> {
    // Build a map of variable name to index for compile-time lookup
    // The order matches collect_var_names_internal in upat.rs
    let mut seen_names = std::collections::HashSet::new();
    let mut unique_var_names = Vec::new();
    for name in var_names {
        let name_str = name.to_string();
        if seen_names.insert(name_str) {
            unique_var_names.push(name.clone());
        }
    }

    // Generate bindings using compile-time indices (no runtime string lookup!)
    // The index matches the order that VarIntern::get_or_insert assigns indices
    let mut bindings: Vec<TokenStream2> = unique_var_names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let idx_u8 = idx as u8;
            quote! {
                let #name = match __bindings.get_by_index(#idx_u8) {
                    Some(v) => v,
                    None => return morok_schedule::pattern::matcher::RewriteResult::NoMatch,
                };
            }
        })
        .collect();

    // Add operation variable binding if in iteration context
    if let Some(ctx) = iter_ctx {
        let var_name = &ctx.var_name;
        let op_ident = &ctx.op_ident;
        let op_binding = match ctx.op_kind {
            OpKind::Unary => quote! {
                let #var_name = morok_ir::UnaryOp::#op_ident;
            },
            OpKind::Binary => quote! {
                let #var_name = morok_ir::BinaryOp::#op_ident;
            },
            OpKind::Ternary => quote! {
                let #var_name = morok_ir::TernaryOp::#op_ident;
            },
        };
        // Prepend op_binding to bindings
        bindings.insert(0, op_binding);
    }

    // Generate field extraction for struct patterns
    if let Pattern::OpStruct { op, fields, .. } = lhs {
        let field_extractions = generate_struct_field_extractions(op, fields)?;
        bindings.extend(field_extractions);
    }

    // Generate ConstValue extraction for ConstWithValue patterns
    let const_value_bindings = collect_const_value_bindings(lhs);
    for (uop_name, value_name) in const_value_bindings {
        bindings.push(quote! {
            let #value_name = match #uop_name.op() {
                morok_ir::Op::Const(cv) => cv.0,
                _ => return morok_schedule::pattern::matcher::RewriteResult::NoMatch,
            };
        });
    }

    // Generate rewrite code based on arrow kind
    let rewrite_code = match arrow {
        ArrowKind::Infallible => generate_infallible_rewrite(rhs, guard),
        ArrowKind::Fallible => generate_fallible_rewrite(rhs, guard),
    };

    Ok((bindings, rewrite_code))
}

/// Generate code for `~>` (infallible) - RHS returns Rc<UOp>
fn generate_infallible_rewrite(rhs: &RewriteExpr, guard: &Option<syn::Expr>) -> TokenStream2 {
    let rhs_expr = match rhs {
        RewriteExpr::Var(name) => quote! { std::rc::Rc::clone(#name) },
        RewriteExpr::Expr(expr) => quote! { #expr },
        RewriteExpr::Block(block) => quote! { #block },
    };

    if let Some(guard_expr) = guard {
        quote! {
            if #guard_expr {
                morok_schedule::pattern::matcher::RewriteResult::Rewritten(#rhs_expr)
            } else {
                morok_schedule::pattern::matcher::RewriteResult::NoMatch
            }
        }
    } else {
        quote! {
            morok_schedule::pattern::matcher::RewriteResult::Rewritten(#rhs_expr)
        }
    }
}

/// Generate code for `=>` (fallible) - RHS returns Option<Rc<UOp>>
fn generate_fallible_rewrite(rhs: &RewriteExpr, guard: &Option<syn::Expr>) -> TokenStream2 {
    let rhs_expr = match rhs {
        // For simple variable, wrap in Some() for convenience
        RewriteExpr::Var(name) => quote! { Some(std::rc::Rc::clone(#name)) },
        // For expressions and blocks, wrap in a closure so `?` operator works
        // The closure returns Option<Rc<UOp>>
        RewriteExpr::Expr(expr) => quote! { (|| #expr)() },
        RewriteExpr::Block(block) => quote! { (|| #block)() },
    };

    let conversion = quote! {
        match #rhs_expr {
            Some(__v) => morok_schedule::pattern::matcher::RewriteResult::Rewritten(__v),
            None => morok_schedule::pattern::matcher::RewriteResult::NoMatch,
        }
    };

    if let Some(guard_expr) = guard {
        quote! {
            if #guard_expr {
                #conversion
            } else {
                morok_schedule::pattern::matcher::RewriteResult::NoMatch
            }
        }
    } else {
        conversion
    }
}

/// Operations that support field extraction in struct patterns.
const EXTRACTABLE_OPS: &[&str] = &["Cast", "Permute", "Reduce", "Bufferize", "Reshape", "Expand", "Index"];

/// Generate field extraction code for a single field from a struct op.
fn generate_field_extraction(op_name: &str, field_name: &Ident) -> TokenStream2 {
    let struct_root = binding_names::STRUCT_ROOT;
    let op_ident = format_ident!("{}", op_name);
    quote! {
        let #field_name = {
            let __root = match __intern.get_index(#struct_root).and_then(|i| __bindings.get_by_index(i)) {
                Some(v) => v,
                None => return morok_schedule::pattern::matcher::RewriteResult::NoMatch,
            };
            match __root.op() {
                morok_ir::Op::#op_ident { #field_name, .. } => #field_name.clone(),
                _ => return morok_schedule::pattern::matcher::RewriteResult::NoMatch,
            }
        };
    }
}

/// Generate field extraction code for struct patterns.
///
/// For patterns like `Cast { src: x, dtype }`, this generates code to extract
/// the `dtype` field from the matched Cast op.
fn generate_struct_field_extractions(op: &Ident, fields: &[FieldPattern]) -> Result<Vec<TokenStream2>> {
    let op_name = op.to_string();

    // Check if this op supports field extraction
    if !EXTRACTABLE_OPS.contains(&op_name.as_str()) {
        return Ok(vec![]);
    }

    let mut extractions = Vec::new();

    // Skip the first field (it's the main UOp child, already bound)
    for field in fields.iter().skip(1) {
        // Only extract if it's a simple variable matching the field name (shorthand syntax)
        if let Pattern::Var(var) = &field.pattern
            && *var == field.name
        {
            extractions.push(generate_field_extraction(&op_name, &field.name));
        }
    }

    Ok(extractions)
}

/// Collect ConstWithValue patterns for value binding generation.
fn collect_const_value_bindings(pattern: &Pattern) -> Vec<(Ident, Ident)> {
    let mut bindings = Vec::new();
    collect_const_value_bindings_recursive(pattern, &mut bindings);
    bindings
}

fn collect_const_value_bindings_recursive(pattern: &Pattern, bindings: &mut Vec<(Ident, Ident)>) {
    match pattern {
        Pattern::ConstWithValue { uop_name, value_name } => {
            bindings.push((uop_name.clone(), value_name.clone()));
        }
        Pattern::OpTuple { args, .. } => {
            for arg in args {
                collect_const_value_bindings_recursive(arg, bindings);
            }
        }
        Pattern::OpStruct { fields, .. } => {
            for field in fields {
                collect_const_value_bindings_recursive(&field.pattern, bindings);
            }
        }
        Pattern::Binding { pattern, .. } => {
            collect_const_value_bindings_recursive(pattern, bindings);
        }
        Pattern::OpVar { args, .. } => {
            for arg in args {
                collect_const_value_bindings_recursive(arg, bindings);
            }
        }
        _ => {}
    }
}
