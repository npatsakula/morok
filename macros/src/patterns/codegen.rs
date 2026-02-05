//! Code generation for the pattern DSL.
//!
//! Converts the parsed pattern AST into Rust code that constructs
//! a `TypedPatternMatcher` using the new TypedPattern infrastructure.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Error, Ident, Result};

use super::parser::{
    ArrowKind, ConstPattern, FieldPattern, ForBlock, IterKind, Pattern, PatternItem, PatternList, PatternRule,
    RewriteExpr,
};

/// Internal binding names used by the code generator.
mod binding_names {
    /// Name for any constant binding
    pub const CONST: &str = "__const";
    /// Name for zero constant binding
    pub const ZERO: &str = "__zero";
    /// Name for one constant binding
    pub const ONE: &str = "__one";
}

/// Tracks duplicate variable names for auto ptr_eq generation.
///
/// When a pattern like `Add(x, x)` is used, this tracker:
/// 1. Records the first `x` normally
/// 2. Renames the second `x` to `x__dup` and records the pair
///
/// The rewrite function then generates `Arc::ptr_eq(&x, &x__dup)` checks.
#[derive(Default, Clone)]
struct DuplicateTracker {
    /// Variable names we've seen so far
    seen: std::collections::HashSet<String>,
    /// Pairs of (original_name, duplicate_name) for ptr_eq generation
    duplicates: Vec<(String, String)>,
}

impl DuplicateTracker {
    /// Process a variable name, returning the name to use in the pattern.
    /// If this is a duplicate, returns a renamed version and records the pair.
    ///
    /// For patterns with 3+ occurrences of the same variable (e.g., `Where(x, x, x)`),
    /// generates unique names: x, x_dup, x_dup_2, x_dup_3, etc.
    fn process_name(&mut self, name: &str) -> String {
        if self.seen.contains(name) {
            // Count existing duplicates of this name to generate unique suffix
            let count = self.duplicates.iter().filter(|(orig, _)| orig == name).count();
            let dup_name = if count == 0 { format!("{}_dup", name) } else { format!("{}_dup_{}", name, count + 1) };
            self.duplicates.push((name.to_string(), dup_name.clone()));
            dup_name
        } else {
            self.seen.insert(name.to_string());
            name.to_string()
        }
    }

    /// Get the duplicate pairs for ptr_eq generation.
    fn get_duplicates(&self) -> &[(String, String)] {
        &self.duplicates
    }
}

// =============================================================================
// Closure Parameter Validation
// =============================================================================

/// Information about a single closure parameter.
struct ClosureParamInfo {
    name: Ident,
    is_ctx: bool,
}

/// Analysis result for a user closure.
struct ClosureAnalysis {
    params: Vec<ClosureParamInfo>,
}

/// Extract name from a closure parameter pattern.
fn extract_pat_name(pat: &syn::Pat) -> Result<Ident> {
    match pat {
        syn::Pat::Ident(pat_ident) => Ok(pat_ident.ident.clone()),
        syn::Pat::Type(pat_type) => extract_pat_name(&pat_type.pat),
        syn::Pat::Reference(pat_ref) => extract_pat_name(&pat_ref.pat),
        _ => Err(Error::new_spanned(pat, "Closure parameters must be simple identifiers")),
    }
}

/// Extract parameter information from a user closure.
fn extract_closure_params(closure: &syn::ExprClosure) -> Result<ClosureAnalysis> {
    let params = closure
        .inputs
        .iter()
        .map(|pat| {
            let name = extract_pat_name(pat)?;
            let is_ctx = name == "ctx";
            Ok(ClosureParamInfo { name, is_ctx })
        })
        .collect::<Result<_>>()?;
    Ok(ClosureAnalysis { params })
}

/// Validate that closure parameters match requirements.
/// Main purpose: catch `ctx` without `@context` declaration early.
fn validate_closure_params(analysis: &ClosureAnalysis, _lhs_bindings: &[Ident], has_context: bool) -> Result<()> {
    for param in &analysis.params {
        if param.is_ctx && !has_context {
            return Err(Error::new_spanned(
                &param.name,
                "`ctx` parameter requires `@context Type;` declaration at start of patterns!",
            ));
        }
    }
    Ok(())
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

/// Classification of operations for code generation.
enum OpClass {
    /// Binary IR operations (Add, Sub, etc.) - uses BinaryOp enum
    Binary,
    /// Unary IR operations (Neg, Sqrt, etc.) - uses UnaryOp enum
    Unary,
    /// Ternary IR operations (Where, MulAcc) - uses TernaryOp enum
    Ternary,
    /// Single-source operations (cast, reshape, etc.)
    SingleSource,
    /// Operations with special handling (Store, Load, etc.)
    Special,
}

/// Binary IR operations.
/// NOTE: Must stay in sync with morok_ir::types::BinaryOp variants.
/// The generated metadata at morok_ir::op::pattern_derived::pattern_metadata::BINARY_OPS
/// can be used for runtime validation.
const BINARY_OPS: &[&str] = &[
    "Add", "Mul", "Sub", "Mod", "Max", "Pow", "Idiv", "Fdiv", "Lt", "Le", "Eq", "Ne", "Gt", "Ge", "And", "Or", "Xor",
    "Shl", "Shr", "Threefry",
];

/// Unary IR operations.
/// NOTE: Must stay in sync with morok_ir::types::UnaryOp variants.
const UNARY_OPS: &[&str] = &[
    "Neg",
    "Not",
    "Abs",
    "Sqrt",
    "Rsqrt",
    "Exp",
    "Exp2",
    "Log",
    "Log2",
    "Sin",
    "Cos",
    "Tan",
    "Reciprocal",
    "Trunc",
    "Floor",
    "Ceil",
    "Round",
    "Sign",
    "Erf",
    "Square",
];

/// Ternary IR operations.
/// NOTE: Must stay in sync with morok_ir::types::TernaryOp variants.
const TERNARY_OPS: &[&str] = &["Where", "MulAcc"];

/// Single-source operations mapped to their pattern helper method names.
/// NOTE: These ops must have `src` as their first child field.
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
    ("Contract", "contract"),
    ("Unroll", "unroll"),
    ("Contiguous", "contiguous"),
    ("Precast", "precast"),
    ("BitCast", "bitcast"),
    ("ReduceAxis", "reduce_axis"),
    ("Multi", "multi"),
];

/// Map of operation names to their child field names (in positional order).
///
/// This is used for tuple-style pattern matching: `Op(x, y)` where we need
/// to know that `x` maps to `buffer` and `y` maps to `index` for Load.
///
/// Ops not in this list default to single `src` field handling via SINGLE_SOURCE_OPS.
const OP_CHILD_FIELDS: &[(&str, &[&str])] = &[
    // Control flow
    ("Range", &["end"]),
    ("Special", &["end"]),
    ("After", &["passthrough"]),
    ("EndIf", &["if_op"]),
    ("End", &["computation"]),
    ("If", &["condition"]),
    ("Barrier", &["src"]),
    // Buffer ops
    ("Buffer", &["unique", "device"]),
    ("BufferView", &["buffer"]),
    ("MSelect", &["buffer"]),
    ("Index", &["buffer"]),
    ("PointerIndex", &["ptr", "offset"]),
    ("Copy", &["src", "device"]),
    ("Bufferize", &["compute"]),
    // Memory ops
    ("Load", &["buffer", "index"]),
    ("LoadGated", &["buffer", "index", "gate"]),
    ("Store", &["buffer", "index", "value"]),
    ("StoreGated", &["buffer", "index", "value", "gate"]),
    // Symbolic
    ("Bind", &["var", "value"]),
    ("Assign", &["target", "value", "movement_ops"]),
    // Reduction
    ("Reduce", &["src"]),
    ("AllReduce", &["src", "device"]),
    // WMMA
    ("Wmma", &["a", "b", "c"]),
    // Kernel
    ("Kernel", &["ast"]), // sources is variadic
];

/// Get child field names for an operation.
///
/// Returns the field names from OP_CHILD_FIELDS, or &["src"] for single-source ops.
fn get_child_field_names(op_name: &str) -> Option<&'static [&'static str]> {
    OP_CHILD_FIELDS.iter().find(|(name, _)| *name == op_name).map(|(_, fields)| *fields)
}

/// Classify an operation name into its category.
fn classify_op(op_name: &str) -> OpClass {
    if BINARY_OPS.contains(&op_name) {
        OpClass::Binary
    } else if UNARY_OPS.contains(&op_name) {
        OpClass::Unary
    } else if TERNARY_OPS.contains(&op_name) {
        OpClass::Ternary
    } else if SINGLE_SOURCE_OPS.iter().any(|(name, _)| *name == op_name) {
        OpClass::SingleSource
    } else {
        OpClass::Special
    }
}

// =============================================================================
// Compile-Time Op Name Validation
// =============================================================================

/// Collect all operation names used in a pattern list.
fn collect_op_names(patterns: &PatternList) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for item in &patterns.items {
        match item {
            PatternItem::Rule(rule) => collect_op_names_from_pattern(&rule.lhs, &mut names),
            PatternItem::ForBlock(for_block) => {
                for rule in &for_block.body {
                    collect_op_names_from_pattern(&rule.lhs, &mut names);
                }
            }
        }
    }
    names
}

/// Recursively collect op names from a pattern.
fn collect_op_names_from_pattern(pattern: &Pattern, names: &mut std::collections::HashSet<String>) {
    match pattern {
        Pattern::OpStruct { op, fields, .. } => {
            names.insert(op.to_string());
            for field in fields {
                collect_op_names_from_field_pattern(field, names);
            }
        }
        Pattern::OpTuple { op, args, .. } => {
            names.insert(op.to_string());
            for arg in args {
                collect_op_names_from_pattern(arg, names);
            }
        }
        Pattern::OpPermute { op, args } => {
            names.insert(op.to_string());
            for arg in args {
                collect_op_names_from_pattern(arg, names);
            }
        }
        Pattern::Binding { pattern, .. } => {
            collect_op_names_from_pattern(pattern, names);
        }
        Pattern::Any(alternatives) => {
            for alt in alternatives {
                collect_op_names_from_pattern(alt, names);
            }
        }
        // Constants, variables, wildcards, op variables don't have op names to validate
        // (OpVar uses iteration context which is separately validated)
        Pattern::Const(_)
        | Pattern::Var(_)
        | Pattern::OpVar { .. }
        | Pattern::ConstWithValue { .. }
        | Pattern::VConstWithValue { .. }
        | Pattern::AnyConstWithValue { .. }
        | Pattern::Wildcard
        | Pattern::OptionNone => {}
        // OptionSome has an inner pattern that may contain ops
        Pattern::OptionSome(inner) => {
            collect_op_names_from_pattern(inner, names);
        }
    }
}

/// Collect op names from a field pattern.
fn collect_op_names_from_field_pattern(field: &FieldPattern, names: &mut std::collections::HashSet<String>) {
    collect_op_names_from_pattern(&field.pattern, names);
}

/// Generate compile-time validation assertions for op names.
///
/// This ensures typos like `Addd(x, y)` fail at compile time with a clear error.
fn generate_op_validation(op_names: &std::collections::HashSet<String>) -> TokenStream2 {
    let validations: Vec<TokenStream2> = op_names
        .iter()
        .map(|name| {
            let op_class = classify_op(name);
            let op_ident = format_ident!("{}", name);
            match op_class {
                OpClass::Binary => quote! {
                    let _ = morok_ir::BinaryOp::#op_ident;
                },
                OpClass::Unary => quote! {
                    let _ = morok_ir::UnaryOp::#op_ident;
                },
                OpClass::Ternary => quote! {
                    let _ = morok_ir::TernaryOp::#op_ident;
                },
                OpClass::SingleSource | OpClass::Special => quote! {
                    let _ = morok_ir::op::pattern_derived::OpKey::#op_ident;
                },
            }
        })
        .collect();

    if validations.is_empty() {
        quote! {}
    } else {
        quote! {
            // Compile-time validation: ensure all op names exist
            const _: () = {
                #(#validations)*
            };
        }
    }
}

// =============================================================================
// SimplifiedPatternMatcher Code Generation (Phase 3)
// =============================================================================

/// Compute OpKey(s) for a pattern.
///
/// Returns TokenStream for the OpKey array used by SimplifiedPatternMatcher.
/// For wildcard patterns (bare variables), returns an empty array.
fn compute_op_keys(pattern: &Pattern, iter_ctx: Option<&IterContext>) -> Vec<TokenStream2> {
    match pattern {
        // Bare variable or wildcard - matches any op (wildcard pattern)
        Pattern::Var(_) | Pattern::Wildcard => vec![],

        // Binding - delegate to inner pattern
        Pattern::Binding { pattern, .. } => compute_op_keys(pattern, iter_ctx),

        // Op tuple - extract OpKey from op name
        Pattern::OpTuple { op, .. } => {
            let op_name = op.to_string();
            compute_op_key_for_op(&op_name)
        }

        // Op struct - extract OpKey from op name
        Pattern::OpStruct { op, .. } => {
            let op_name = op.to_string();
            compute_op_key_for_op(&op_name)
        }

        // Constants - OpKey::Const
        Pattern::Const(_) | Pattern::ConstWithValue { .. } => {
            vec![quote! { morok_ir::op::pattern_derived::OpKey::Const }]
        }

        // VConst - OpKey::VConst
        Pattern::VConstWithValue { .. } => {
            vec![quote! { morok_ir::op::pattern_derived::OpKey::VConst }]
        }

        // AnyConst - matches both Const and VConst
        Pattern::AnyConstWithValue { .. } => {
            vec![
                quote! { morok_ir::op::pattern_derived::OpKey::Const },
                quote! { morok_ir::op::pattern_derived::OpKey::VConst },
            ]
        }

        // Op variable (in for-loop) - get OpKey from iteration context
        Pattern::OpVar { var_name, .. } => {
            if let Some(ctx) = iter_ctx {
                if var_name == &ctx.var_name {
                    let op_ident = &ctx.op_ident;
                    match ctx.op_kind {
                        OpKind::Unary => vec![quote! {
                            morok_ir::op::pattern_derived::OpKey::Unary(morok_ir::UnaryOp::#op_ident)
                        }],
                        OpKind::Binary => vec![quote! {
                            morok_ir::op::pattern_derived::OpKey::Binary(morok_ir::BinaryOp::#op_ident)
                        }],
                        OpKind::Ternary => vec![quote! {
                            morok_ir::op::pattern_derived::OpKey::Ternary(morok_ir::TernaryOp::#op_ident)
                        }],
                    }
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        }

        // Alternatives - collect all OpKeys from all alternatives
        Pattern::Any(alternatives) => {
            let mut keys = Vec::new();
            for alt in alternatives {
                keys.extend(compute_op_keys(alt, iter_ctx));
            }
            // Deduplicate by string representation
            let mut seen = std::collections::HashSet::new();
            keys.retain(|k| seen.insert(k.to_string()));
            keys
        }

        // Permutation - same OpKey as non-permuted
        Pattern::OpPermute { op, .. } => {
            let op_name = op.to_string();
            compute_op_key_for_op(&op_name)
        }

        // Option patterns - these are for matching Option<T> fields, not top-level ops
        // They act like wildcards at the top level (shouldn't be used as top-level patterns)
        Pattern::OptionSome(inner) => compute_op_keys(inner, iter_ctx),
        Pattern::OptionNone => vec![],
    }
}

/// Get OpKey for a named operation.
fn compute_op_key_for_op(op_name: &str) -> Vec<TokenStream2> {
    // Check if it's a binary op
    if BINARY_OPS.contains(&op_name) {
        let op_ident = format_ident!("{}", op_name);
        return vec![quote! {
            morok_ir::op::pattern_derived::OpKey::Binary(morok_ir::BinaryOp::#op_ident)
        }];
    }

    // Check if it's a unary op
    if UNARY_OPS.contains(&op_name) {
        let op_ident = format_ident!("{}", op_name);
        return vec![quote! {
            morok_ir::op::pattern_derived::OpKey::Unary(morok_ir::UnaryOp::#op_ident)
        }];
    }

    // Check if it's a ternary op
    if TERNARY_OPS.contains(&op_name) {
        let op_ident = format_ident!("{}", op_name);
        return vec![quote! {
            morok_ir::op::pattern_derived::OpKey::Ternary(morok_ir::TernaryOp::#op_ident)
        }];
    }

    // For other ops, use the variant-specific OpKey
    let op_ident = format_ident!("{}", op_name);
    vec![quote! {
        morok_ir::op::pattern_derived::OpKey::#op_ident
    }]
}

/// Represents one possible ordering for pattern matching.
/// For non-commutative patterns, there's exactly 1 ordering.
/// For commutative patterns, there are 2+ orderings (cross-product of nested commutative).
#[derive(Clone)]
struct Ordering {
    /// Code to match children in this ordering
    child_match_code: TokenStream2,
    /// Variable bindings for this ordering
    bindings: Vec<(Ident, TokenStream2)>,
}

/// Output from inline match generation.
struct InlineMatchOutput {
    /// Code to destructure the outer UOp (independent of orderings)
    match_code: TokenStream2,
    /// All possible orderings to try (1 for non-commutative, 2+ for commutative)
    orderings: Vec<Ordering>,
}

impl InlineMatchOutput {
    /// Create a simple non-commutative output with one ordering
    fn simple(match_code: TokenStream2, bindings: Vec<(Ident, TokenStream2)>) -> Self {
        Self { match_code, orderings: vec![Ordering { child_match_code: quote! {}, bindings }] }
    }

    /// Convenience: get bindings from first ordering (for non-commutative or normal ordering)
    fn bindings(&self) -> &[(Ident, TokenStream2)] {
        &self.orderings[0].bindings
    }
}

/// Generate inline match code for a pattern.
///
/// This generates Rust code that destructures the UOp and binds variables.
/// Returns the match code and a list of bindings.
fn generate_inline_match(
    pattern: &Pattern,
    tree_var: &Ident,
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    match pattern {
        Pattern::Var(name) => {
            // Variable binds the whole tree
            let actual_name = dup_tracker.process_name(&name.to_string());
            let name_ident = Ident::new(&actual_name, name.span());
            Ok(InlineMatchOutput::simple(quote! {}, vec![(name_ident, quote! { #tree_var })]))
        }

        Pattern::Wildcard => {
            // Wildcard matches anything, no bindings
            Ok(InlineMatchOutput::simple(quote! {}, vec![]))
        }

        Pattern::Binding { name, pattern } => {
            // Named binding wraps inner pattern
            let inner = generate_inline_match(pattern, tree_var, iter_ctx, dup_tracker)?;
            let actual_name = dup_tracker.process_name(&name.to_string());
            let name_ident = Ident::new(&actual_name, name.span());

            // Add the binding to all orderings
            let orderings = inner
                .orderings
                .into_iter()
                .map(|mut ord| {
                    ord.bindings.push((name_ident.clone(), quote! { #tree_var }));
                    ord
                })
                .collect();

            Ok(InlineMatchOutput { match_code: inner.match_code, orderings })
        }

        Pattern::OpTuple { op, args, rest: _ } => {
            generate_inline_op_tuple_match(op, args, tree_var, iter_ctx, dup_tracker)
        }

        Pattern::OpStruct { op, fields, rest: _ } => generate_inline_op_struct_match(op, fields, tree_var, dup_tracker),

        Pattern::Const(const_pat) => generate_inline_const_match(const_pat, tree_var),

        Pattern::ConstWithValue { uop_name, value_name } => {
            // Match Const and extract both the UOp and the value
            let actual_name = dup_tracker.process_name(&uop_name.to_string());
            let uop_ident = Ident::new(&actual_name, uop_name.span());
            Ok(InlineMatchOutput::simple(
                quote! {
                    let morok_ir::Op::Const(__cv) = #tree_var.op() else {
                        return morok_ir::pattern::RewriteResult::NoMatch;
                    };
                    let #value_name = __cv.0.clone();
                },
                vec![(uop_ident, quote! { #tree_var })],
            ))
        }

        Pattern::VConstWithValue { uop_name, values_name } => {
            // Match VConst and extract both the UOp and the values
            let actual_name = dup_tracker.process_name(&uop_name.to_string());
            let uop_ident = Ident::new(&actual_name, uop_name.span());
            Ok(InlineMatchOutput::simple(
                quote! {
                    let morok_ir::Op::VConst { values: __vconst_values } = #tree_var.op() else {
                        return morok_ir::pattern::RewriteResult::NoMatch;
                    };
                    let #values_name = __vconst_values.clone();
                },
                vec![(uop_ident, quote! { #tree_var })],
            ))
        }

        Pattern::AnyConstWithValue { uop_name, values_name } => {
            // Match either Const or VConst and extract values as Vec<ConstValue>
            let actual_name = dup_tracker.process_name(&uop_name.to_string());
            let uop_ident = Ident::new(&actual_name, uop_name.span());
            Ok(InlineMatchOutput::simple(
                quote! {
                    let #values_name: Vec<morok_ir::ConstValue> = match #tree_var.op() {
                        morok_ir::Op::Const(cv) => vec![cv.0.clone()],
                        morok_ir::Op::VConst { values } => values.clone(),
                        _ => return morok_ir::pattern::RewriteResult::NoMatch,
                    };
                },
                vec![(uop_ident, quote! { #tree_var })],
            ))
        }

        Pattern::OpVar { var_name, args } => {
            generate_inline_op_var_match(var_name, args, tree_var, iter_ctx, dup_tracker)
        }

        Pattern::Any(alternatives) => generate_inline_alternatives_match(alternatives, tree_var, iter_ctx, dup_tracker),

        Pattern::OpPermute { op, args } => generate_inline_commutative_match(op, args, tree_var, iter_ctx, dup_tracker),

        Pattern::OptionNone => {
            // Match Option::None - reject if value is Some
            Ok(InlineMatchOutput::simple(
                quote! {
                    if #tree_var.is_some() {
                        return morok_ir::pattern::RewriteResult::NoMatch;
                    }
                },
                vec![],
            ))
        }

        Pattern::OptionSome(inner) => {
            // Match Option::Some and recursively match inner pattern
            let inner_var = format_ident!("{}_some", tree_var);
            let inner_match = generate_inline_match(inner, &inner_var, iter_ctx, dup_tracker)?;
            let inner_code = &inner_match.match_code;

            // Prepend the Option unwrap to match_code, keep inner's orderings
            let match_code = quote! {
                let Some(#inner_var) = #tree_var else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
                #inner_code
            };

            Ok(InlineMatchOutput { match_code, orderings: inner_match.orderings })
        }
    }
}

/// Generate inline match for tuple-style ops like Add(x, y).
fn generate_inline_op_tuple_match(
    op: &Ident,
    args: &[Pattern],
    tree_var: &Ident,
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    let op_name = op.to_string();

    match classify_op(&op_name) {
        OpClass::Binary => {
            if args.len() != 2 {
                return Err(Error::new_spanned(op, format!("{} requires exactly 2 arguments", op_name)));
            }

            let binary_op = format_ident!("{}", op_name);
            // Use unique names based on tree_var to avoid shadowing in nested patterns
            let left_var = format_ident!("{}_left", tree_var);
            let right_var = format_ident!("{}_right", tree_var);

            // Generate match code
            let match_code = quote! {
                let morok_ir::Op::Binary(morok_ir::BinaryOp::#binary_op, #left_var, #right_var) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
            };

            // Recursively match children
            let left_match = generate_inline_match(&args[0], &left_var, iter_ctx, dup_tracker)?;
            let right_match = generate_inline_match(&args[1], &right_var, iter_ctx, dup_tracker)?;

            let left_code = &left_match.match_code;
            let right_code = &right_match.match_code;

            // Cross-product all orderings from children
            // For each left ordering × each right ordering, create a combined ordering
            let orderings: Vec<Ordering> = left_match
                .orderings
                .iter()
                .flat_map(|left_ord| {
                    let left_child = &left_ord.child_match_code;
                    right_match.orderings.iter().map(move |right_ord| {
                        let right_child = &right_ord.child_match_code;
                        let mut bindings = left_ord.bindings.clone();
                        bindings.extend(right_ord.bindings.clone());
                        Ordering {
                            child_match_code: quote! {
                                #left_code
                                #left_child
                                #right_code
                                #right_child
                            },
                            bindings,
                        }
                    })
                })
                .collect();

            Ok(InlineMatchOutput { match_code, orderings })
        }

        OpClass::Unary => {
            if args.len() != 1 {
                return Err(Error::new_spanned(op, format!("{} requires exactly 1 argument", op_name)));
            }

            let unary_op = format_ident!("{}", op_name);
            // Use unique names based on tree_var to avoid shadowing
            let src_var = format_ident!("{}_src", tree_var);

            let match_code = quote! {
                let morok_ir::Op::Unary(morok_ir::UnaryOp::#unary_op, #src_var) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
            };

            let src_match = generate_inline_match(&args[0], &src_var, iter_ctx, dup_tracker)?;
            let src_code = &src_match.match_code;

            // Propagate all orderings from child, prepending its match_code
            let orderings = src_match
                .orderings
                .into_iter()
                .map(|ord| {
                    let ord_child = &ord.child_match_code;
                    Ordering {
                        child_match_code: quote! {
                            #src_code
                            #ord_child
                        },
                        bindings: ord.bindings,
                    }
                })
                .collect();

            Ok(InlineMatchOutput { match_code, orderings })
        }

        OpClass::Ternary => {
            if args.len() != 3 {
                return Err(Error::new_spanned(op, format!("{} requires exactly 3 arguments", op_name)));
            }

            let ternary_op = format_ident!("{}", op_name);
            // Use unique names based on tree_var to avoid shadowing
            let a_var = format_ident!("{}_a", tree_var);
            let b_var = format_ident!("{}_b", tree_var);
            let c_var = format_ident!("{}_c", tree_var);

            let match_code = quote! {
                let morok_ir::Op::Ternary(morok_ir::TernaryOp::#ternary_op, #a_var, #b_var, #c_var) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
            };

            let a_match = generate_inline_match(&args[0], &a_var, iter_ctx, dup_tracker)?;
            let b_match = generate_inline_match(&args[1], &b_var, iter_ctx, dup_tracker)?;
            let c_match = generate_inline_match(&args[2], &c_var, iter_ctx, dup_tracker)?;

            let a_code = &a_match.match_code;
            let b_code = &b_match.match_code;
            let c_code = &c_match.match_code;

            // Cross-product all three children's orderings using explicit loops
            let mut orderings = Vec::new();
            for a_ord in &a_match.orderings {
                let a_child = &a_ord.child_match_code;
                for b_ord in &b_match.orderings {
                    let b_child = &b_ord.child_match_code;
                    for c_ord in &c_match.orderings {
                        let c_child = &c_ord.child_match_code;
                        let mut bindings = a_ord.bindings.clone();
                        bindings.extend(b_ord.bindings.clone());
                        bindings.extend(c_ord.bindings.clone());
                        orderings.push(Ordering {
                            child_match_code: quote! {
                                #a_code
                                #a_child
                                #b_code
                                #b_child
                                #c_code
                                #c_child
                            },
                            bindings,
                        });
                    }
                }
            }

            Ok(InlineMatchOutput { match_code, orderings })
        }

        OpClass::SingleSource => generate_inline_single_source_match(&op_name, args, tree_var, iter_ctx, dup_tracker),

        OpClass::Special => generate_inline_special_op_match(op, &op_name, args, tree_var, iter_ctx, dup_tracker),
    }
}

/// Generate inline match for single-source ops (Cast, Reshape, etc.)
fn generate_inline_single_source_match(
    op_name: &str,
    args: &[Pattern],
    tree_var: &Ident,
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    if args.len() != 1 {
        return Err(Error::new(proc_macro2::Span::call_site(), format!("{} requires exactly 1 argument", op_name)));
    }

    let op_ident = format_ident!("{}", op_name);
    // Use unique names based on tree_var to avoid shadowing
    let src_var = format_ident!("{}_src", tree_var);

    let match_code = quote! {
        let morok_ir::Op::#op_ident { src: #src_var, .. } = #tree_var.op() else {
            return morok_ir::pattern::RewriteResult::NoMatch;
        };
    };

    let src_match = generate_inline_match(&args[0], &src_var, iter_ctx, dup_tracker)?;
    let src_code = &src_match.match_code;

    // Propagate all orderings from child
    let orderings = src_match
        .orderings
        .into_iter()
        .map(|ord| {
            let ord_child = &ord.child_match_code;
            Ordering {
                child_match_code: quote! {
                    #src_code
                    #ord_child
                },
                bindings: ord.bindings,
            }
        })
        .collect();

    Ok(InlineMatchOutput { match_code, orderings })
}

/// Generate inline match for special ops using field metadata from OP_CHILD_FIELDS.
///
/// This uses the generic field mapping to destructure ops like Store, Load, Range, etc.
/// The field names are looked up from `get_child_field_names()` and used to generate
/// appropriate match patterns.
fn generate_inline_special_op_match(
    op: &Ident,
    op_name: &str,
    args: &[Pattern],
    tree_var: &Ident,
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    let op_ident = format_ident!("{}", op_name);

    // Handle zero-argument case: match any op of this type
    if args.is_empty() {
        let match_code = quote! {
            let morok_ir::Op::#op_ident { .. } = #tree_var.op() else {
                return morok_ir::pattern::RewriteResult::NoMatch;
            };
        };
        return Ok(InlineMatchOutput::simple(match_code, vec![]));
    }

    // Look up field names for this op
    let field_names = get_child_field_names(op_name).ok_or_else(|| {
        Error::new_spanned(
            op,
            format!(
                "Op '{}' not found in OP_CHILD_FIELDS. Add it to the map or use struct syntax `{} {{ field: pattern }}`",
                op_name, op_name
            ),
        )
    })?;

    // Validate argument count
    if args.len() > field_names.len() {
        return Err(Error::new_spanned(
            op,
            format!(
                "{} has {} child fields ({:?}), but {} arguments were provided",
                op_name,
                field_names.len(),
                field_names,
                args.len()
            ),
        ));
    }

    // Generate temp variables for each field (unique based on tree_var)
    let field_vars: Vec<Ident> =
        field_names.iter().take(args.len()).map(|name| format_ident!("{}_{}", tree_var, name)).collect();

    // Generate field bindings for the match pattern
    let field_bindings: Vec<TokenStream2> = field_names
        .iter()
        .take(args.len())
        .zip(&field_vars)
        .map(|(name, var)| {
            let field_ident = format_ident!("{}", name);
            quote! { #field_ident: #var }
        })
        .collect();

    // Generate the match code
    let match_code = quote! {
        let morok_ir::Op::#op_ident { #(#field_bindings,)* .. } = #tree_var.op() else {
            return morok_ir::pattern::RewriteResult::NoMatch;
        };
    };

    // Recursively match each child pattern and collect their orderings
    let mut child_matches: Vec<(TokenStream2, InlineMatchOutput)> = Vec::new();
    for (arg, var) in args.iter().zip(&field_vars) {
        let child_match = generate_inline_match(arg, var, iter_ctx, dup_tracker)?;
        child_matches.push((child_match.match_code.clone(), child_match));
    }

    // Compute cross-product of all children's orderings
    let orderings = compute_ordering_cross_product(&child_matches);

    Ok(InlineMatchOutput { match_code, orderings })
}

/// Compute cross-product of orderings from multiple children.
/// Each child has a match_code and N orderings. We produce all combinations.
fn compute_ordering_cross_product(children: &[(TokenStream2, InlineMatchOutput)]) -> Vec<Ordering> {
    if children.is_empty() {
        return vec![Ordering { child_match_code: quote! {}, bindings: vec![] }];
    }

    // Start with orderings from first child
    let (first_code, first_match) = &children[0];
    let mut result: Vec<Ordering> = first_match
        .orderings
        .iter()
        .map(|ord| {
            let child_code = &ord.child_match_code;
            Ordering {
                child_match_code: quote! {
                    #first_code
                    #child_code
                },
                bindings: ord.bindings.clone(),
            }
        })
        .collect();

    // Cross with each subsequent child
    for (child_code, child_match) in children.iter().skip(1) {
        let mut new_result = Vec::new();
        for existing in &result {
            for child_ord in &child_match.orderings {
                let existing_code = &existing.child_match_code;
                let child_child = &child_ord.child_match_code;
                let mut bindings = existing.bindings.clone();
                bindings.extend(child_ord.bindings.clone());
                new_result.push(Ordering {
                    child_match_code: quote! {
                        #existing_code
                        #child_code
                        #child_child
                    },
                    bindings,
                });
            }
        }
        result = new_result;
    }

    result
}

/// Generate inline match for struct-style ops like Bufferize { compute: x, .. }
fn generate_inline_op_struct_match(
    op: &Ident,
    fields: &[FieldPattern],
    tree_var: &Ident,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    let op_name = op.to_string();
    let op_ident = format_ident!("{}", op_name);

    if fields.is_empty() {
        return Err(Error::new_spanned(op, "Struct pattern must have at least one field"));
    }

    // Generate field extraction for all fields at once
    let field_vars: Vec<Ident> = fields.iter().map(|f| format_ident!("{}_{}", tree_var, f.name)).collect();
    let field_bindings: Vec<TokenStream2> = fields
        .iter()
        .zip(&field_vars)
        .map(|(f, var)| {
            let name = &f.name;
            quote! { #name: #var }
        })
        .collect();

    let match_code = quote! {
        let morok_ir::Op::#op_ident { #(#field_bindings,)* .. } = #tree_var.op() else {
            return morok_ir::pattern::RewriteResult::NoMatch;
        };
    };

    // Collect matches for all fields
    let mut child_matches: Vec<(TokenStream2, InlineMatchOutput)> = Vec::new();
    for (field, var) in fields.iter().zip(&field_vars) {
        let field_match = generate_inline_match(&field.pattern, var, None, dup_tracker)?;
        child_matches.push((field_match.match_code.clone(), field_match));
    }

    // Compute cross-product of all fields' orderings
    let orderings = compute_ordering_cross_product(&child_matches);

    Ok(InlineMatchOutput { match_code, orderings })
}

/// Generate inline match for constant patterns.
fn generate_inline_const_match(const_pat: &ConstPattern, tree_var: &Ident) -> Result<InlineMatchOutput> {
    match const_pat {
        ConstPattern::Any => {
            let match_code = quote! {
                let morok_ir::Op::Const(_) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
            };
            let const_ident = Ident::new(binding_names::CONST, proc_macro2::Span::call_site());
            Ok(InlineMatchOutput::simple(match_code, vec![(const_ident, quote! { #tree_var })]))
        }

        ConstPattern::Zero | ConstPattern::Int(0) => {
            let match_code = quote! {
                if !morok_ir::pattern::helpers::is_zero(#tree_var) {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                }
            };
            let zero_ident = Ident::new(binding_names::ZERO, proc_macro2::Span::call_site());
            Ok(InlineMatchOutput::simple(match_code, vec![(zero_ident, quote! { #tree_var })]))
        }

        ConstPattern::One => {
            let match_code = quote! {
                if !morok_ir::pattern::helpers::is_one(#tree_var) {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                }
            };
            let one_ident = Ident::new(binding_names::ONE, proc_macro2::Span::call_site());
            Ok(InlineMatchOutput::simple(match_code, vec![(one_ident, quote! { #tree_var })]))
        }

        ConstPattern::Int(value) => {
            let match_code = quote! {
                let morok_ir::Op::Const(cv) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
                if cv.0.try_int().map(|v| v == #value).unwrap_or(false) == false {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                }
            };
            Ok(InlineMatchOutput::simple(match_code, vec![]))
        }

        ConstPattern::Float(value) => {
            let match_code = quote! {
                let morok_ir::Op::Const(cv) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
                if cv.0.try_float().map(|v| (v - #value).abs() < 1e-10).unwrap_or(false) == false {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                }
            };
            Ok(InlineMatchOutput::simple(match_code, vec![]))
        }
    }
}

/// Generate inline match for op variable (in for-loop context).
fn generate_inline_op_var_match(
    var_name: &Ident,
    args: &[Pattern],
    tree_var: &Ident,
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    let ctx = iter_ctx.ok_or_else(|| Error::new_spanned(var_name, "Operation variable used outside of for-block"))?;

    if var_name != &ctx.var_name {
        return Err(Error::new_spanned(
            var_name,
            format!("Unknown operation variable '{}', expected '{}'", var_name, ctx.var_name),
        ));
    }

    let op_ident = &ctx.op_ident;

    match ctx.op_kind {
        OpKind::Unary => {
            if args.len() != 1 {
                return Err(Error::new_spanned(var_name, "Unary op variable requires 1 argument"));
            }
            let src_var = format_ident!("__src");
            let match_code = quote! {
                let morok_ir::Op::Unary(morok_ir::UnaryOp::#op_ident, #src_var) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
            };

            let src_match = generate_inline_match(&args[0], &src_var, iter_ctx, dup_tracker)?;
            let src_code = &src_match.match_code;

            // Propagate all orderings from child
            let orderings = src_match
                .orderings
                .into_iter()
                .map(|ord| {
                    let ord_child = &ord.child_match_code;
                    Ordering {
                        child_match_code: quote! {
                            #src_code
                            #ord_child
                        },
                        bindings: ord.bindings,
                    }
                })
                .collect();

            Ok(InlineMatchOutput { match_code, orderings })
        }

        OpKind::Binary => {
            if args.len() != 2 {
                return Err(Error::new_spanned(var_name, "Binary op variable requires 2 arguments"));
            }
            let left_var = format_ident!("__left");
            let right_var = format_ident!("__right");
            let match_code = quote! {
                let morok_ir::Op::Binary(morok_ir::BinaryOp::#op_ident, #left_var, #right_var) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
            };

            let left_match = generate_inline_match(&args[0], &left_var, iter_ctx, dup_tracker)?;
            let right_match = generate_inline_match(&args[1], &right_var, iter_ctx, dup_tracker)?;

            let left_code = &left_match.match_code;
            let right_code = &right_match.match_code;

            // Cross-product orderings from children
            let orderings: Vec<Ordering> = left_match
                .orderings
                .iter()
                .flat_map(|left_ord| {
                    let left_child = &left_ord.child_match_code;
                    right_match.orderings.iter().map(move |right_ord| {
                        let right_child = &right_ord.child_match_code;
                        let mut bindings = left_ord.bindings.clone();
                        bindings.extend(right_ord.bindings.clone());
                        Ordering {
                            child_match_code: quote! {
                                #left_code
                                #left_child
                                #right_code
                                #right_child
                            },
                            bindings,
                        }
                    })
                })
                .collect();

            Ok(InlineMatchOutput { match_code, orderings })
        }

        OpKind::Ternary => {
            if args.len() != 3 {
                return Err(Error::new_spanned(var_name, "Ternary op variable requires 3 arguments"));
            }
            let a_var = format_ident!("__a");
            let b_var = format_ident!("__b");
            let c_var = format_ident!("__c");
            let match_code = quote! {
                let morok_ir::Op::Ternary(morok_ir::TernaryOp::#op_ident, #a_var, #b_var, #c_var) = #tree_var.op() else {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                };
            };

            let a_match = generate_inline_match(&args[0], &a_var, iter_ctx, dup_tracker)?;
            let b_match = generate_inline_match(&args[1], &b_var, iter_ctx, dup_tracker)?;
            let c_match = generate_inline_match(&args[2], &c_var, iter_ctx, dup_tracker)?;

            let a_code = &a_match.match_code;
            let b_code = &b_match.match_code;
            let c_code = &c_match.match_code;

            // Cross-product of all three children's orderings using explicit loops
            let mut orderings = Vec::new();
            for a_ord in &a_match.orderings {
                let a_child = &a_ord.child_match_code;
                for b_ord in &b_match.orderings {
                    let b_child = &b_ord.child_match_code;
                    for c_ord in &c_match.orderings {
                        let c_child = &c_ord.child_match_code;
                        let mut bindings = a_ord.bindings.clone();
                        bindings.extend(b_ord.bindings.clone());
                        bindings.extend(c_ord.bindings.clone());
                        orderings.push(Ordering {
                            child_match_code: quote! {
                                #a_code
                                #a_child
                                #b_code
                                #b_child
                                #c_code
                                #c_child
                            },
                            bindings,
                        });
                    }
                }
            }

            Ok(InlineMatchOutput { match_code, orderings })
        }
    }
}

/// Generate inline match for alternative patterns (A | B | C).
///
/// Supports two cases:
/// 1. Simple identifier alternatives in struct fields (e.g., `reduce_op: Add | Mul`)
///    - All alternatives must be `Pattern::Var` (identifiers)
///    - Generates `matches!(field, Enum::A | Enum::B)` check
///    - Enum type is inferred from the tree_var name (e.g., `__tree_reduce_op` → `ReduceOp`)
///    - Binds the field value as a closure parameter (e.g., `reduce_op`)
///
/// 2. Top-level alternatives are handled by `generate_simplified_alternatives_rule`
fn generate_inline_alternatives_match(
    alternatives: &[Pattern],
    tree_var: &Ident,
    _iter_ctx: Option<&IterContext>,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    // Check if all alternatives are simple Var patterns (enum discriminants)
    let variant_names: Vec<&Ident> = alternatives
        .iter()
        .map(|p| match p {
            Pattern::Var(ident) => Ok(ident),
            _ => Err(Error::new(
                proc_macro2::Span::call_site(),
                "Nested alternatives must be simple identifiers (e.g., Add | Mul). Complex patterns in alternatives are not yet supported.",
            )),
        })
        .collect::<Result<Vec<_>>>()?;

    // Infer enum type and field name from tree_var name (e.g., `__tree_reduce_op` → `ReduceOp`, `reduce_op`)
    // Convention: field names like `reduce_op`, `unary_op`, `binary_op` map to their enum types
    let tree_var_str = tree_var.to_string();
    let (enum_type, field_name) = if tree_var_str.ends_with("_reduce_op") || tree_var_str == "reduce_op" {
        (quote! { morok_ir::ReduceOp }, "reduce_op")
    } else if tree_var_str.ends_with("_unary_op") || tree_var_str == "unary_op" {
        (quote! { morok_ir::UnaryOp }, "unary_op")
    } else if tree_var_str.ends_with("_binary_op") || tree_var_str == "binary_op" {
        (quote! { morok_ir::BinaryOp }, "binary_op")
    } else if tree_var_str.ends_with("_ternary_op") || tree_var_str == "ternary_op" {
        (quote! { morok_ir::TernaryOp }, "ternary_op")
    } else {
        return Err(Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "Cannot infer enum type for field '{}'. Nested alternatives are only supported for known discriminant fields (reduce_op, unary_op, binary_op, ternary_op).",
                tree_var_str
            ),
        ));
    };

    // Generate matches! check: matches!(field_var, Enum::A | Enum::B | ...)
    let variant_patterns: Vec<TokenStream2> = variant_names.iter().map(|name| quote! { #enum_type::#name }).collect();

    let match_code = quote! {
        if !matches!(#tree_var, #(#variant_patterns)|*) {
            return morok_ir::pattern::RewriteResult::NoMatch;
        }
    };

    // Create binding for the field (e.g., reduce_op → &ReduceOp)
    let actual_name = dup_tracker.process_name(field_name);
    let binding_ident = Ident::new(&actual_name, proc_macro2::Span::call_site());

    Ok(InlineMatchOutput::simple(match_code, vec![(binding_ident, quote! { #tree_var })]))
}

/// Generate inline match for commutative patterns [x, y].
///
/// For commutative patterns like `Add[Mul[a, b], c]`, we generate ALL possible orderings:
/// - Outer Add: 2 orderings (left, right) and (right, left)
/// - Inner Mul: 2 orderings (a, b) and (b, a)
/// - Total: 2 × 2 = 4 orderings to try
///
/// This is computed as a cross-product of the outer ordering with all children's orderings.
fn generate_inline_commutative_match(
    op: &Ident,
    args: &[Pattern],
    tree_var: &Ident,
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut DuplicateTracker,
) -> Result<InlineMatchOutput> {
    let op_name = op.to_string();

    if args.len() != 2 {
        return Err(Error::new_spanned(op, "Commutative pattern requires exactly 2 arguments"));
    }

    if !BINARY_OPS.contains(&op_name.as_str()) {
        return Err(Error::new_spanned(op, format!("Commutative pattern requires binary op, got: {}", op_name)));
    }

    let binary_op = format_ident!("{}", op_name);
    // Use unique names based on tree_var to avoid shadowing in nested commutative patterns
    let left_var = format_ident!("{}_left", tree_var);
    let right_var = format_ident!("{}_right", tree_var);

    // Generate ONLY the outer op match (shared across all orderings)
    let match_code = quote! {
        let morok_ir::Op::Binary(morok_ir::BinaryOp::#binary_op, #left_var, #right_var) = #tree_var.op() else {
            return morok_ir::pattern::RewriteResult::NoMatch;
        };
    };

    // Clone the tracker BEFORE processing either ordering.
    // Both orderings need independent duplicate detection, and both should start
    // from the same initial state (variables seen by parent patterns).
    let mut dup_tracker_swapped = dup_tracker.clone();

    // Generate matches for normal ordering: args[0] matches left, args[1] matches right
    let left_match = generate_inline_match(&args[0], &left_var, iter_ctx, dup_tracker)?;
    let right_match = generate_inline_match(&args[1], &right_var, iter_ctx, dup_tracker)?;

    let left_code = &left_match.match_code;
    let right_code = &right_match.match_code;

    // Cross-product all orderings for normal outer ordering: left × right
    let normal_orderings: Vec<Ordering> = left_match
        .orderings
        .iter()
        .flat_map(|left_ord| {
            let left_child = &left_ord.child_match_code;
            right_match.orderings.iter().map(move |right_ord| {
                let right_child = &right_ord.child_match_code;
                let mut bindings = left_ord.bindings.clone();
                bindings.extend(right_ord.bindings.clone());
                Ordering {
                    child_match_code: quote! {
                        #left_code
                        #left_child
                        #right_code
                        #right_child
                    },
                    bindings,
                }
            })
        })
        .collect();

    // Generate matches for swapped ordering: args[0] matches right, args[1] matches left
    let left_match_swap = generate_inline_match(&args[0], &right_var, iter_ctx, &mut dup_tracker_swapped)?;
    let right_match_swap = generate_inline_match(&args[1], &left_var, iter_ctx, &mut dup_tracker_swapped)?;

    let left_code_swap = &left_match_swap.match_code;
    let right_code_swap = &right_match_swap.match_code;

    // Cross-product all orderings for swapped outer ordering: left_swap × right_swap
    let swapped_orderings: Vec<Ordering> = left_match_swap
        .orderings
        .iter()
        .flat_map(|left_ord| {
            let left_child = &left_ord.child_match_code;
            right_match_swap.orderings.iter().map(move |right_ord| {
                let right_child = &right_ord.child_match_code;
                let mut bindings = left_ord.bindings.clone();
                bindings.extend(right_ord.bindings.clone());
                Ordering {
                    child_match_code: quote! {
                        #left_code_swap
                        #left_child
                        #right_code_swap
                        #right_child
                    },
                    bindings,
                }
            })
        })
        .collect();

    // Combine all orderings: normal + swapped
    let mut all_orderings = normal_orderings;
    all_orderings.extend(swapped_orderings);

    Ok(InlineMatchOutput { match_code, orderings: all_orderings })
}

// =============================================================================
// Simplified Rule Generation
// =============================================================================

/// Generate a rule for alternative patterns (A | B | C) at the top level.
///
/// For each alternative, we generate a separate attempt block. The first
/// alternative that matches will execute and return the rewrite result.
fn generate_simplified_alternatives_rule(
    alternatives: &[Pattern],
    rule: &PatternRule,
    iter_ctx: Option<&IterContext>,
    has_context: bool,
) -> Result<TokenStream2> {
    let tree_var = format_ident!("__tree");

    // Collect OpKeys from all alternatives
    let op_keys = compute_op_keys(&rule.lhs, iter_ctx);

    // Generate attempt blocks for each alternative
    // Each alternative is wrapped in an inner closure so that `return NoMatch`
    // only returns from that alternative's attempt, not the entire pattern closure.
    let mut alt_blocks = Vec::new();

    for alt in alternatives {
        let mut dup_tracker = DuplicateTracker::default();
        let alt_match = generate_inline_match(alt, &tree_var, iter_ctx, &mut dup_tracker)?;

        let match_code = &alt_match.match_code;

        // Generate ptr_eq checks for duplicates (return NoMatch from inner closure)
        let duplicate_pairs = dup_tracker.get_duplicates();
        let ptr_eq_checks: Vec<TokenStream2> = duplicate_pairs
            .iter()
            .map(|(orig, dup)| {
                let orig_ident = format_ident!("{}", orig);
                let dup_ident = format_ident!("{}", dup);
                quote! {
                    if !std::sync::Arc::ptr_eq(#orig_ident, #dup_ident) {
                        return morok_ir::pattern::RewriteResult::NoMatch;
                    }
                }
            })
            .collect();

        // Generate rewrite expression (same for all alternatives)
        let rewrite_expr = generate_simplified_rewrite_expr(&rule.rhs, &rule.guard, rule.arrow);

        // For each alternative, try all its orderings (handles nested commutative)
        for ord in &alt_match.orderings {
            let child_code = &ord.child_match_code;
            let binding_stmts: Vec<TokenStream2> =
                ord.bindings.iter().map(|(name, expr)| quote! { let #name = #expr; }).collect();

            // Wrap alternative in a closure so `return NoMatch` only exits this attempt
            alt_blocks.push(quote! {
                {
                    let __try_result = (|| {
                        #match_code
                        #child_code
                        #(#binding_stmts)*
                        #(#ptr_eq_checks)*
                        #rewrite_expr
                    })();
                    if !matches!(__try_result, morok_ir::pattern::RewriteResult::NoMatch) {
                        return __try_result;
                    }
                }
            });
        }
    }

    // Generate context parameter
    let ctx_param = if has_context {
        quote! { ctx: &mut _ }
    } else {
        quote! { _ctx: &mut () }
    };

    if op_keys.is_empty() {
        Ok(quote! {
            __matcher.add_wildcard(
                |#tree_var: &std::sync::Arc<morok_ir::UOp>, #ctx_param| {
                    #(#alt_blocks)*
                    morok_ir::pattern::RewriteResult::NoMatch
                }
            );
        })
    } else {
        Ok(quote! {
            __matcher.add(
                &[#(#op_keys),*],
                |#tree_var: &std::sync::Arc<morok_ir::UOp>, #ctx_param| {
                    #(#alt_blocks)*
                    morok_ir::pattern::RewriteResult::NoMatch
                }
            );
        })
    }
}

/// Generate a rule for SimplifiedPatternMatcher with inline matching.
fn generate_simplified_rule(
    rule: &PatternRule,
    iter_ctx: Option<&IterContext>,
    has_context: bool,
) -> Result<TokenStream2> {
    // Check for alternative patterns at the top level - handle specially
    if let Pattern::Any(alternatives) = &rule.lhs {
        return generate_simplified_alternatives_rule(alternatives, rule, iter_ctx, has_context);
    }

    // Compute OpKeys for dispatch
    let op_keys = compute_op_keys(&rule.lhs, iter_ctx);
    let tree_var = format_ident!("__tree");

    // Generate inline match
    let mut dup_tracker = DuplicateTracker::default();
    let match_output = generate_inline_match(&rule.lhs, &tree_var, iter_ctx, &mut dup_tracker)?;

    // Get duplicate pairs for ptr_eq generation
    let duplicate_pairs: Vec<(String, String)> = dup_tracker.get_duplicates().to_vec();

    // Generate ptr_eq checks for duplicates
    let ptr_eq_checks: Vec<TokenStream2> = duplicate_pairs
        .iter()
        .map(|(orig, dup)| {
            let orig_ident = format_ident!("{}", orig);
            let dup_ident = format_ident!("{}", dup);
            quote! {
                if !std::sync::Arc::ptr_eq(#orig_ident, #dup_ident) {
                    return morok_ir::pattern::RewriteResult::NoMatch;
                }
            }
        })
        .collect();

    // Add operation variable binding if in iteration context
    let op_var_binding = if let Some(ctx) = iter_ctx {
        let var_name = &ctx.var_name;
        let op_ident = &ctx.op_ident;
        match ctx.op_kind {
            OpKind::Unary => Some(quote! { let #var_name = morok_ir::UnaryOp::#op_ident; }),
            OpKind::Binary => Some(quote! { let #var_name = morok_ir::BinaryOp::#op_ident; }),
            OpKind::Ternary => Some(quote! { let #var_name = morok_ir::TernaryOp::#op_ident; }),
        }
    } else {
        None
    };

    // Validate closure params if RHS is a closure (use bindings from first ordering)
    if let RewriteExpr::Closure(closure) = &rule.rhs {
        let analysis = extract_closure_params(closure)?;
        let var_names: Vec<Ident> = match_output.bindings().iter().map(|(n, _)| n.clone()).collect();
        validate_closure_params(&analysis, &var_names, has_context)?;
    }

    // Generate rewrite expression
    let rewrite_expr = generate_simplified_rewrite_expr(&rule.rhs, &rule.guard, rule.arrow);

    let match_code = &match_output.match_code;

    // Handle patterns with multiple orderings (commutative patterns)
    let body = if match_output.orderings.len() > 1 {
        // Multiple orderings - wrap each in closure, try in sequence
        let num_orderings = match_output.orderings.len();
        let try_blocks: Vec<TokenStream2> = match_output
            .orderings
            .iter()
            .enumerate()
            .map(|(i, ord)| {
                let binding_stmts: Vec<TokenStream2> =
                    ord.bindings.iter().map(|(name, expr)| quote! { let #name = #expr; }).collect();
                let child_match = &ord.child_match_code;
                let is_last = i == num_orderings - 1;

                if is_last {
                    // Last ordering: no closure wrapper
                    quote! {
                        #child_match
                        #(#binding_stmts)*
                        #(#ptr_eq_checks)*
                        #rewrite_expr
                    }
                } else {
                    // Non-last: wrap in closure so return NoMatch only exits this attempt
                    quote! {
                        let __result = (|| {
                            #child_match
                            #(#binding_stmts)*
                            #(#ptr_eq_checks)*
                            #rewrite_expr
                        })();
                        if !matches!(__result, morok_ir::pattern::RewriteResult::NoMatch) {
                            return __result;
                        }
                    }
                }
            })
            .collect();

        quote! {
            #match_code
            #op_var_binding
            #(#try_blocks)*
        }
    } else {
        // Single ordering - simple case
        let ord = &match_output.orderings[0];
        let binding_stmts: Vec<TokenStream2> =
            ord.bindings.iter().map(|(name, expr)| quote! { let #name = #expr; }).collect();
        let child_match = &ord.child_match_code;
        quote! {
            #match_code
            #op_var_binding
            #child_match
            #(#binding_stmts)*
            #(#ptr_eq_checks)*
            #rewrite_expr
        }
    };

    // Generate the closure with appropriate context parameter
    let ctx_param = if has_context {
        quote! { ctx: &mut _ }
    } else {
        quote! { _ctx: &mut () }
    };

    if op_keys.is_empty() {
        // Wildcard pattern
        Ok(quote! {
            __matcher.add_wildcard(
                move |#tree_var: &std::sync::Arc<morok_ir::UOp>, #ctx_param| {
                    #body
                }
            );
        })
    } else {
        Ok(quote! {
            __matcher.add(
                &[#(#op_keys),*],
                move |#tree_var: &std::sync::Arc<morok_ir::UOp>, #ctx_param| {
                    #body
                }
            );
        })
    }
}

/// Generate rewrite expression for simplified matcher.
fn generate_simplified_rewrite_expr(rhs: &RewriteExpr, guard: &Option<syn::Expr>, arrow: ArrowKind) -> TokenStream2 {
    let rhs_expr = match rhs {
        RewriteExpr::Var(name) => quote! { std::sync::Arc::clone(#name) },
        RewriteExpr::Expr(expr) => quote! { #expr },
        RewriteExpr::Block(block) => quote! { #block },
        RewriteExpr::Closure(closure) => {
            let body = &closure.body;
            quote! { #body }
        }
    };

    match arrow {
        ArrowKind::Infallible => {
            if let Some(guard_expr) = guard {
                quote! {
                    if #guard_expr {
                        morok_ir::pattern::RewriteResult::Rewritten(#rhs_expr)
                    } else {
                        morok_ir::pattern::RewriteResult::NoMatch
                    }
                }
            } else {
                quote! {
                    morok_ir::pattern::RewriteResult::Rewritten(#rhs_expr)
                }
            }
        }
        ArrowKind::Fallible => {
            let wrapped = match rhs {
                RewriteExpr::Var(name) => quote! { Some(std::sync::Arc::clone(#name)) },
                RewriteExpr::Expr(expr) => quote! { (|| #expr)() },
                RewriteExpr::Block(block) => quote! { (|| #block)() },
                RewriteExpr::Closure(closure) => {
                    let body = &closure.body;
                    quote! { (|| #body)() }
                }
            };

            let conversion = quote! {
                match #wrapped {
                    Some(__v) => morok_ir::pattern::RewriteResult::Rewritten(__v),
                    None => morok_ir::pattern::RewriteResult::NoMatch,
                }
            };

            if let Some(guard_expr) = guard {
                quote! {
                    if #guard_expr {
                        #conversion
                    } else {
                        morok_ir::pattern::RewriteResult::NoMatch
                    }
                }
            } else {
                conversion
            }
        }
    }
}

/// Generate a SimplifiedPatternMatcher from the parsed pattern list.
pub fn generate_simplified_pattern_matcher(patterns: &PatternList) -> Result<TokenStream2> {
    let mut pattern_exprs = Vec::new();
    let has_context = patterns.context_type.is_some();

    // Collect all op names for compile-time validation
    let op_names = collect_op_names(patterns);
    let validation_code = generate_op_validation(&op_names);

    for item in &patterns.items {
        match item {
            PatternItem::Rule(rule) => {
                pattern_exprs.push(generate_simplified_rule(rule, None, has_context)?);
            }
            PatternItem::ForBlock(for_block) => {
                let expanded = expand_simplified_for_block(for_block, has_context)?;
                pattern_exprs.extend(expanded);
            }
        }
    }

    // Generate code based on whether context type is declared
    if let Some(ref ctx_type) = patterns.context_type {
        Ok(quote! {
            {
                #validation_code
                let mut __matcher = morok_ir::pattern::SimplifiedPatternMatcher::<#ctx_type>::new();
                #(#pattern_exprs)*
                __matcher
            }
        })
    } else {
        Ok(quote! {
            {
                #validation_code
                let mut __matcher = morok_ir::pattern::SimplifiedPatternMatcher::<()>::new();
                #(#pattern_exprs)*
                __matcher
            }
        })
    }
}

/// Expand a for-block for simplified matcher.
fn expand_simplified_for_block(for_block: &ForBlock, has_context: bool) -> Result<Vec<TokenStream2>> {
    let var_name = &for_block.var;
    let mut results = Vec::new();

    // Resolve ops and kind - handle both explicit lists and wildcards
    let (ops, kind): (Vec<Ident>, OpKind) = match &for_block.iter_kind {
        IterKind::Unary(ops) => (ops.clone(), OpKind::Unary),
        IterKind::Binary(ops) => (ops.clone(), OpKind::Binary),
        IterKind::Ternary(ops) => (ops.clone(), OpKind::Ternary),
        // Expand wildcards to full op lists
        IterKind::UnaryAll => {
            let ops = UNARY_OPS.iter().map(|s| format_ident!("{}", s)).collect();
            (ops, OpKind::Unary)
        }
        IterKind::BinaryAll => {
            let ops = BINARY_OPS.iter().map(|s| format_ident!("{}", s)).collect();
            (ops, OpKind::Binary)
        }
        IterKind::TernaryAll => {
            let ops = TERNARY_OPS.iter().map(|s| format_ident!("{}", s)).collect();
            (ops, OpKind::Ternary)
        }
    };

    for op_ident in &ops {
        for rule in &for_block.body {
            let ctx = IterContext { var_name: var_name.clone(), op_ident: op_ident.clone(), op_kind: kind };
            results.push(generate_simplified_rule(rule, Some(&ctx), has_context)?);
        }
    }

    Ok(results)
}
