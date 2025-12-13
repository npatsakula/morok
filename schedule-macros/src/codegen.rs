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
    /// Prefix for nested struct bindings (e.g., __nested_0, __nested_1)
    pub const NESTED_PREFIX: &str = "__nested_";
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

/// Tracks duplicate variable names for auto ptr_eq generation.
///
/// When a pattern like `Add(x, x)` is used, this tracker:
/// 1. Records the first `x` normally
/// 2. Renames the second `x` to `x__dup` and records the pair
///
/// The rewrite function then generates `Arc::ptr_eq(&x, &x__dup)` checks.
#[derive(Default)]
struct DuplicateTracker {
    /// Variable names we've seen so far
    seen: std::collections::HashSet<String>,
    /// Pairs of (original_name, duplicate_name) for ptr_eq generation
    duplicates: Vec<(String, String)>,
}

impl DuplicateTracker {
    /// Process a variable name, returning the name to use in the pattern.
    /// If this is a duplicate, returns a renamed version and records the pair.
    fn process_name(&mut self, name: &str) -> String {
        if self.seen.contains(name) {
            // This is a duplicate - create a unique name
            let dup_name = format!("{}_dup", name);
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

/// Tracks nested struct patterns for field extraction.
///
/// When we have nested patterns like `Index { buffer: Bufferize { compute, ranges, .. }, indices }`,
/// we need to bind both the outer `Index` and inner `Bufferize` to extract their fields.
struct NestedStructInfo {
    /// Binding name for this struct level (e.g., "__struct_root" or "__nested_0")
    binding_name: String,
    /// The Op name (e.g., "Index", "Bufferize")
    op_name: String,
    /// Fields that need extraction at this level: (field_name, var_name)
    extractable_fields: Vec<(Ident, Ident)>,
}

/// Recursively collect nested struct information for field extraction.
///
/// Walks the pattern tree to find all struct patterns that have extractable fields,
/// assigning unique binding names to each level (__struct_root for depth 0, __nested_N for deeper).
fn collect_nested_struct_info(pattern: &Pattern, depth: usize) -> Vec<NestedStructInfo> {
    let mut result = Vec::new();

    if let Pattern::OpStruct { op, fields, .. } = pattern {
        let op_name = op.to_string();

        // Check if this op supports field extraction
        if !EXTRACTABLE_OPS.contains(&op_name.as_str()) {
            return result;
        }

        // Collect extractable fields at this level (shorthand fields after the first)
        let extractable: Vec<(Ident, Ident)> = fields
            .iter()
            .skip(1)
            .filter_map(|f| {
                if let Pattern::Var(var) = &f.pattern
                    && *var == f.name
                {
                    return Some((f.name.clone(), var.clone()));
                }
                None
            })
            .collect();

        // Add this level if it has extractable fields
        if !extractable.is_empty() {
            let binding_name = if depth == 0 {
                binding_names::STRUCT_ROOT.to_string()
            } else {
                format!("{}{}", binding_names::NESTED_PREFIX, depth - 1)
            };

            result.push(NestedStructInfo { binding_name, op_name: op_name.clone(), extractable_fields: extractable });
        }

        // Recurse into first field (the main UOp child) to find nested structs
        if let Some(first_field) = fields.first() {
            result.extend(collect_nested_struct_info(&first_field.pattern, depth + 1));
        }
    }

    result
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
///
/// When `@context Type;` is declared:
/// - Generates `PatternMatcher<Type>` instead of `PatternMatcher<()>`
/// - Pattern closures receive `ctx: &mut Type`
/// - `ctx` is available in RHS expressions
pub fn generate_pattern_matcher(patterns: &PatternList) -> Result<TokenStream2> {
    let mut pattern_exprs = Vec::new();

    // Determine if we have a context type
    let has_context = patterns.context_type.is_some();

    for item in &patterns.items {
        match item {
            PatternItem::Rule(rule) => {
                pattern_exprs.push(generate_rule(rule, None, has_context)?);
            }
            PatternItem::ForBlock(for_block) => {
                let expanded = expand_for_block(for_block, has_context)?;
                pattern_exprs.extend(expanded);
            }
        }
    }

    // Generate code based on whether context type is declared
    if let Some(ref ctx_type) = patterns.context_type {
        Ok(quote! {
            {
                let mut __patterns: Vec<(
                    morok_ir::pattern::UPat,
                    morok_ir::pattern::matcher::RewriteFn<#ctx_type>
                )> = Vec::new();
                #(#pattern_exprs)*
                morok_ir::pattern::PatternMatcher::<#ctx_type>::new(__patterns)
            }
        })
    } else {
        Ok(quote! {
            {
                let mut __patterns: Vec<(
                    morok_ir::pattern::UPat,
                    morok_ir::pattern::matcher::RewriteFn<()>
                )> = Vec::new();
                #(#pattern_exprs)*
                morok_ir::pattern::PatternMatcher::<()>::new(__patterns)
            }
        })
    }
}

/// Expand a for-block into multiple pattern rules.
fn expand_for_block(for_block: &ForBlock, has_context: bool) -> Result<Vec<TokenStream2>> {
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
            results.push(generate_rule(rule, Some(&ctx), has_context)?);
        }
    }

    Ok(results)
}

/// Generate code for a single pattern rule.
///
/// When `has_context` is true:
/// - Closure receives `ctx: &mut _` (type inferred from PatternMatcher<C>)
/// - `ctx` is available for use in RHS expressions
fn generate_rule(rule: &PatternRule, iter_ctx: Option<&IterContext>, has_context: bool) -> Result<TokenStream2> {
    let mut dup_tracker = DuplicateTracker::default();
    let pattern_output = generate_pattern_with_tracker(&rule.lhs, iter_ctx, &mut Some(&mut dup_tracker))?;
    let var_names = pattern_output.all_names();
    let pattern_code = pattern_output.code;

    // Get duplicate pairs for auto ptr_eq generation
    let duplicate_pairs: Vec<(String, String)> = dup_tracker.get_duplicates().to_vec();

    let (bindings, rewrite_code) =
        generate_rewrite(&rule.lhs, &rule.rhs, &rule.guard, rule.arrow, iter_ctx, &var_names, &duplicate_pairs)?;

    // Generate closure with appropriate context parameter name
    // When has_context is true, use `ctx` so it's available in RHS expressions
    // When has_context is false, use `_ctx` to silence unused warnings
    if has_context {
        Ok(quote! {
            __patterns.push((
                #pattern_code,
                Box::new(|__bindings: &morok_ir::pattern::BindingStore, __intern: &morok_ir::pattern::VarIntern, ctx: &mut _| {
                    use morok_ir::pattern::BindingStoreExt;
                    #(#bindings)*
                    #rewrite_code
                }),
            ));
        })
    } else {
        Ok(quote! {
            __patterns.push((
                #pattern_code,
                Box::new(|__bindings: &morok_ir::pattern::BindingStore, __intern: &morok_ir::pattern::VarIntern, _ctx: &mut ()| {
                    use morok_ir::pattern::BindingStoreExt;
                    #(#bindings)*
                    #rewrite_code
                }) as morok_ir::pattern::matcher::RewriteFn<()>,
            ));
        })
    }
}

/// Generate a UPat expression from a pattern, returning both code and collected names.
/// This is a wrapper that doesn't track duplicates.
fn generate_pattern(pattern: &Pattern, iter_ctx: Option<&IterContext>) -> Result<PatternOutput> {
    generate_pattern_with_tracker(pattern, iter_ctx, &mut None)
}

/// Generate a UPat expression from a pattern with optional duplicate tracking.
///
/// When `dup_tracker` is Some, variable names are tracked for auto ptr_eq generation.
/// Duplicate variable names (like `x` in `Add(x, x)`) are renamed and recorded.
fn generate_pattern_with_tracker(
    pattern: &Pattern,
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut Option<&mut DuplicateTracker>,
) -> Result<PatternOutput> {
    match pattern {
        Pattern::Wildcard => {
            // Wildcard matches any UOp but doesn't bind
            let code = quote! {
                morok_ir::pattern::UPat::Match {
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
            // Process name through duplicate tracker if available
            let actual_name = if let Some(tracker) = dup_tracker {
                tracker.process_name(&name.to_string())
            } else {
                name.to_string()
            };
            let code = quote! { morok_ir::pattern::UPat::var(#actual_name) };
            Ok(PatternOutput::self_only(code, Ident::new(&actual_name, name.span())))
        }

        Pattern::Binding { name, pattern } => {
            let inner = generate_pattern_with_tracker(pattern, iter_ctx, dup_tracker)?;
            // Process name through duplicate tracker if available
            let actual_name = if let Some(tracker) = dup_tracker {
                tracker.process_name(&name.to_string())
            } else {
                name.to_string()
            };
            let inner_code = inner.code;
            let code = quote! { #inner_code.named(#actual_name) };
            // KEY: .named() replaces inner's self_name, keeps inner's children
            Ok(PatternOutput {
                code,
                self_name: Some(Ident::new(&actual_name, name.span())),
                child_names: inner.child_names,
            })
        }

        Pattern::OpTuple { op, args, rest } => {
            generate_op_tuple_pattern_with_tracker(op, args, iter_ctx, *rest, dup_tracker)
        }

        Pattern::OpStruct { op, fields, rest } => generate_op_struct_pattern_with_depth(op, fields, *rest, 0),

        Pattern::Const(const_pat) => generate_const_pattern(const_pat),

        Pattern::OpVar { var_name, args } => {
            generate_op_var_pattern_with_tracker(var_name, args, iter_ctx, dup_tracker)
        }

        Pattern::ConstWithValue { uop_name, .. } => {
            // Process name through duplicate tracker if available
            let actual_name = if let Some(tracker) = dup_tracker {
                tracker.process_name(&uop_name.to_string())
            } else {
                uop_name.to_string()
            };
            let code = quote! { morok_ir::pattern::UPat::cvar(#actual_name) };
            // Only uop_name is bound; value_name is extracted separately in generate_rewrite
            Ok(PatternOutput::self_only(code, Ident::new(&actual_name, uop_name.span())))
        }

        Pattern::Any(alternatives) => {
            // For alternatives, we don't track duplicates ACROSS alternatives
            // because each alternative is independent. The same variable x in
            // (Add(x, y) | Mul(x, y)) should NOT be considered a duplicate.
            // We pass &mut None to disable duplicate tracking within alternatives.
            let alt_outputs: Vec<PatternOutput> = alternatives
                .iter()
                .map(|p| generate_pattern_with_tracker(p, iter_ctx, &mut None))
                .collect::<Result<_>>()?;

            // Collect all names from all alternatives (deduplicated)
            let mut seen = std::collections::HashSet::new();
            let child_names: Vec<Ident> =
                alt_outputs.iter().flat_map(|o| o.all_names()).filter(|name| seen.insert(name.to_string())).collect();

            let alt_codes: Vec<&TokenStream2> = alt_outputs.iter().map(|o| &o.code).collect();
            let code = quote! {
                morok_ir::pattern::UPat::any(vec![#(#alt_codes),*])
            };

            Ok(PatternOutput::children_only(code, child_names))
        }

        Pattern::OpPermute { op, args } => {
            // Generate permutation pattern - tries all orderings of arguments
            let op_name = op.to_string();

            // Collect all child patterns and their names
            let arg_outputs: Vec<PatternOutput> =
                args.iter().map(|a| generate_pattern_with_tracker(a, iter_ctx, dup_tracker)).collect::<Result<_>>()?;

            // Collect all names from children in order
            let child_names: Vec<Ident> = arg_outputs.iter().flat_map(|o| o.all_names()).collect();

            // Extract codes for use in quote!
            let arg_codes: Vec<&TokenStream2> = arg_outputs.iter().map(|o| &o.code).collect();

            // For binary operations, use binary_commutative
            // For other arities, generate all permutations as Any
            let code = if args.len() == 2 {
                // Binary commutative - use specialized helper
                // binary_commutative(ops: Vec<BinaryOp>, src: Vec<UPat>)
                let left = &arg_codes[0];
                let right = &arg_codes[1];

                // Check if it's a known binary op
                if BINARY_OPS.contains(&op_name.as_str()) {
                    let binary_op = format_ident!("{}", op_name);
                    quote! {
                        morok_ir::pattern::UPat::binary_commutative(
                            vec![morok_ir::BinaryOp::#binary_op],
                            vec![#left, #right]
                        )
                    }
                } else {
                    return Err(Error::new_spanned(
                        op,
                        format!("Permutation pattern requires binary op, got: {}", op_name),
                    ));
                }
            } else {
                return Err(Error::new_spanned(
                    op,
                    format!("Permutation pattern currently only supports binary ops (2 args), got {} args", args.len()),
                ));
            };

            Ok(PatternOutput::children_only(code, child_names))
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
const UNARY_OPS: &[&str] =
    &["Neg", "Not", "Abs", "Sqrt", "Exp", "Log", "Sin", "Cos", "Exp2", "Log2", "Tan", "Rsqrt", "Erf"];

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
fn generate_op_tuple_pattern_with_tracker(
    op: &Ident,
    args: &[Pattern],
    iter_ctx: Option<&IterContext>,
    rest: bool,
    dup_tracker: &mut Option<&mut DuplicateTracker>,
) -> Result<PatternOutput> {
    let op_name = op.to_string();

    // Collect all child patterns and their names
    let arg_outputs: Vec<PatternOutput> =
        args.iter().map(|a| generate_pattern_with_tracker(a, iter_ctx, dup_tracker)).collect::<Result<_>>()?;

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
                morok_ir::pattern::UPat::binary(
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
                morok_ir::pattern::UPat::unary(
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
                morok_ir::pattern::UPat::ternary(
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
            quote! { morok_ir::pattern::UPat::#helper_ident(#src) }
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
                Ok(quote! { morok_ir::pattern::UPat::reduce_any() })
            } else {
                if args.len() != 1 {
                    return Err(Error::new_spanned(
                        op,
                        "Reduce requires exactly 1 argument (or use `..` for variable ranges)",
                    ));
                }
                let src = &arg_codes[0];
                Ok(quote! { morok_ir::pattern::UPat::reduce(#src) })
            }
        }

        "End" => {
            if rest {
                Ok(quote! { morok_ir::pattern::UPat::end_any() })
            } else {
                if args.len() != 1 {
                    return Err(Error::new_spanned(
                        op,
                        "End requires exactly 1 argument (or use `..` for variable ranges)",
                    ));
                }
                let computation = &arg_codes[0];
                Ok(quote! { morok_ir::pattern::UPat::end(#computation) })
            }
        }

        "Store" => {
            if args.len() != 3 {
                return Err(Error::new_spanned(op, "Store requires exactly 3 arguments (buffer, index, value)"));
            }
            let buffer = &arg_codes[0];
            let index = &arg_codes[1];
            let value = &arg_codes[2];
            Ok(quote! { morok_ir::pattern::UPat::store(#buffer, #index, #value) })
        }

        "Load" => {
            if args.len() != 2 {
                return Err(Error::new_spanned(op, "Load requires exactly 2 arguments (buffer, index)"));
            }
            let buffer = &arg_codes[0];
            let index = &arg_codes[1];
            Ok(quote! { morok_ir::pattern::UPat::load(#buffer, #index) })
        }

        "Bind" => {
            if args.len() != 2 {
                return Err(Error::new_spanned(op, "Bind requires exactly 2 arguments (var, value)"));
            }
            let var = &arg_codes[0];
            let value = &arg_codes[1];
            Ok(quote! { morok_ir::pattern::UPat::bind(#var, #value) })
        }

        "Const" => Err(Error::new_spanned(op, "Use Const(value) or Const(_) syntax for constants")),

        _ => Err(Error::new_spanned(op, format!("Unknown operation: {}", op_name))),
    }
}

/// Generate pattern for struct-style op: `Bufferize { compute: x, .. }`
///
/// This generates a UPat that matches the op and binds it to a unique name
/// based on nesting depth, allowing field extraction in the rewrite closure.
///
/// - Depth 0: binds to `__struct_root`
/// - Depth N > 0: binds to `__nested_N-1` (e.g., depth 1 -> `__nested_0`)
fn generate_op_struct_pattern_with_depth(
    op: &Ident,
    fields: &[FieldPattern],
    _rest: bool,
    depth: usize,
) -> Result<PatternOutput> {
    let op_name = op.to_string();

    // Find the first UOp child field (the main source pattern)
    let first_field =
        fields.first().ok_or_else(|| Error::new_spanned(op, "Struct pattern must have at least one field"))?;

    // Recursively generate the first field's pattern, passing incremented depth
    // if it's also a struct pattern
    let first_output = generate_pattern_for_nested_struct(&first_field.pattern, depth + 1)?;
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
        "Copy" => quote! { #first_code.f_copy() },
        "Detach" => quote! { morok_ir::pattern::UPat::detach(#first_code) },
        "ContiguousBackward" => quote! { morok_ir::pattern::UPat::contiguous_backward(#first_code) },
        _ => {
            return Err(Error::new_spanned(op, format!("Unknown operation for struct pattern: {}", op_name)));
        }
    };

    // If there are fields beyond the main UOp child that need extraction,
    // bind this op so we can extract them in the rewrite closure
    if has_extractable_fields(fields) {
        // Choose binding name based on depth
        let binding_name = if depth == 0 {
            binding_names::STRUCT_ROOT.to_string()
        } else {
            format!("{}{}", binding_names::NESTED_PREFIX, depth - 1)
        };

        let code = quote! { #base.named(#binding_name) };
        Ok(PatternOutput {
            code,
            self_name: Some(Ident::new(&binding_name, proc_macro2::Span::call_site())),
            child_names,
        })
    } else {
        // No self_name, just child names from the first field
        Ok(PatternOutput::children_only(base, child_names))
    }
}

/// Generate pattern for a field that might be a nested struct.
/// Passes depth through to nested struct patterns.
fn generate_pattern_for_nested_struct(pattern: &Pattern, depth: usize) -> Result<PatternOutput> {
    match pattern {
        Pattern::OpStruct { op, fields, rest } => generate_op_struct_pattern_with_depth(op, fields, *rest, depth),
        // For non-struct patterns, use the regular generator
        _ => generate_pattern(pattern, None),
    }
}

/// Generate pattern for constant: `Const(_)`, `Const(0)`, `@zero`, `@one`
fn generate_const_pattern(const_pat: &ConstPattern) -> Result<PatternOutput> {
    match const_pat {
        ConstPattern::Any => {
            let name = binding_names::CONST;
            let code = quote! { morok_ir::pattern::UPat::cvar(#name) };
            Ok(PatternOutput::self_only(code, Ident::new(binding_names::CONST, proc_macro2::Span::call_site())))
        }
        ConstPattern::Int(0) | ConstPattern::Zero => {
            let name = binding_names::ZERO;
            let code = quote! { morok_ir::pattern::UPat::zero_const(#name) };
            Ok(PatternOutput::self_only(code, Ident::new(binding_names::ZERO, proc_macro2::Span::call_site())))
        }
        ConstPattern::Int(value) => {
            let code = quote! { morok_ir::pattern::UPat::int(#value) };
            // Int patterns (except 0) don't bind - no names
            Ok(PatternOutput::no_names(code))
        }
        ConstPattern::Float(value) => {
            let code = quote! { morok_ir::pattern::UPat::float(#value) };
            // Float patterns don't bind - no names
            Ok(PatternOutput::no_names(code))
        }
        ConstPattern::One => {
            let name = binding_names::ONE;
            let code = quote! { morok_ir::pattern::UPat::one_const(#name) };
            Ok(PatternOutput::self_only(code, Ident::new(binding_names::ONE, proc_macro2::Span::call_site())))
        }
    }
}

/// Generate pattern for operation variable: `op(x, y)` where `op` is an iteration variable.
fn generate_op_var_pattern_with_tracker(
    var_name: &Ident,
    args: &[Pattern],
    iter_ctx: Option<&IterContext>,
    dup_tracker: &mut Option<&mut DuplicateTracker>,
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
    let arg_outputs: Vec<PatternOutput> =
        args.iter().map(|a| generate_pattern_with_tracker(a, iter_ctx, dup_tracker)).collect::<Result<_>>()?;

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
                morok_ir::pattern::UPat::unary(
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
                morok_ir::pattern::UPat::binary(
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
                morok_ir::pattern::UPat::ternary(
                    vec![morok_ir::TernaryOp::#op_ident],
                    vec![#cond, #then_val, #else_val]
                )
            }
        }
    };

    Ok(PatternOutput::children_only(code, child_names))
}

/// Collect identifiers used in an expression.
/// This is used to determine which pattern bindings are actually needed.
fn collect_used_identifiers(expr: &syn::Expr, used: &mut std::collections::HashSet<String>) {
    use syn::visit::Visit;

    struct IdentCollector<'a> {
        used: &'a mut std::collections::HashSet<String>,
    }

    impl<'ast> Visit<'ast> for IdentCollector<'_> {
        fn visit_ident(&mut self, ident: &'ast syn::Ident) {
            self.used.insert(ident.to_string());
        }

        fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
            // For paths like `x` or `Arc::clone`, only collect the first segment
            // if it's a simple identifier (not a type path like `Arc`)
            if node.path.segments.len() == 1 {
                self.used.insert(node.path.segments[0].ident.to_string());
            }
            syn::visit::visit_expr_path(self, node);
        }
    }

    let mut collector = IdentCollector { used };
    collector.visit_expr(expr);
}

/// Collect identifiers used in the RHS and guard expressions.
fn collect_rhs_used_identifiers(rhs: &RewriteExpr, guard: &Option<syn::Expr>) -> std::collections::HashSet<String> {
    let mut used = std::collections::HashSet::new();

    match rhs {
        RewriteExpr::Var(name) => {
            used.insert(name.to_string());
        }
        RewriteExpr::Block(block) => {
            // Visit all statements in the block
            for stmt in &block.block.stmts {
                match stmt {
                    syn::Stmt::Expr(expr, _) => {
                        collect_used_identifiers(expr, &mut used);
                    }
                    syn::Stmt::Local(syn::Local { init: Some(init), .. }) => {
                        collect_used_identifiers(&init.expr, &mut used);
                    }
                    _ => {}
                }
            }
        }
        RewriteExpr::Expr(expr) => {
            collect_used_identifiers(expr, &mut used);
        }
    }

    if let Some(guard_expr) = guard {
        collect_used_identifiers(guard_expr, &mut used);
    }

    used
}

/// Add dependency bindings for ConstWithValue patterns.
/// If `c_val` is used, we need to extract `c` first.
fn add_const_value_dependencies(lhs: &Pattern, used: &mut std::collections::HashSet<String>) {
    let const_value_bindings = collect_const_value_bindings(lhs);
    for (uop_name, value_name) in const_value_bindings {
        // If value_name is used, we need to extract uop_name first
        if used.contains(&value_name.to_string()) {
            used.insert(uop_name.to_string());
        }
    }
}

/// Add dependency bindings for struct field extraction.
/// If any extracted field is used, we need the corresponding struct binding
/// (__struct_root for outer, __nested_N for nested structs).
fn add_struct_field_dependencies(lhs: &Pattern, used: &mut std::collections::HashSet<String>) {
    // Collect all nested struct info and check if any of their fields are used
    let nested_infos = collect_nested_struct_info(lhs, 0);
    for info in nested_infos {
        for (field_name, _) in &info.extractable_fields {
            if used.contains(&field_name.to_string()) {
                // This field is used, so we need its struct binding
                used.insert(info.binding_name.clone());
            }
        }
    }
}

/// Generate variable bindings and rewrite code.
fn generate_rewrite(
    lhs: &Pattern,
    rhs: &RewriteExpr,
    guard: &Option<syn::Expr>,
    arrow: ArrowKind,
    iter_ctx: Option<&IterContext>,
    var_names: &[Ident],
    duplicate_pairs: &[(String, String)],
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

    // Collect identifiers actually used in RHS and guard
    // This is important for alternative patterns where different alternatives
    // may bind different names - we only extract names that are actually used
    let mut used_identifiers = collect_rhs_used_identifiers(rhs, guard);

    // Add dependency bindings: if c_val is used, we need to extract c first
    add_const_value_dependencies(lhs, &mut used_identifiers);

    // Add struct field dependencies: if any field is used, we need __struct_root
    add_struct_field_dependencies(lhs, &mut used_identifiers);

    // Add duplicate pair dependencies: both original and dup names are needed for ptr_eq
    for (orig, dup) in duplicate_pairs {
        used_identifiers.insert(orig.clone());
        used_identifiers.insert(dup.clone());
    }

    // Generate bindings using compile-time indices (no runtime string lookup!)
    // The index matches the order that VarIntern::get_or_insert assigns indices
    // Only generate bindings for names that are actually used in the RHS/guard
    let mut bindings: Vec<TokenStream2> = unique_var_names
        .iter()
        .enumerate()
        .filter(|(_, name)| used_identifiers.contains(&name.to_string()))
        .map(|(idx, name)| {
            let idx_u8 = idx as u8;
            quote! {
                let #name = match __bindings.get_by_index(#idx_u8) {
                    Some(v) => v,
                    None => return morok_ir::pattern::matcher::RewriteResult::NoMatch,
                };
            }
        })
        .collect();

    // Generate auto ptr_eq checks for duplicate variable names
    // e.g., Add(x, x) generates: if !Arc::ptr_eq(&x, &x__dup) { return NoMatch }
    for (orig, dup) in duplicate_pairs {
        let orig_ident = format_ident!("{}", orig);
        let dup_ident = format_ident!("{}", dup);
        bindings.push(quote! {
            if !std::sync::Arc::ptr_eq(#orig_ident, #dup_ident) {
                return morok_ir::pattern::matcher::RewriteResult::NoMatch;
            }
        });
    }

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

    // Generate field extraction for struct patterns (including nested structs)
    let field_extractions = generate_nested_field_extractions(lhs);
    bindings.extend(field_extractions);

    // Generate ConstValue extraction for ConstWithValue patterns
    let const_value_bindings = collect_const_value_bindings(lhs);
    for (uop_name, value_name) in const_value_bindings {
        bindings.push(quote! {
            let #value_name = match #uop_name.op() {
                morok_ir::Op::Const(cv) => cv.0,
                _ => return morok_ir::pattern::matcher::RewriteResult::NoMatch,
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

/// Generate code for `~>` (infallible) - RHS returns Arc<UOp>
fn generate_infallible_rewrite(rhs: &RewriteExpr, guard: &Option<syn::Expr>) -> TokenStream2 {
    let rhs_expr = match rhs {
        RewriteExpr::Var(name) => quote! { std::sync::Arc::clone(#name) },
        RewriteExpr::Expr(expr) => quote! { #expr },
        RewriteExpr::Block(block) => quote! { #block },
    };

    if let Some(guard_expr) = guard {
        quote! {
            if #guard_expr {
                morok_ir::pattern::matcher::RewriteResult::Rewritten(#rhs_expr)
            } else {
                morok_ir::pattern::matcher::RewriteResult::NoMatch
            }
        }
    } else {
        quote! {
            morok_ir::pattern::matcher::RewriteResult::Rewritten(#rhs_expr)
        }
    }
}

/// Generate code for `=>` (fallible) - RHS returns Option<Arc<UOp>>
fn generate_fallible_rewrite(rhs: &RewriteExpr, guard: &Option<syn::Expr>) -> TokenStream2 {
    let rhs_expr = match rhs {
        // For simple variable, wrap in Some() for convenience
        RewriteExpr::Var(name) => quote! { Some(std::sync::Arc::clone(#name)) },
        // For expressions and blocks, wrap in a closure so `?` operator works
        // The closure returns Option<Arc<UOp>>
        RewriteExpr::Expr(expr) => quote! { (|| #expr)() },
        RewriteExpr::Block(block) => quote! { (|| #block)() },
    };

    let conversion = quote! {
        match #rhs_expr {
            Some(__v) => morok_ir::pattern::matcher::RewriteResult::Rewritten(__v),
            None => morok_ir::pattern::matcher::RewriteResult::NoMatch,
        }
    };

    if let Some(guard_expr) = guard {
        quote! {
            if #guard_expr {
                #conversion
            } else {
                morok_ir::pattern::matcher::RewriteResult::NoMatch
            }
        }
    } else {
        conversion
    }
}

/// Operations that support field extraction in struct patterns.
const EXTRACTABLE_OPS: &[&str] = &["Cast", "Permute", "Reduce", "Bufferize", "Reshape", "Expand", "Index", "Copy"];

/// Generate field extraction code for a single field from a struct op.
///
/// The `binding_name` parameter specifies which bound UOp to extract from
/// (e.g., "__struct_root" for outer struct, "__nested_0" for first nested struct).
fn generate_field_extraction_from_binding(binding_name: &str, op_name: &str, field_name: &Ident) -> TokenStream2 {
    let op_ident = format_ident!("{}", op_name);
    quote! {
        let #field_name = {
            let __bound = match __intern.get_index(#binding_name).and_then(|i| __bindings.get_by_index(i)) {
                Some(v) => v,
                None => return morok_ir::pattern::matcher::RewriteResult::NoMatch,
            };
            match __bound.op() {
                morok_ir::Op::#op_ident { #field_name, .. } => #field_name.clone(),
                _ => return morok_ir::pattern::matcher::RewriteResult::NoMatch,
            }
        };
    }
}

/// Generate field extraction code for all nested struct patterns.
///
/// This recursively collects all nested struct patterns and generates extraction
/// code for each level. For example:
/// - `Index { buffer: x, indices }` extracts `indices` from `__struct_root`
/// - `Index { buffer: Bufferize { compute, ranges, .. }, indices }` extracts
///   `indices` from `__struct_root` and `ranges` from `__nested_0`
fn generate_nested_field_extractions(lhs: &Pattern) -> Vec<TokenStream2> {
    let nested_infos = collect_nested_struct_info(lhs, 0);
    let mut extractions = Vec::new();

    for info in nested_infos {
        for (field_name, _var_name) in &info.extractable_fields {
            extractions.push(generate_field_extraction_from_binding(&info.binding_name, &info.op_name, field_name));
        }
    }

    extractions
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
        Pattern::Any(alternatives) => {
            for alt in alternatives {
                collect_const_value_bindings_recursive(alt, bindings);
            }
        }
        Pattern::OpPermute { args, .. } => {
            for arg in args {
                collect_const_value_bindings_recursive(arg, bindings);
            }
        }
        _ => {}
    }
}
