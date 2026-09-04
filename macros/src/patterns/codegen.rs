//! Code generation for the pattern DSL.
//!
//! Every rule becomes one closure registered under its root `OpKey`s. Matching is a
//! chain of `let ... else` destructures, since a Rust pattern cannot cross an
//! `Arc<UOp>` edge. A commutative `Op[a, b]` splits the chain: everything outside the
//! permuted subtrees is matched once, each ordering of the permuted subtrees becomes a
//! lazy candidate yielding its bindings, and the guard and rewrite body — emitted once —
//! run per candidate until one rewrites, which is Tinygrad's per-permutation retry.

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote, quote_spanned};
use syn::{Error, Ident, Pat, Result};

use super::parser::{FieldPat, ForBlock, OpRef, Pattern, PatternItem, PatternList, PatternRule, RewriteExpr};

fn no_match() -> TokenStream2 {
    quote! { svod_ir::pattern::RewriteResult::NoMatch }
}

/// Renames repeated binding names (`Add(x, x)`) so every occurrence is bound and the
/// pairs can be checked with `Arc::ptr_eq`.
#[derive(Default, Clone)]
struct DuplicateTracker {
    seen: HashSet<String>,
    duplicates: Vec<(Ident, Ident)>,
}

impl DuplicateTracker {
    fn bind(&mut self, name: &Ident) -> Ident {
        let key = name.to_string();
        if !self.seen.insert(key.clone()) {
            let count = self.duplicates.iter().filter(|(original, _)| original == name).count();
            let dup = if count == 0 { format_ident!("{key}_dup") } else { format_ident!("{key}_dup_{}", count + 1) };
            self.duplicates.push((name.clone(), dup.clone()));
            return dup;
        }
        name.clone()
    }

    fn ptr_eq_checks(&self, fail: &TokenStream2) -> TokenStream2 {
        let checks = self.duplicates.iter().map(|(original, dup)| {
            quote! { if !std::sync::Arc::ptr_eq(#original, #dup) { #fail } }
        });
        quote! { #(#checks)* }
    }
}

/// A commutative node whose children are matched per ordering.
#[derive(Clone)]
struct PermuteSite {
    left: Ident,
    right: Ident,
    args: Vec<Pattern>,
}

/// Emits the match code of one pattern into a straight-line chain, deferring the
/// children of commutative nodes to [`PermuteSite`]s.
struct Emitter {
    code: Vec<TokenStream2>,
    dup: DuplicateTracker,
    /// Names bound so far, in order — the tuple an ordering candidate yields.
    bound: Vec<Ident>,
    permutes: Vec<PermuteSite>,
    /// Statement run when a check fails; `None` means the chain is inside an
    /// `Option`-returning closure and a failed `Option` destructure uses `?`.
    fail: Option<TokenStream2>,
}

impl Emitter {
    fn new(fail: Option<TokenStream2>, dup: DuplicateTracker, bound: Vec<Ident>) -> Self {
        Self { code: Vec::new(), dup, bound, permutes: Vec::new(), fail }
    }

    fn fail(&self) -> TokenStream2 {
        self.fail.clone().unwrap_or_else(|| quote! { return None; })
    }

    /// `let #pat = #option else { fail }`, as `?` where the closure returns an `Option`.
    fn unwrap_or_fail(&self, pat: TokenStream2, option: TokenStream2) -> TokenStream2 {
        match &self.fail {
            Some(fail) => quote! { let Some(#pat) = #option else { #fail }; },
            None => quote! { let #pat = #option?; },
        }
    }

    fn bind(&mut self, name: &Ident, value: &Ident) {
        let name = self.dup.bind(name);
        self.code.push(quote! { let #name = #value; });
        self.bound.push(name);
    }

    fn emit(&mut self, pattern: &Pattern, var: &Ident) -> Result<()> {
        let fail = self.fail();
        match pattern {
            Pattern::Wildcard => {}
            Pattern::Var(name) => self.bind(name, var),
            Pattern::Binding { name, pattern } => {
                self.emit(pattern, var)?;
                self.bind(name, var);
            }
            Pattern::Unit(op) => {
                self.code.push(quote! { let svod_ir::Op::#op = #var.op() else { #fail }; });
            }
            Pattern::Alu { op, args, commutative } => {
                let op_expr = op_expr(op);
                let span = op_span(op);
                if *commutative && args.len() != 2 {
                    return Err(Error::new(span, "commutative pattern takes exactly two arguments"));
                }
                let children: Vec<Ident> = (0..args.len()).map(|i| format_ident!("{var}_{i}")).collect();
                let tuple = quote_spanned! {span=> (#(#children,)*) };
                let destructure = quote! { svod_ir::op::alu::AluOp::destructure(#op_expr, #var.op()) };
                self.code.push(self.unwrap_or_fail(tuple, destructure));
                if *commutative {
                    let (left, right) = (children[0].clone(), children[1].clone());
                    self.permutes.push(PermuteSite { left, right, args: args.clone() });
                } else {
                    for (arg, child) in args.iter().zip(&children) {
                        self.emit(arg, child)?;
                    }
                }
            }
            Pattern::Struct { op, fields } => {
                let mut field_pats = Vec::new();
                let mut children = Vec::new();
                for field in fields {
                    let name = &field.name;
                    match &field.pattern {
                        FieldPat::Child(pattern) => {
                            let child = format_ident!("{var}_{name}");
                            field_pats.push(quote! { #name: #child });
                            children.push((pattern, child));
                        }
                        FieldPat::Verbatim(pat) => {
                            field_pats.push(quote! { #name: #pat });
                            self.bind_verbatim(pat);
                        }
                    }
                }
                self.code.push(quote! {
                    let svod_ir::Op::#op(svod_ir::ops::#op { #(#field_pats,)* .. }) = #var.op() else { #fail };
                });
                for (pattern, child) in children {
                    self.emit(pattern, &child)?;
                }
            }
            Pattern::Const(pat) => {
                let value = format_ident!("{var}_cv");
                self.code.push(quote! {
                    let svod_ir::Op::Const(#value) = #var.op() else { #fail };
                    #[allow(irrefutable_let_patterns)]
                    let #pat = #value.0 else { #fail };
                });
                self.bind_verbatim(pat);
            }
            Pattern::Zero => self.code.push(quote! { if !svod_ir::pattern::helpers::is_zero(#var) { #fail } }),
            Pattern::One => self.code.push(quote! { if !svod_ir::pattern::helpers::is_one(#var) { #fail } }),
            Pattern::ConstValue { uop, value } => {
                let cv = format_ident!("{var}_cv");
                self.code.push(quote! {
                    let svod_ir::Op::Const(#cv) = #var.op() else { #fail };
                    let #value = #cv.0;
                });
                self.bound.push(value.clone());
                self.bind(uop, var);
            }
            Pattern::VConst { uop, values } => {
                let vals = format_ident!("{var}_values");
                self.code.push(quote! {
                    let svod_ir::Op::VConst(svod_ir::ops::VConst { values: #vals }) = #var.op() else { #fail };
                    let #values = #vals.clone();
                });
                self.bound.push(values.clone());
                self.bind(uop, var);
            }
            Pattern::AnyConst { uop, values } => {
                self.code.push(quote! {
                    let #values: Vec<svod_ir::ConstValue> = match #var.op() {
                        svod_ir::Op::Const(cv) => vec![cv.0],
                        svod_ir::Op::VConst(svod_ir::ops::VConst { values }) => values.clone(),
                        _ => { #fail }
                    };
                });
                self.bound.push(values.clone());
                self.bind(uop, var);
            }
            Pattern::Some(inner) => {
                let some = format_ident!("{var}_some");
                self.code.push(self.unwrap_or_fail(quote! { #some }, quote! { #var.as_ref() }));
                self.emit(inner, &some)?;
            }
        }
        Ok(())
    }

    /// Register the names a verbatim Rust pattern binds, so they reach the rewrite body
    /// through an ordering tuple and clash with DSL bindings instead of shadowing them.
    fn bind_verbatim(&mut self, pat: &Pat) {
        let mut names = Vec::new();
        verbatim_bindings(pat, &mut names);
        for name in names {
            self.dup.seen.insert(name.to_string());
            self.bound.push(name);
        }
    }
}

fn op_expr(op: &OpRef) -> TokenStream2 {
    match op {
        OpRef::Named(ident) => quote! { svod_ir::op::alu::#ident },
        OpRef::Var(ident) => quote! { #ident },
    }
}

fn op_span(op: &OpRef) -> Span {
    match op {
        OpRef::Named(ident) | OpRef::Var(ident) => ident.span(),
    }
}

/// Snake-case identifiers bound by a Rust pattern; `Pat::Or` branches bind the same set.
fn verbatim_bindings(pat: &Pat, out: &mut Vec<Ident>) {
    match pat {
        Pat::Ident(ident) => {
            if ident.ident.to_string().starts_with(|c: char| c.is_lowercase() || c == '_') {
                out.push(ident.ident.clone());
            }
            if let Some((_, sub)) = &ident.subpat {
                verbatim_bindings(sub, out);
            }
        }
        Pat::Or(or) => {
            if let Some(first) = or.cases.first() {
                verbatim_bindings(first, out);
            }
        }
        Pat::Paren(paren) => verbatim_bindings(&paren.pat, out),
        Pat::Reference(reference) => verbatim_bindings(&reference.pat, out),
        Pat::Type(typed) => verbatim_bindings(&typed.pat, out),
        Pat::Tuple(tuple) => tuple.elems.iter().for_each(|p| verbatim_bindings(p, out)),
        Pat::TupleStruct(tuple) => tuple.elems.iter().for_each(|p| verbatim_bindings(p, out)),
        Pat::Slice(slice) => slice.elems.iter().for_each(|p| verbatim_bindings(p, out)),
        Pat::Struct(strukt) => strukt.fields.iter().for_each(|f| verbatim_bindings(&f.pat, out)),
        _ => {}
    }
}

/// `OpKey`s a pattern's root can be dispatched under; empty means any op.
fn root_keys(pattern: &Pattern) -> Vec<TokenStream2> {
    let key = |name: &Ident| quote! { svod_ir::op::pattern_derived::OpKey::#name };
    match pattern {
        Pattern::Wildcard | Pattern::Var(_) => vec![],
        Pattern::Binding { pattern, .. } | Pattern::Some(pattern) => root_keys(pattern),
        Pattern::Unit(op) | Pattern::Struct { op, .. } => vec![key(op)],
        Pattern::Alu { op, .. } => {
            let op = op_expr(op);
            vec![quote! { svod_ir::op::alu::AluOp::key(#op) }]
        }
        Pattern::Const(_) | Pattern::Zero | Pattern::One | Pattern::ConstValue { .. } => {
            vec![key(&format_ident!("Const"))]
        }
        Pattern::VConst { .. } => vec![key(&format_ident!("VConst"))],
        Pattern::AnyConst { .. } => vec![key(&format_ident!("Const")), key(&format_ident!("VConst"))],
    }
}

/// Op kinds the root demands of its direct children: the fixed-position sources that pin
/// exactly one kind (Tinygrad's `UPat.early_reject`, uop/ops.py:1390).
fn early_reject_keys(pattern: &Pattern) -> Vec<TokenStream2> {
    let sources: Vec<&Pattern> = match pattern {
        Pattern::Binding { pattern, .. } => return early_reject_keys(pattern),
        Pattern::Alu { args, .. } => args.iter().collect(),
        Pattern::Struct { fields, .. } => fields
            .iter()
            .filter_map(|field| match &field.pattern {
                FieldPat::Child(pattern) => Some(pattern),
                FieldPat::Verbatim(_) => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let mut seen = HashSet::new();
    sources
        .into_iter()
        .map(root_keys)
        .filter(|keys| keys.len() == 1)
        .map(|keys| keys.into_iter().next().expect("one key"))
        .filter(|key| seen.insert(key.to_string()))
        .collect()
}

/// One complete ordering of every commutative node: its extra match code and bindings.
struct Candidate {
    code: Vec<TokenStream2>,
    bound: Vec<Ident>,
    dup: DuplicateTracker,
}

/// Expand the permutation sites depth-first: a site's two orderings are tried in source
/// order, its nested sites before the sites that follow it.
fn expand(prefix: Candidate, sites: &[PermuteSite]) -> Result<Vec<Candidate>> {
    let Some((site, rest)) = sites.split_first() else { return Ok(vec![prefix]) };
    let mut candidates = Vec::new();
    for (first, second) in [(&site.left, &site.right), (&site.right, &site.left)] {
        let mut emitter = Emitter::new(None, prefix.dup.clone(), prefix.bound.clone());
        emitter.emit(&site.args[0], first)?;
        emitter.emit(&site.args[1], second)?;
        let mut code = prefix.code.clone();
        code.append(&mut emitter.code);
        let mut queue = emitter.permutes;
        queue.extend(rest.iter().cloned());
        candidates.extend(expand(Candidate { code, bound: emitter.bound, dup: emitter.dup }, &queue)?);
    }
    Ok(candidates)
}

fn rewrite_expr(rhs: &RewriteExpr) -> TokenStream2 {
    let body = match rhs {
        RewriteExpr::Var(name) => quote! { std::sync::Arc::clone(#name) },
        RewriteExpr::Closure(closure) => {
            let body = &closure.body;
            quote! { #body }
        }
        RewriteExpr::Expr(expr) => quote! { #expr },
    };
    quote! { svod_ir::pattern::IntoRewriteResult::into_rewrite_result((|| #body)()) }
}

fn validate_closure_params(rule: &PatternRule, has_context: bool) -> Result<()> {
    let RewriteExpr::Closure(closure) = &rule.rhs else { return Ok(()) };
    for param in &closure.inputs {
        if !has_context && matches!(param, Pat::Ident(ident) if ident.ident == "ctx") {
            return Err(Error::new_spanned(
                param,
                "`ctx` requires a `@context Type;` declaration at the start of patterns!",
            ));
        }
    }
    Ok(())
}

fn generate_rule(rule: &PatternRule, has_context: bool) -> Result<TokenStream2> {
    validate_closure_params(rule, has_context)?;
    let tree = format_ident!("__tree");
    let no_match = no_match();
    let mut emitter = Emitter::new(Some(quote! { return #no_match; }), DuplicateTracker::default(), Vec::new());
    emitter.emit(&rule.lhs, &tree)?;

    let rewrite = rewrite_expr(&rule.rhs);
    let shared = &emitter.code;
    let body = if emitter.permutes.is_empty() {
        let checks = emitter.dup.ptr_eq_checks(&quote! { return #no_match; });
        let tail = match &rule.guard {
            Some(guard) => quote! { if #guard { #rewrite } else { #no_match } },
            None => rewrite,
        };
        quote! { #(#shared)* #checks #tail }
    } else {
        let candidates =
            expand(Candidate { code: Vec::new(), bound: Vec::new(), dup: emitter.dup.clone() }, &emitter.permutes)?;
        let names = &candidates[0].bound;
        debug_assert!(candidates.iter().all(|c| c.bound == *names), "orderings bind the same names");
        let closures = candidates.iter().map(|candidate| {
            let code = &candidate.code;
            quote! { std::iter::once_with(|| { #(#code)* Some((#(#names,)*)) }) }
        });
        let chain = closures.reduce(|acc, next| quote! { #acc.chain(#next) }).expect("at least one ordering");
        let checks = candidates[0].dup.ptr_eq_checks(&quote! { continue; });
        let guard = rule.guard.as_ref().map(|guard| quote! { if !(#guard) { continue; } });
        quote! {
            #(#shared)*
            let __orderings = #chain.flatten();
            for (#(#names,)*) in __orderings {
                #checks
                #guard
                match #rewrite {
                    svod_ir::pattern::RewriteResult::NoMatch => continue,
                    __result => return __result,
                }
            }
            #no_match
        }
    };

    let ctx_param = if has_context {
        quote! { ctx: &mut _ }
    } else {
        quote! { _ctx: &mut () }
    };
    let closure = quote! { move |#tree: &std::sync::Arc<svod_ir::UOp>, #ctx_param| { #body } };
    let keys = root_keys(&rule.lhs);
    if keys.is_empty() {
        return Ok(quote! { __matcher.add_wildcard(#closure); });
    }
    let reject = early_reject_keys(&rule.lhs);
    Ok(quote! { __matcher.add_rejecting(&[#(#keys),*], &[#(#reject),*], #closure); })
}

fn generate_for_block(block: &ForBlock, has_context: bool) -> Result<TokenStream2> {
    let (var, kind) = (&block.var, &block.kind);
    let ops = match &block.ops {
        Some(ops) => {
            let count = ops.len();
            quote! { { let __ops: [svod_ir::op::alu::#kind; #count] = [#(svod_ir::op::alu::#ops),*]; __ops } }
        }
        None => quote! { <svod_ir::op::alu::#kind as svod_ir::op::alu::AluOp>::ALL.iter().copied() },
    };
    let rules = block.body.iter().map(|rule| generate_rule(rule, has_context)).collect::<Result<Vec<_>>>()?;
    Ok(quote! { for #var in #ops { #(#rules)* } })
}

/// Generate a `SimplifiedPatternMatcher` from the parsed pattern list.
pub fn generate_simplified_pattern_matcher(patterns: &PatternList) -> Result<TokenStream2> {
    let has_context = patterns.context_type.is_some();
    let ctx_type = patterns.context_type.as_ref().map_or_else(|| quote! { () }, |ty| quote! { #ty });
    let items = patterns
        .items
        .iter()
        .map(|item| match item {
            PatternItem::Rule(rule) => generate_rule(rule, has_context),
            PatternItem::ForBlock(block) => generate_for_block(block, has_context),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(quote! {
        {
            let mut __matcher = svod_ir::pattern::SimplifiedPatternMatcher::<#ctx_type>::new();
            #(#items)*
            __matcher
        }
    })
}

/// Like [`generate_simplified_pattern_matcher`], but the matcher is built once into a
/// `LazyLock` and borrowed as `&'static`.
pub fn generate_cached_pattern_matcher(patterns: &PatternList) -> Result<TokenStream2> {
    let inner = generate_simplified_pattern_matcher(patterns)?;
    let ctx_type = patterns.context_type.as_ref().map_or_else(|| quote! { () }, |ty| quote! { #ty });
    Ok(quote! {
        {
            static __CACHED: std::sync::LazyLock<svod_ir::pattern::SimplifiedPatternMatcher<#ctx_type>> =
                std::sync::LazyLock::new(|| #inner);
            &*__CACHED
        }
    })
}
