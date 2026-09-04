//! Code generation for the pattern DSL.
//!
//! A block compiles to one closure `Fn(&Arc<UOp>, OpMask, &mut C) -> RewriteResult`
//! that dispatches on the root's dense `OpKey` index with a `match` over constant keys,
//! then tries that kind's rules in source order; each rule first tests its constant
//! early-reject mask against the children mask it is handed. Rules that cannot be keyed
//! statically — wildcard roots, `for op in kind [*]`, `@anyconst` — become sequential
//! steps between the `match`es so source order is preserved exactly.
//!
//! Inside a rule, matching is a chain of `let ... else` destructures, since a Rust
//! pattern cannot cross an `Arc<UOp>` edge. A commutative `Op[a, b]` matches its
//! children lazily per ordering: each ordering is a candidate yielding its bindings,
//! nested commutative nodes become nested candidate loops, and the guard and rewrite
//! body — emitted once at the innermost level — run per candidate until one rewrites,
//! which is Tinygrad's per-permutation retry.

use std::collections::{BTreeMap, HashSet};

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote, quote_spanned};
use syn::{Error, Ident, Pat, Result};

use super::parser::{FieldPat, ForBlock, OpRef, Pattern, PatternItem, PatternList, PatternRule, RewriteExpr};

fn no_match() -> TokenStream2 {
    quote! { __Result::NoMatch }
}

fn op_mask() -> TokenStream2 {
    quote! { __OpMask }
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

/// The for-block a rule belongs to: its variable, grouped variant, and listed ops
/// (`None` for `[*]`). The variable is bound at runtime from the root, so one copy of
/// the rule serves every listed op.
#[derive(Clone, Copy)]
struct Iter<'a> {
    var: &'a Ident,
    kind: &'a Ident,
    ops: Option<&'a [Ident]>,
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
    fn new(fail: Option<TokenStream2>, dup: DuplicateTracker) -> Self {
        Self { code: Vec::new(), dup, bound: Vec::new(), permutes: Vec::new(), fail }
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
                self.code.push(quote! { let __Op::#op = #var.op() else { #fail }; });
            }
            Pattern::Alu { op, args, commutative } => {
                let op_expr = op_expr(op);
                let span = op_span(op);
                if *commutative && args.len() != 2 {
                    return Err(Error::new(span, "commutative pattern takes exactly two arguments"));
                }
                let children: Vec<Ident> = (0..args.len()).map(|i| format_ident!("{var}_{i}")).collect();
                let tuple = quote_spanned! {span=> (#(#children,)*) };
                let destructure = quote! { __alu::AluOp::destructure(#op_expr, #var.op()) };
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
                    let __Op::#op(__ops::#op { #(#field_pats,)* .. }) = #var.op() else { #fail };
                });
                for (pattern, child) in children {
                    self.emit(pattern, &child)?;
                }
            }
            Pattern::Const(pat) => {
                let value = format_ident!("{var}_cv");
                self.code.push(quote! {
                    let __Op::Const(#value) = #var.op() else { #fail };
                    #[allow(irrefutable_let_patterns)]
                    let #pat = #value.0 else { #fail };
                });
                self.bind_verbatim(pat);
            }
            Pattern::Zero => self.code.push(quote! { if !__helpers::is_zero(#var) { #fail } }),
            Pattern::One => self.code.push(quote! { if !__helpers::is_one(#var) { #fail } }),
            Pattern::ConstValue { uop, value } => {
                let cv = format_ident!("{var}_cv");
                self.code.push(quote! {
                    let __Op::Const(#cv) = #var.op() else { #fail };
                    let #value = #cv.0;
                });
                self.bound.push(value.clone());
                self.bind(uop, var);
            }
            Pattern::VConst { uop, values } => {
                let vals = format_ident!("{var}_values");
                self.code.push(quote! {
                    let __Op::VConst(__ops::VConst { values: #vals }) = #var.op() else { #fail };
                    let #values = #vals.clone();
                });
                self.bound.push(values.clone());
                self.bind(uop, var);
            }
            Pattern::AnyConst { uop, values } => {
                self.code.push(quote! {
                    let #values: Vec<svod_ir::ConstValue> = match #var.op() {
                        __Op::Const(cv) => vec![cv.0],
                        __Op::VConst(__ops::VConst { values }) => values.clone(),
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

/// The expression naming an ALU op: a path into `alu`, or the for-block variable.
fn op_expr(op: &OpRef) -> TokenStream2 {
    match op {
        OpRef::Named(ident) => quote! { __alu::#ident },
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

/// An `OpKey` expression: `name` identifies it for merging match arms, `constant` says
/// whether it can seed a `const`.
#[derive(Clone)]
struct Key {
    name: String,
    expr: TokenStream2,
    constant: bool,
}

impl Key {
    fn named(name: &Ident) -> Self {
        Self { name: name.to_string(), expr: quote! { __keys::OpKey::#name }, constant: true }
    }

    fn alu(ident: &Ident) -> Self {
        Self { name: ident.to_string(), expr: quote! { __alu::#ident.op_key() }, constant: true }
    }

    fn mask(&self) -> TokenStream2 {
        let (op_mask, expr) = (op_mask(), &self.expr);
        quote! { #op_mask::of_key(&#expr) }
    }
}

/// What a pattern's root can be dispatched under.
enum Roots<'a> {
    Any,
    /// The ops of a for-block, bound to its variable at runtime.
    Block(Iter<'a>),
    Keys(Vec<Key>),
}

fn roots<'a>(pattern: &Pattern, iter: Option<Iter<'a>>) -> Roots<'a> {
    match pattern {
        Pattern::Wildcard | Pattern::Var(_) => Roots::Any,
        Pattern::Binding { pattern, .. } | Pattern::Some(pattern) => roots(pattern, iter),
        Pattern::Unit(op) | Pattern::Struct { op, .. } => Roots::Keys(vec![Key::named(op)]),
        Pattern::Alu { op: OpRef::Named(op), .. } => Roots::Keys(vec![Key::alu(op)]),
        Pattern::Alu { op: OpRef::Var(var), .. } => match iter {
            Some(iter) if iter.var == var => Roots::Block(iter),
            _ => Roots::Keys(vec![Key { name: var.to_string(), expr: quote! { #var.op_key() }, constant: false }]),
        },
        Pattern::Const(_) | Pattern::Zero | Pattern::One | Pattern::ConstValue { .. } => {
            Roots::Keys(vec![Key::named(&format_ident!("Const"))])
        }
        Pattern::VConst { .. } => Roots::Keys(vec![Key::named(&format_ident!("VConst"))]),
        Pattern::AnyConst { .. } => {
            Roots::Keys(vec![Key::named(&format_ident!("Const")), Key::named(&format_ident!("VConst"))])
        }
    }
}

/// Op kinds the root demands of its direct children: the fixed-position sources that pin
/// exactly one kind (Tinygrad's `UPat.early_reject`, uop/ops.py:1390).
fn early_reject_keys(pattern: &Pattern, iter: Option<Iter<'_>>) -> Vec<Key> {
    let sources: Vec<&Pattern> = match pattern {
        Pattern::Binding { pattern, .. } => return early_reject_keys(pattern, iter),
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
        .filter_map(|source| match roots(source, iter) {
            Roots::Keys(keys) if keys.len() == 1 => keys.into_iter().next(),
            _ => None,
        })
        .filter(|key| seen.insert(key.name.clone()))
        .collect()
}

/// Union of the keys' masks as a single expression.
fn union_mask(keys: &[Key]) -> TokenStream2 {
    let mut masks = keys.iter().map(Key::mask);
    let first = masks.next().unwrap_or_else(|| quote! { __OpMask::EMPTY });
    masks.fold(first, |acc, mask| quote! { #acc.union(#mask) })
}

/// Nested candidate loops for the permutation sites, innermost running `tail`.
///
/// A site's two orderings become lazy candidates yielding the bindings they make plus
/// uniformly named handles to the commutative nodes found inside them, so each nested
/// site is emitted once and iterated in its own loop rather than once per outer
/// ordering. Enumeration order is outer-ordering-major, nested sites before later
/// siblings — the same order as enumerating every full ordering up front.
fn permutation_loops(
    sites: Vec<PermuteSite>,
    dup: DuplicateTracker,
    depth: usize,
    tail: &dyn Fn(&DuplicateTracker) -> TokenStream2,
) -> Result<TokenStream2> {
    let Some((site, rest)) = sites.split_first() else { return Ok(tail(&dup)) };
    let mut candidates = Vec::new();
    let mut inner: Option<(Vec<Ident>, Vec<PermuteSite>, DuplicateTracker)> = None;
    for (first, second) in [(&site.left, &site.right), (&site.right, &site.left)] {
        let mut emitter = Emitter::new(None, dup.clone());
        emitter.emit(&site.args[0], first)?;
        emitter.emit(&site.args[1], second)?;
        let nested: Vec<PermuteSite> = emitter
            .permutes
            .iter()
            .enumerate()
            .map(|(j, nested)| PermuteSite {
                left: format_ident!("__site{depth}_{j}_l"),
                right: format_ident!("__site{depth}_{j}_r"),
                args: nested.args.clone(),
            })
            .collect();
        let handles = emitter.permutes.iter().zip(&nested).map(|(found, uniform)| {
            let (l, r, ul, ur) = (&found.left, &found.right, &uniform.left, &uniform.right);
            quote! { let #ul = #l; let #ur = #r; }
        });
        let mut yielded = emitter.bound.clone();
        yielded.extend(nested.iter().flat_map(|site| [site.left.clone(), site.right.clone()]));
        let code = &emitter.code;
        candidates.push(quote! { __once_with(|| { #(#code)* #(#handles)* Some((#(#yielded,)*)) }) });
        inner.get_or_insert((yielded, nested, emitter.dup));
    }
    let (names, mut inner_sites, inner_dup) = inner.expect("two orderings");
    inner_sites.extend(rest.iter().cloned());
    let body = permutation_loops(inner_sites, inner_dup, depth + 1, tail)?;
    let chain = candidates.into_iter().reduce(|acc, next| quote! { #acc.chain(#next) }).expect("two orderings");
    let orderings = format_ident!("__orderings{depth}");
    Ok(quote! {
        let #orderings = #chain.flatten();
        for (#(#names,)*) in #orderings {
            #body
        }
    })
}

fn rewrite_expr(rhs: &RewriteExpr) -> TokenStream2 {
    let body = match rhs {
        RewriteExpr::Var(name) => quote! { std::sync::Arc::clone(#name) },
        RewriteExpr::Expr(expr) => quote! { #expr },
    };
    quote! { __Into::into_rewrite_result((|| #body)()) }
}

/// Where a compiled rule goes in the block's dispatch sequence.
enum Placement {
    /// Inside the `match` arm of one constant key.
    Keyed(Key),
    /// A sequential step that decides applicability itself.
    Step,
}

struct CompiledRule {
    placement: Placement,
    /// The rule's code, a labeled block that either returns a result or falls through.
    code: TokenStream2,
    /// `const` items the code refers to, by name so blocks can share them.
    consts: Vec<(String, TokenStream2)>,
    /// Constant mask of root kinds the rule can match.
    root: TokenStream2,
    /// Constant early-reject mask, `EMPTY` when it is computed at runtime.
    early_reject: TokenStream2,
}

fn compile_rule(rule: &PatternRule, iter: Option<Iter<'_>>) -> Result<CompiledRule> {
    let (tree, op_mask, no_match) = (format_ident!("__tree"), op_mask(), no_match());
    let label = quote! { '__rule };
    let fail = quote! { break #label; };

    let mut consts = Vec::new();
    let reject_keys = early_reject_keys(&rule.lhs, iter);
    let reject_mask = union_mask(&reject_keys);
    let (reject_check, early_reject) = if reject_keys.is_empty() {
        (quote! {}, quote! { #op_mask::EMPTY })
    } else if reject_keys.iter().all(|key| key.constant) {
        let names: Vec<&str> = reject_keys.iter().map(|key| key.name.as_str()).collect();
        let name = format_ident!("__REJECT_{}", names.join("_"));
        consts.push((name.to_string(), quote! { const #name: #op_mask = #reject_mask; }));
        (quote! { if !#name.is_subset_of(__src_ops) { #fail } }, quote! { #name })
    } else {
        (quote! { if !#reject_mask.is_subset_of(__src_ops) { #fail } }, quote! { #op_mask::EMPTY })
    };

    let mut emitter = Emitter::new(Some(fail.clone()), DuplicateTracker::default());
    emitter.emit(&rule.lhs, &tree)?;
    let shared = &emitter.code;
    let rewrite = rewrite_expr(&rule.rhs);
    let guard = &rule.guard;

    let body = if emitter.permutes.is_empty() {
        let checks = emitter.dup.ptr_eq_checks(&fail);
        let guard = guard.as_ref().map(|guard| quote! { if !(#guard) { #fail } });
        quote! {
            #(#shared)*
            #checks
            #guard
            match #rewrite {
                #no_match => break #label,
                __result => return __result,
            }
        }
    } else {
        let tail = |dup: &DuplicateTracker| {
            let checks = dup.ptr_eq_checks(&quote! { continue; });
            let guard = guard.as_ref().map(|guard| quote! { if !(#guard) { continue; } });
            quote! {
                #checks
                #guard
                match #rewrite {
                    #no_match => continue,
                    __result => return __result,
                }
            }
        };
        let loops = permutation_loops(emitter.permutes.clone(), emitter.dup.clone(), 0, &tail)?;
        quote! {
            #(#shared)*
            #loops
            break #label;
        }
    };

    let (placement, root, wrap): (Placement, TokenStream2, Box<dyn Fn(TokenStream2) -> TokenStream2>) =
        match roots(&rule.lhs, iter) {
            Roots::Any => (Placement::Step, quote! { #op_mask::ALL }, Box::new(|code| code)),
            Roots::Block(Iter { var, kind, ops }) => {
                let (root, member) = match ops {
                    Some(ops) => {
                        let keys: Vec<Key> = ops.iter().map(Key::alu).collect();
                        (union_mask(&keys), quote! { && matches!(*#var, #(__alu::#ops)|*) })
                    }
                    None => {
                        let (base, end) = kind_bounds(kind);
                        (quote! { #op_mask::of_range(#base, #end) }, quote! {})
                    }
                };
                let wrap = move |code| {
                    quote! { if let __Op::#kind(#var, ..) = #tree.op() #member { let #var = *#var; #code } }
                };
                (Placement::Step, root, Box::new(wrap))
            }
            Roots::Keys(keys) if keys.len() == 1 && keys[0].constant => {
                let key = keys.into_iter().next().expect("one key");
                let root = key.mask();
                (Placement::Keyed(key), root, Box::new(|code| code))
            }
            Roots::Keys(keys) => {
                let root = union_mask(&keys);
                let tests = keys.iter().map(|key| {
                    let expr = &key.expr;
                    quote! { __key == #expr.index() }
                });
                let test = quote! { #(#tests)||* };
                (Placement::Step, root, Box::new(move |code| quote! { if #test { #code } }))
            }
        };

    let code = wrap(quote! {
        #label: {
            #reject_check
            #body
        }
    });
    Ok(CompiledRule { placement, code, consts, root, early_reject })
}

fn kind_bounds(kind: &Ident) -> (TokenStream2, TokenStream2) {
    let upper = kind.to_string().to_uppercase();
    let (base, end) = (format_ident!("OP_KEY_BASE_{upper}"), format_ident!("OP_KEY_END_{upper}"));
    (quote! { __keys::#base }, quote! { __keys::#end })
}

/// Compile every rule of a block in source order, expanding for-blocks.
fn compile_block(patterns: &PatternList) -> Result<Vec<CompiledRule>> {
    let mut compiled = Vec::new();
    let mut push = |rule: &PatternRule, iter: Option<Iter<'_>>| -> Result<()> {
        compiled.push(compile_rule(rule, iter)?);
        Ok(())
    };
    for item in &patterns.items {
        match item {
            PatternItem::Rule(rule) => push(rule, None)?,
            PatternItem::ForBlock(ForBlock { var, kind, ops, body }) => {
                for rule in body {
                    push(rule, Some(Iter { var, kind, ops: ops.as_deref() }))?;
                }
            }
        }
    }
    Ok(compiled)
}

/// The `[T; N]` check that every listed op of a for-block is of the block's kind.
fn for_block_checks(patterns: &PatternList) -> Vec<TokenStream2> {
    patterns
        .items
        .iter()
        .filter_map(|item| match item {
            PatternItem::ForBlock(ForBlock { kind, ops: Some(ops), .. }) => {
                let count = ops.len();
                Some(quote! { const _: [__alu::#kind; #count] = [#(__alu::#ops),*]; })
            }
            _ => None,
        })
        .collect()
}

/// Generate a `SimplifiedPatternMatcher` holding this block as one segment.
pub fn generate_simplified_pattern_matcher(patterns: &PatternList) -> Result<TokenStream2> {
    let has_context = patterns.context_type.is_some();
    let ctx_type = patterns.context_type.as_ref().map_or_else(|| quote! { () }, |ty| quote! { #ty });
    let op_mask = op_mask();
    let rules = compile_block(patterns)?;

    // Consecutive keyed rules share one `match`; every other rule is its own step.
    let mut key_consts: BTreeMap<String, TokenStream2> = BTreeMap::new();
    let mut steps = Vec::new();
    let mut arms: Vec<(String, Vec<TokenStream2>)> = Vec::new();
    let flush = |arms: &mut Vec<(String, Vec<TokenStream2>)>, steps: &mut Vec<TokenStream2>| {
        if arms.is_empty() {
            return;
        }
        let branches = arms.drain(..).map(|(name, codes)| {
            let key = format_ident!("__KEY_{name}");
            quote! { #key => { #(#codes)* } }
        });
        steps.push(quote! { match __key { #(#branches)* _ => {} } });
    };
    for rule in &rules {
        match &rule.placement {
            Placement::Keyed(key) => {
                let expr = &key.expr;
                key_consts.entry(key.name.clone()).or_insert_with(|| {
                    let name = format_ident!("__KEY_{}", key.name);
                    quote! { const #name: usize = #expr.index(); }
                });
                match arms.iter_mut().find(|(name, _)| *name == key.name) {
                    Some((_, codes)) => codes.push(rule.code.clone()),
                    None => arms.push((key.name.clone(), vec![rule.code.clone()])),
                }
            }
            Placement::Step => {
                flush(&mut arms, &mut steps);
                steps.push(rule.code.clone());
            }
        }
    }
    flush(&mut arms, &mut steps);

    let uses_key = !key_consts.is_empty() || rules.iter().any(|rule| rule.code.to_string().contains("__key"));
    let key_binding = uses_key.then(|| quote! { let __key = __keys::OpKey::from_op(__tree.op()).index(); });
    let key_consts = key_consts.into_values();
    let rule_consts: BTreeMap<&String, &TokenStream2> =
        rules.iter().flat_map(|rule| &rule.consts).map(|(name, item)| (name, item)).collect();
    let rule_consts = rule_consts.into_values();
    let table = rules.iter().map(|rule| {
        let (root, reject) = (&rule.root, &rule.early_reject);
        quote! { (#root, #reject) }
    });
    let checks = for_block_checks(patterns);
    let ctx_param = if has_context {
        quote! { ctx: &mut _ }
    } else {
        quote! { _ctx: &mut () }
    };
    let src_ops = if rules.iter().any(|rule| rule.code.to_string().contains("__src_ops")) {
        quote! { __src_ops }
    } else {
        quote! { _src_ops }
    };
    let no_match = no_match();
    Ok(quote! {
        {
            use svod_ir::op::{OpMask as __OpMask, alu as __alu, pattern_derived as __keys};
            use svod_ir::pattern::{IntoRewriteResult as __Into, RewriteResult as __Result, helpers as __helpers};
            use std::iter::once_with as __once_with;
            use svod_ir::{Op as __Op, ops as __ops};
            #(#checks)*
            #(#key_consts)*
            #(#rule_consts)*
            const __RULES: &[(#op_mask, #op_mask)] = &[#(#table),*];
            let mut __matcher = svod_ir::pattern::SimplifiedPatternMatcher::<#ctx_type>::new();
            __matcher.add_block(
                __RULES,
                move |__tree: &std::sync::Arc<svod_ir::UOp>, #src_ops: #op_mask, #ctx_param| {
                    #key_binding
                    #(#steps)*
                    #no_match
                },
            );
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
